// k19-gramdev: a request with a grammar (JSON schema, a lazy tool-call grammar that triggers mid-reply or never, a grammar that
// rejects most tokens) or a reasoning budget samples the same tokens from the device top-k candidates (common_sampler_device_topk,
// the raw rows kept on the host by llama_set_sampler_keep_logits) as from the full rows (today's host path), greedy and sampled
// (Studio presets), one row per pass and in multi-row verify windows (common_sampler_sample_and_accept_n and the rejection sampler).
// The device top-k runs on the CPU backend here (a backend sampler chain), with a small k_dev so that the full-row fallback is taken.
//
// The model's tokenizer is replaced by an SPM vocabulary of real text pieces (single characters, JSON pieces, <tool_call>, <think>,
// words), written next to the input model, so the grammars see text. Its F32 / F16 weights are drawn again from a fixed seed
// (DTOPK_TEST_SEED, default below), as test-llama-archs draws them: the generate-models fixture seeds from std::random_device, so
// without this every ctest run tested another model, and on some the greedy reply had no token to trigger a lazy grammar mid-reply.
//
// Usage: test-dtopk-grammar <model.gguf (vocab >= 1024)> [--expect-diff]
//   --expect-diff: run with LLAMA_DEVICE_TOPK_GRAMMAR_NOFALLBACK=1 (the candidates decide every position, inexact): passes only if
//   at least one case differs from the full-row run, i.e. the comparison can fail

#include "common.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include "gguf.h"
#include "ggml.h"

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const std::string & name, bool ok) {
    printf("%-100s %s\n", name.c_str(), ok ? "OK" : "FAIL");
    fflush(stdout);
    if (!ok) {
        g_fail = 1;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the SPM vocabulary
static std::vector<std::string> make_vocab(uint32_t n_vocab) {
    std::vector<std::string> v;
    std::set<std::string> seen;
    auto add = [&](const std::string & s) {
        if (v.size() < n_vocab && seen.insert(s).second) {
            v.push_back(s);
        }
    };
    add("<unk>");
    add("<s>");
    add("</s>");
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", b);
        add(buf);
    }
    add("\xe2\x96\x81"); // the SPM space
    for (int c = 0x21; c < 0x7f; ++c) {
        add(std::string(1, (char) c));
    }
    for (const char * s : { "<tool_call>", "</tool_call>", "<think>", "</think>", "{\"", "\":", "\",", "\"}", "\":\"", "\xe2\x96\x81{",
                            "}}", "name", "arguments", "city", "days", "get_weather", "search_web", "run_python", "true", "false",
                            "null", "\":\xe2\x96\x81", ",\xe2\x96\x81\"", "\xe2\x96\x81\"", "\"name\"", "\"arguments\"", "12", "34", "56",
                            "78", "90", "100" }) {
        add(s);
    }
    const char * cons = "bcdfghklmnprstvwz";
    const char * vow  = "aeiou";
    for (int len = 2; v.size() < n_vocab && len <= 4; ++len) {
        for (int i = 0; v.size() < n_vocab && i < 17*5*17*5; ++i) {
            std::string w;
            int x = i;
            for (int j = 0; j < len; ++j) {
                if (j % 2 == 0) {
                    w += cons[x % 17];
                    x /= 17;
                } else {
                    w += vow[x % 5];
                    x /= 5;
                }
            }
            add(w);
            add("\xe2\x96\x81" + w);
        }
    }
    for (int i = 0; v.size() < n_vocab; ++i) {
        add("x" + std::to_string(i));
    }
    return v;
}

// a seed with which every case has its setup (a token to trigger a lazy grammar mid-reply in each preset and mode, a reply that
// differs under LLAMA_DEVICE_TOPK_GRAMMAR_NOFALLBACK=1)
#define DTOPK_SEED_DEFAULT 1234

static bool write_spm_model(const char * in, const std::string & out, uint32_t * n_vocab_out) {
    ggml_context * gctx = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ false, /*ctx =*/ &gctx };
    gguf_context * src = gguf_init_from_file(in, ip);
    if (src == nullptr) {
        return false;
    }
    const int64_t kid = gguf_find_key(src, "tokenizer.ggml.tokens");
    if (kid < 0) {
        return false;
    }
    const uint32_t n_vocab = (uint32_t) gguf_get_arr_n(src, kid);
    *n_vocab_out = n_vocab;

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);

    const std::vector<std::string> toks = make_vocab(n_vocab);
    std::vector<const char *> tp;
    std::vector<int32_t> types;
    std::vector<float> scores(n_vocab, 0.0f);
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tp.push_back(toks[i].c_str());
        types.push_back(i == 0 ? 2 : i <= 2 ? 3 : i < 3 + 256 ? 6 : 1); // unknown, control, byte, normal
        scores[i] = -(float) i;
    }
    gguf_set_val_str(dst, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(dst, "tokenizer.ggml.tokens", tp.data(), tp.size());
    gguf_set_arr_data(dst, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_data(dst, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_val_u32(dst, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(dst, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(dst, "tokenizer.ggml.eos_token_id", 2);

    // the weights: normal(0, 0.1) keyed on the tensor name, as test-llama-archs fill_normal, from a fixed seed (its property fillers,
    // ssm_a / _resid_, keep the fixture's values: the architectures used here have none)
    const char * es = getenv("DTOPK_TEST_SEED");
    const size_t seed = es != nullptr ? (size_t) strtoull(es, nullptr, 10) : (size_t) DTOPK_SEED_DEFAULT;
    for (int64_t i = 0; i < gguf_get_n_tensors(src); ++i) {
        ggml_tensor * t = ggml_get_tensor(gctx, gguf_get_tensor_name(src, i));
        if ((t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && strstr(t->name, "ssm_a") == nullptr && strstr(t->name, "_resid_") == nullptr) {
            std::mt19937 gen(seed ^ std::hash<std::string>{}(t->name));
            std::normal_distribution<float> dis(0.0f, 0.1f);
            const int64_t n = ggml_nelements(t);
            for (int64_t j = 0; j < n; ++j) {
                const float v = dis(gen);
                if (t->type == GGML_TYPE_F32) {
                    ((float *) t->data)[j] = v;
                } else {
                    ((ggml_fp16_t *) t->data)[j] = ggml_fp32_to_fp16(v);
                }
            }
        }
        gguf_add_tensor(dst, t);
    }
    const bool ok = gguf_write_to_file(dst, out.c_str(), false);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(gctx);
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------------------------

static llama_model * g_model = nullptr;
static const llama_vocab * g_vocab = nullptr;
static int32_t g_n_vocab = 0;

static llama_token tok_of(const std::string & s) {
    for (llama_token t = 0; t < g_n_vocab; ++t) {
        if (common_token_to_piece(g_vocab, t, true) == s) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static std::string gbnf_lit(const std::string & s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            r += '\\';
            r += c;
        } else if (c == '\n') {
            r += "\\n";
        } else {
            r += c;
        }
    }
    return r + "\"";
}

struct run_result {
    std::vector<llama_token> toks;
    common_sampler_dev_stats st;
    bool threw = false;
    std::string err;
};

enum run_mode { MODE_ROW, MODE_VERIFY, MODE_VERIFY_REJ };

struct run_cfg {
    common_params_sampling sp;
    int n_gen = 48;
    run_mode mode = MODE_ROW;
    int32_t k_dev = 64;
    llama_token pre_accept = LLAMA_TOKEN_NULL; // accepted before the first sample (e.g. <think> to start the reasoning budget)
    const std::vector<llama_token> * ref = nullptr; // the tokens the drafts are taken from
};

// device = false: today's host path on the full rows; true: the device top-k with the raw rows kept on the host
static run_result run(const run_cfg & cfg, const std::vector<llama_token> & prompt, bool device) {
    run_result res;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 512;
    cp.n_batch   = 64;
    cp.n_ubatch  = 64;
    cp.n_outputs_max = 64;
    cp.n_outputs_max_per_seq = 8; // verify windows of up to 5 rows with the backend sampler
    cp.n_threads = std::max(1, std::min(4, (int) std::thread::hardware_concurrency())); // no more threads than cores
    cp.n_threads_batch = cp.n_threads;
    llama_context * ctx = llama_init_from_model(g_model, cp);
    if (ctx == nullptr) {
        res.threw = true;
        res.err = "no context";
        return res;
    }
    common_params_sampling sp = cfg.sp;
    common_sampler * smpl = nullptr;
    llama_sampler * dev = nullptr;
    try {
        smpl = common_sampler_init(g_model, sp);
        if (device) {
            if (common_sampler_device_topk(smpl, cfg.k_dev) == 0) {
                throw std::runtime_error("request does not qualify for the device top-k");
            }
            dev = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(dev, llama_sampler_init_top_k(cfg.k_dev));
            if (!llama_set_sampler(ctx, 0, dev)) {
                throw std::runtime_error("llama_set_sampler failed");
            }
            llama_set_sampler_keep_logits(ctx, 0, common_sampler_device_topk_needs_raw(smpl));
            common_sampler_set_device_topk(smpl, true);
        }
        llama_batch b = llama_batch_init(64, 0, 1);
        auto push = [&](llama_token t, llama_pos p, bool out) {
            b.token[b.n_tokens] = t;
            b.pos[b.n_tokens] = p;
            b.n_seq_id[b.n_tokens] = 1;
            b.seq_id[b.n_tokens][0] = 0;
            b.logits[b.n_tokens] = out;
            b.n_tokens++;
        };
        b.n_tokens = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            push(prompt[i], (llama_pos) i, i + 1 == prompt.size());
        }
        if (llama_decode(ctx, b) != 0) {
            throw std::runtime_error("prompt decode failed");
        }
        llama_pos pos = (llama_pos) prompt.size();
        if (cfg.pre_accept != LLAMA_TOKEN_NULL) {
            common_sampler_accept(smpl, cfg.pre_accept, true);
        }
        llama_token id = common_sampler_sample(smpl, ctx, b.n_tokens - 1);
        common_sampler_accept(smpl, id, true);
        res.toks.push_back(id);
        int step = 0;
        while ((int) res.toks.size() < cfg.n_gen) {
            b.n_tokens = 0;
            if (cfg.mode == MODE_ROW) {
                push(id, pos, true);
                if (llama_decode(ctx, b) != 0) {
                    throw std::runtime_error("decode failed");
                }
                id = common_sampler_sample(smpl, ctx, 0);
                common_sampler_accept(smpl, id, true);
                res.toks.push_back(id);
                pos += 1;
                continue;
            }
            // a verify window: the last token + d drafts, mostly the reference continuation, some of them wrong
            const int d = 1 + (step % 4);
            llama_tokens draft;
            const size_t at = res.toks.size();
            for (int j = 0; j < d; ++j) {
                llama_token t = cfg.ref && at + j < cfg.ref->size() ? (*cfg.ref)[at + j] : (llama_token) ((at * 31 + j * 7) % g_n_vocab);
                if (((at + j) * 13 + step) % 5 == 0) {
                    t = (llama_token) ((t + 17) % g_n_vocab);
                }
                draft.push_back(t);
            }
            push(id, pos, true);
            for (int j = 0; j < d; ++j) {
                push(draft[j], pos + 1 + j, true);
            }
            if (llama_decode(ctx, b) != 0) {
                throw std::runtime_error("verify decode failed");
            }
            std::vector<int> idxs(d + 1);
            for (int j = 0; j <= d; ++j) {
                idxs[j] = j;
            }
            std::vector<llama_token> ids;
            if (cfg.mode == MODE_VERIFY_REJ) {
                // a drafter distribution: the draft token and one other
                std::vector<std::vector<llama_token_data>> q(d);
                for (int j = 0; j < d; ++j) {
                    q[j] = { { draft[j], 0.0f, 0.55f }, { (llama_token) ((draft[j] + 1) % g_n_vocab), 0.0f, 0.45f } };
                }
                ids = common_sampler_sample_and_accept_n_rejection(smpl, ctx, idxs, draft, q);
            } else {
                ids = common_sampler_sample_and_accept_n(smpl, ctx, idxs, draft);
            }
            for (auto t : ids) {
                res.toks.push_back(t);
            }
            // keep the last token + the accepted drafts
            llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + (llama_pos) ids.size(), -1);
            pos += (llama_pos) ids.size();
            id = ids.back();
            step++;
        }
        res.toks.resize(cfg.n_gen);
        llama_batch_free(b);
    } catch (const std::exception & e) {
        res.threw = true;
        res.err = e.what();
    }
    if (smpl) {
        res.st = common_sampler_get_dev_stats(smpl);
        common_sampler_free(smpl);
    }
    llama_free(ctx);
    if (dev) {
        llama_sampler_free(dev);
    }
    return res;
}

static std::string text_of(const std::vector<llama_token> & toks) {
    std::string s;
    for (auto t : toks) {
        s += t >= 0 && t < g_n_vocab ? common_token_to_piece(g_vocab, t, true) : std::string("<bad id ") + std::to_string(t) + ">";
    }
    return s;
}

static bool in_vocab(const std::vector<llama_token> & toks) {
    return std::all_of(toks.begin(), toks.end(), [](llama_token t) { return t >= 0 && t < g_n_vocab; });
}

static int run_main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--expect-diff]\n", argv[0]);
        return 2;
    }
    const bool expect_diff = argc > 2 && std::string(argv[2]) == "--expect-diff";
    if (expect_diff) {
        setenv("LLAMA_DEVICE_TOPK_GRAMMAR_NOFALLBACK", "1", 1);
    }
    const bool kill = getenv("LLAMA_DEVICE_TOPK_GRAMMAR") != nullptr && atoi(getenv("LLAMA_DEVICE_TOPK_GRAMMAR")) == 0;

    llama_backend_init();
    if (getenv("DTOPK_TEST_LOG") == nullptr) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    // not *.gguf: the copy sits in the shared test-models dir, which test-llama-context scans for *.gguf while ctest -j runs
    // this test beside it (it found the copy half written or already removed)
    const std::string spm = std::string(argv[1]) + (expect_diff ? ".dtopk-spm-x.tmp" : kill ? ".dtopk-spm-k.tmp" : ".dtopk-spm.tmp");
    uint32_t nv = 0;
    if (!write_spm_model(argv[1], spm, &nv) || nv < 1024) {
        fprintf(stderr, "cannot write the SPM copy of %s (vocab %u)\n", argv[1], nv);
        return 1;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(spm.c_str(), mp);
    if (g_model == nullptr) {
        fprintf(stderr, "cannot load %s\n", spm.c_str());
        return 1;
    }
    g_vocab = llama_model_get_vocab(g_model);
    g_n_vocab = llama_vocab_n_tokens(g_vocab);
    printf("model %s: vocab %d\n", spm.c_str(), g_n_vocab);

    const llama_token t_call  = tok_of("<tool_call>");
    const llama_token t_think = tok_of("<think>");
    const llama_token t_ethink = tok_of("</think>");
    check("vocabulary has <tool_call>, <think>, </think> as single tokens", t_call >= 0 && t_think >= 0 && t_ethink >= 0);

    std::vector<llama_token> prompt = { 1 };
    for (const char * s : { "city", ":", " ", "{\"", "name", "\":" }) {
        prompt.push_back(tok_of(s));
    }
    check("prompt tokens resolved", std::find(prompt.begin(), prompt.end(), LLAMA_TOKEN_NULL) == prompt.end());

    // a token with two biases: the candidates would get only the first one (the logit-bias sampler), so no device top-k
    {
        common_params_sampling sp;
        sp.logit_bias = { { 7, -1.0f }, { 7, -1.0f } };
        common_sampler * s1 = common_sampler_init(g_model, sp);
        sp.logit_bias = { { 7, -1.0f }, { 8, -1.0f } };
        common_sampler * s2 = common_sampler_init(g_model, sp);
        check("a token biased twice does not qualify for the device top-k (two distinct tokens do)",
              common_sampler_device_topk(s1, 64) == 0 && common_sampler_device_topk(s2, 64) == 64);
        common_sampler_free(s1);
        common_sampler_free(s2);
    }

    // presets: greedy, Studio sampled (temp 0.6 / 1.0, top_p 0.95, top_k 20, min_p 0), one with repetition penalties and ignore_eos
    struct preset { const char * name; float temp; int top_k; float top_p; float rep; int last_n; bool ignore_eos; int32_t k_dev; };
    const std::vector<preset> presets = {
        { "greedy",          0.0f, 40, 0.95f, 1.0f,  64, false,  64 },
        { "greedy-k1",       0.0f,  1, 1.00f, 1.0f,  64, true,   16 },
        { "t0.6-p0.95-k20",  0.6f, 20, 0.95f, 1.0f,  64, false,  64 },
        { "t1.0-p0.95-k20",  1.0f, 20, 0.95f, 1.0f,  64, true,   64 },
        { "t0.8-k20-rep1.1", 0.8f, 20, 0.95f, 1.1f,  32, false, 128 },
    };

    // grammars
    const std::string g_json = [] {
        const common_json schema = common_json::parse(R"({"type": "object", "properties": {"city": {"type": "string"},
            "days": {"type": "integer"}, "metric": {"type": "boolean"}}, "required": ["city", "days"]})");
        std::string g = json_schema_to_grammar(schema);
        // the object repeats (the request runs past its end with EOS excluded)
        const size_t p = g.find("root ::=");
        g.replace(p, 8, "obj ::=");
        return g + "\nroot ::= obj (\" \" obj)*\n";
    }();
    const std::string g_digits = "root ::= [0-9]+ (\",\" [0-9]+)*\n";
    auto g_tool = [&](const std::string & trig) {
        return "root ::= call (\",\" call)*\n"
               "call ::= " + gbnf_lit(trig) + " ws \"{\" ws \"\\\"name\\\"\" ws \":\" ws str ws \",\" ws \"\\\"arguments\\\"\" ws \":\" ws args ws \"}\"\n"
               "args ::= \"{\" ws ( str ws \":\" ws val ( ws \",\" ws str ws \":\" ws val )* )? ws \"}\"\n"
               "val ::= str | [0-9]+ | \"true\" | \"false\"\n"
               "str ::= \"\\\"\" [a-z_ ]* \"\\\"\"\n"
               "ws ::= [ ]?\n";
    };

    int n_cases = 0, n_diff = 0;
    int64_t tot_rows = 0, tot_rej = 0, tot_fb = 0, tot_forced = 0;

    for (const auto & pr : presets) {
        common_params_sampling base;
        base.seed  = 1234;
        base.temp  = pr.temp;
        base.top_k = pr.top_k;
        base.top_p = pr.top_p;
        base.min_p = 0.0f;
        base.penalty_repeat = pr.rep;
        base.penalty_last_n = pr.last_n;
        base.dry_multiplier = 0.0f;
        if (pr.ignore_eos) {
            base.logit_bias.push_back({ 2, -INFINITY });
        }
        if (pr.top_k == 1) {
            base.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
        }
        const bool sampled = pr.temp > 0.0f;

        // the reply without a grammar, one row per pass
        run_cfg c0;
        c0.sp = base;
        c0.k_dev = pr.k_dev;
        c0.n_gen = 64;
        const run_result r0row = run(c0, prompt, false);
        if (r0row.threw) {
            printf("  %s\n", r0row.err.c_str());
        }
        check(std::string(pr.name) + ": reference reply", !r0row.threw && r0row.toks.size() == 64);
        if (r0row.toks.size() != 64) {
            break;
        }

      for (run_mode mode : { MODE_ROW, MODE_VERIFY, MODE_VERIFY_REJ }) {
        if (mode == MODE_VERIFY_REJ && !sampled) {
            continue;
        }
        const char * mname = mode == MODE_ROW ? "row" : mode == MODE_VERIFY ? "verify" : "verify-rej";
        // the reply without a grammar in this mode (drafts from the row reply): the lazy grammars trigger on one of its tokens
        // (mid-reply) or on one it never emits
        run_result r0 = r0row;
        if (mode != MODE_ROW) {
            run_cfg cm = c0;
            cm.mode = mode;
            cm.ref = &r0row.toks;
            r0 = run(cm, prompt, false);
            if (r0.threw || r0.toks.size() != 64 || !in_vocab(r0.toks)) {
                check(std::string(pr.name) + " / " + mname + ": reference reply (" + r0.err + ")", false);
                continue;
            }
        }
        llama_token t_mid = LLAMA_TOKEN_NULL;
        size_t at_mid = 0;
        for (size_t i = 8; i < 40 && t_mid < 0; ++i) {
            const llama_token t = r0.toks[i];
            const std::string pc = common_token_to_piece(g_vocab, t, true);
            const bool printable = !pc.empty() && std::all_of(pc.begin(), pc.end(), [](char ch) { return ch > 0x20 && ch < 0x7f && ch != '"' && ch != '\\'; });
            // first occurrence as a token and as text (a word trigger matches the text)
            const std::string before = text_of(std::vector<llama_token>(r0.toks.begin(), r0.toks.begin() + i));
            if (t >= 3 + 256 && printable && std::find(r0.toks.begin(), r0.toks.begin() + i, t) == r0.toks.begin() + i &&
                    (before + pc).find(pc) == before.size()) {
                t_mid = t;
                at_mid = i;
            }
        }
        llama_token t_never = t_call;
        for (llama_token t = 300; std::find(r0.toks.begin(), r0.toks.end(), t_never) != r0.toks.end(); ++t) {
            t_never = t;
        }
        check(std::string(pr.name) + " / " + mname + ": a trigger token mid-reply", t_mid >= 0);
        if (t_mid < 0) {
            // the reply repeats itself from the start: the lazy cases cannot be set up (a FAIL above, not a skip)
            printf("  no token of the reply at 8..39 is new (as a token and as text) and printable: %s\n", text_of(r0.toks).c_str());
            continue;
        }
        const std::string p_mid = common_token_to_piece(g_vocab, t_mid, true);

        struct gcase { std::string name; common_params_sampling sp; llama_token pre = LLAMA_TOKEN_NULL; std::string expect = {}; bool ref0 = false; };
        std::vector<gcase> cases;
        {
            gcase c{ "none", base, LLAMA_TOKEN_NULL, {}, false };
            cases.push_back(c);
        }
        {
            gcase c{ "json-schema", base, LLAMA_TOKEN_NULL, {}, false };
            c.sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, g_json);
            cases.push_back(c);
        }
        {
            gcase c{ "digits (rejects most)", base, LLAMA_TOKEN_NULL, {}, false };
            c.sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, g_digits);
            c.expect = "fallback";
            cases.push_back(c);
        }
        {
            gcase c{ "lazy tool, token trigger mid-reply", base, LLAMA_TOKEN_NULL, {}, false };
            c.sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, g_tool(p_mid));
            c.sp.grammar_lazy = true;
            c.sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, p_mid, t_mid });
            c.expect = "trigger";
            c.ref0 = true;
            cases.push_back(c);
        }
        {
            gcase c{ "lazy tool, word trigger mid-reply", base, LLAMA_TOKEN_NULL, {}, false };
            c.sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, g_tool(p_mid));
            c.sp.grammar_lazy = true;
            c.sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, p_mid, LLAMA_TOKEN_NULL });
            c.expect = "trigger";
            c.ref0 = true;
            cases.push_back(c);
        }
        {
            gcase c{ "lazy tool, never triggers", base, LLAMA_TOKEN_NULL, {}, false };
            const std::string p_never = common_token_to_piece(g_vocab, t_never, true);
            c.sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, g_tool(p_never));
            c.sp.grammar_lazy = true;
            c.sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, p_never, t_never });
            c.expect = "never";
            c.ref0 = true;
            cases.push_back(c);
        }
        {
            gcase c{ "reasoning budget 6 (forced </think>)", base, LLAMA_TOKEN_NULL, {}, false };
            c.sp.reasoning_budget_tokens = 6;
            c.sp.reasoning_budget_start  = { t_think };
            c.sp.reasoning_budget_end    = { { t_ethink } };
            c.sp.reasoning_budget_forced = { t_ethink };
            c.pre = t_think;
            c.expect = "forced";
            cases.push_back(c);
        }
        {
            gcase c{ "reasoning budget 6 + lazy tool trigger", base, LLAMA_TOKEN_NULL, {}, false };
            c.sp.reasoning_budget_tokens = 6;
            c.sp.reasoning_budget_start  = { t_think };
            c.sp.reasoning_budget_end    = { { t_ethink } };
            c.sp.reasoning_budget_forced = { t_ethink };
            c.sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, g_tool(p_mid));
            c.sp.grammar_lazy = true;
            c.sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, p_mid, t_mid });
            c.pre = t_think;
            c.expect = "forced";
            cases.push_back(c);
        }

        for (const auto & gc : cases) {
            {
                run_cfg c;
                c.sp = gc.sp;
                c.k_dev = pr.k_dev;
                c.mode = mode;
                c.n_gen = 64;
                c.pre_accept = gc.pre;
                // drafts from the full-row run of the same request (single rows), so windows accept several tokens; the lazy cases
                // take the no-grammar row reply, as r0 did (the same windows up to the trigger)
                run_cfg cr = c;
                cr.mode = MODE_ROW;
                const run_result rr = mode == MODE_ROW || gc.ref0 ? r0row : run(cr, prompt, false);
                c.ref = &rr.toks;

                const run_result a = run(c, prompt, false);
                const std::string nm = std::string(pr.name) + " / " + gc.name + " / " + mname;

                if (kill) {
                    // the kill switch: grammar / budget requests do not qualify (the server copies the full rows as before)
                    common_params_sampling sp = c.sp;
                    common_sampler * s = common_sampler_init(g_model, sp);
                    const bool has = !c.sp.grammar.empty() || c.sp.reasoning_budget_tokens >= 0;
                    check(nm + ": LLAMA_DEVICE_TOPK_GRAMMAR=0 keeps it on the full rows", !has || common_sampler_device_topk(s, c.k_dev) == 0);
                    common_sampler_free(s);
                    continue;
                }

                const run_result d = run(c, prompt, true);
                n_cases++;
                const bool same = !a.threw && !d.threw && a.toks == d.toks;
                if (!expect_diff) {
                    check(nm + ": every sampled id is in the vocabulary", in_vocab(a.toks) && in_vocab(d.toks));
                }
                n_diff += !same;
                tot_rows += d.st.n_rows; tot_rej += d.st.n_rejected; tot_fb += d.st.n_fallback; tot_forced += d.st.n_forced;
                char info[160];
                snprintf(info, sizeof(info), " [rows %lld rej %lld fb %lld forced %lld]", (long long) d.st.n_rows,
                         (long long) d.st.n_rejected, (long long) d.st.n_fallback, (long long) d.st.n_forced);
                if (expect_diff) {
                    printf("%-100s %s%s\n", nm.c_str(), same ? "same" : "DIFF", info);
                    continue;
                }
                if (a.threw || d.threw) {
                    printf("  full: %s / device: %s\n", a.err.c_str(), d.err.c_str());
                }
                check(nm + ": same tokens" + info, same);
                if (!same && !a.threw && !d.threw) {
                    printf("  full:   %s\n  device: %s\n", text_of(a.toks).c_str(), text_of(d.toks).c_str());
                }
                const bool has = !c.sp.grammar.empty() || c.sp.reasoning_budget_tokens >= 0;
                check(nm + ": rows took the device top-k path", !has || d.st.n_rows > 0);
                if (gc.expect == "trigger") {
                    // the reply up to the trigger is the reference's; after it the grammar constrains
                    const std::string t = text_of(a.toks);
                    check(nm + ": triggered mid-reply (at token " + std::to_string(at_mid) + ")",
                          std::equal(r0.toks.begin(), r0.toks.begin() + at_mid + 1, a.toks.begin()) && t.find('{') != std::string::npos);
                }
                if (gc.expect == "never") {
                    check(nm + ": never triggered (reply = the no-grammar reply, nothing rejected)", a.toks == r0.toks && d.st.n_rejected == 0);
                }
                if (gc.expect == "forced") {
                    check(nm + ": the budget forced </think>", d.st.n_forced > 0 &&
                          std::find(a.toks.begin(), a.toks.end(), t_ethink) != a.toks.end());
                }
                if (gc.expect == "fallback" && sampled) {
                    check(nm + ": the full-row fallback ran", d.st.n_fallback > 0);
                }
            }
        }
      }
    }
    if (!kill) {
        printf("totals: %d cases, %d differ; device rows %lld, rejected %lld, fallback %lld, forced %lld\n", n_cases, n_diff,
               (long long) tot_rows, (long long) tot_rej, (long long) tot_fb, (long long) tot_forced);
        // the resample from the candidates alone (rejected, no fallback) must have been exercised too
        check("some rejections were resampled from the candidates (rejected > fallback)", expect_diff || tot_rej > tot_fb);
    }
    if (expect_diff) {
        check("LLAMA_DEVICE_TOPK_GRAMMAR_NOFALLBACK=1 (inexact) changes at least one reply: the comparison can fail", n_diff > 0);
    }

    llama_model_free(g_model);
    std::remove(spm.c_str());
    llama_backend_free();
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}

int main(int argc, char ** argv) {
    // a FAIL with the message, never an abort
    try {
        return run_main(argc, argv);
    } catch (const std::exception & e) {
        printf("uncaught exception: %s\nFAIL\n", e.what());
    } catch (...) {
        printf("uncaught exception\nFAIL\n");
    }
    return 1;
}
