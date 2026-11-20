// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// exact verify (llama_set_verify_exact, LLAMA_MTP_EXACT) gives each verify row an attention over the cells a decode of that row alone
// would see. With several sequences in one unified cache the cells do not follow the positions: sequence A in cells 0..127, B (positions
// 0..254) in cells 128..382, a verify of B at positions 255, 256 must still attend all of B. Compares the verify rows with the same rows
// decoded one at a time (equal bytes), and checks that the trimmed attention is still used where it is valid (B alone, its cells
// following its positions).

#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

struct fa_extents {
    std::multiset<int64_t> n_kv; // K rows of every flash attention node of the last graph
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask && t->op == GGML_OP_FLASH_ATTN_EXT) {
        ((fa_extents *) user_data)->n_kv.insert(t->src[1]->ne[1]);
    }
    return false;
}

static llama_context * make_ctx(llama_model * model, fa_extents * ext) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = 2;
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = ext;
    return llama_init_from_model(model, cp);
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq, bool logits_all) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = p0 + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = logits_all || i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static double nmse(const float * a, const float * b, int n) {
    double e = 0.0;
    double r = 0.0;
    for (int i = 0; i < n; ++i) {
        e += (double) (a[i] - b[i])*(a[i] - b[i]);
        r += (double) a[i]*a[i];
    }
    return e/r;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> tok_a(128);
    std::vector<llama_token> tok_b(257);
    unsigned x = 7;
    for (auto & t : tok_a) { x = x*1103515245u + 12345u; t = (llama_token) ((x >> 8) % n_vocab); }
    for (auto & t : tok_b) { x = x*1103515245u + 12345u; t = (llama_token) ((x >> 8) % n_vocab); }
    const std::vector<llama_token> prompt_b(tok_b.begin(), tok_b.begin() + 255);
    const std::vector<llama_token> verify_b(tok_b.begin() + 255, tok_b.end());

    int fail = 0;

    // B alone, then A in cells 0..127 and B after it; positions 255 and 256 of B decoded one token at a time (the reference) and as one
    // exact verify window (bytes equal to the reference)
    for (const bool with_a : { false, true }) {
        std::vector<float> ref;
        {
            fa_extents ext;
            llama_context * ctx = make_ctx(model, &ext);
            if (with_a && !decode(ctx, tok_a, 0, 0, false)) { return 1; }
            if (!decode(ctx, prompt_b, 0, 1, false)) { return 1; }
            for (int k = 0; k < 2; ++k) {
                if (!decode(ctx, { verify_b[k] }, 255 + k, 1, true)) { return 1; }
                const float * l = llama_get_logits_ith(ctx, -1);
                ref.insert(ref.end(), l, l + n_vocab);
            }
            llama_free(ctx);
        }
        fa_extents ext;
        llama_context * ctx = make_ctx(model, &ext);
        if (with_a && !decode(ctx, tok_a, 0, 0, false)) { return 1; }
        if (!decode(ctx, prompt_b, 0, 1, false)) { return 1; }
        ext.n_kv.clear();
        llama_set_verify(ctx, true);
        llama_set_verify_exact(ctx, true);
        if (!decode(ctx, verify_b, 255, 1, true)) { return 1; }
        llama_set_verify(ctx, false);
        double err  = 0.0;
        bool   same = true;
        for (int k = 0; k < 2; ++k) {
            const float * l = llama_get_logits_ith(ctx, k);
            err  = std::max(err, nmse(ref.data() + (size_t) k*n_vocab, l, n_vocab));
            same = same && memcmp(ref.data() + (size_t) k*n_vocab, l, n_vocab*sizeof(float)) == 0;
        }
        const bool trimmed = ext.n_kv.count(256) > 0;
        const bool ok_err  = same;
        // B alone: the row at position 255 reads 256 cells; with A in front of B every cell of B is needed
        const bool ok_trim = with_a ? !trimmed : trimmed;
        printf("%-34s nmse %.3e (%s), 256-cell rows %s (%s)\n", with_a ? "A in cells 0..127, B after it:" : "B alone:",
            err, ok_err ? "OK" : "FAIL", trimmed ? "yes" : "no", ok_trim ? "OK" : "FAIL");
        fail |= !ok_err || !ok_trim;
        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    return fail;
}
