// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// k19-rollback: the host copy of a recurrent state (llama_rs_snap_save / llama_rs_snap_restore) as the rollback of a drafted verify
// window without per-token rows on the device (n_rs_seq 0). A run that saves, decodes a 4-token window, restores, removes the window's
// positions and decodes the accepted tokens again gives the logits of a run that decoded those tokens without the window, byte for byte
// (the same batches from the same state); a run whose windows are all accepted (no restore) gives the logits of the window decodes
#include "llama.h"
#include "../src/llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// mode 0: no windows; 1: windows rolled back by the host copy; 2: windows taken whole (a copy saved, never restored); 3: as 2 without copies
static bool run(const char * path, int mode, std::vector<float> & out, int & n_snap) {
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 1; cp.n_rs_seq = 0; cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        llama_model_free(model);
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);
    int pos = 0;
    bool ok = true;
    auto tok = [&](int p, int salt) { return (llama_token) ((p * 17 + salt * 7) % (n_vocab - 10) + 5); };
    auto dec = [&](const std::vector<llama_token> & t, bool keep) {
        llama_batch b = llama_batch_init(64, 0, 1);
        for (size_t i = 0; i < t.size(); i++) {
            b.token[i] = t[i]; b.pos[i] = pos++; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = (int) t.size();
        if (llama_decode(ctx, b) != 0) {
            ok = false;
        } else if (keep) {
            for (size_t i = 0; i < t.size(); i++) {
                const float * l = llama_get_logits_ith(ctx, (int) i);
                out.insert(out.end(), l, l + n_vocab);
            }
        }
        llama_batch_free(b);
    };
    std::vector<llama_token> p;
    for (int i = 0; i < 12; i++) {
        p.push_back(tok(i, 0));
    }
    dec(p, true);
    if (mode != 0 && !llama_rs_snap_supported(ctx)) {
        fprintf(stderr, "host rollback copy not supported by this context\n");
        ok = false;
    }
    for (int step = 1; step <= 6 && ok; step++) {
        const int p0 = pos;
        // the window: the accepted token, then 3 drafts (the first of them rejected in mode 1)
        const std::vector<llama_token> win = { tok(p0, step), tok(p0 + 1, step + 30), tok(p0 + 2, step + 30), tok(p0 + 3, step + 30) };
        if (mode == 1 || mode == 4) {
            if (mode == 1 || step == 1) {
                ok = ok && llama_rs_snap_save(ctx, 0);   // mode 4 (negative control) keeps the first copy: stale from step 2 on
            }
            dec(win, false);
            ok = ok && llama_rs_snap_restore(ctx, 0);
            ok = ok && llama_memory_seq_rm(mem, 0, p0, -1);
            pos = p0;
            n_snap++;
            dec({ win[0], tok(p0 + 1, step + 50) }, true);   // the accepted token and the target's own next one, again from the copy
        } else if (mode >= 2) {
            if (mode == 2) {
                ok = ok && llama_rs_snap_save(ctx, 0);
                n_snap++;
            }
            dec(win, true);
        } else {
            dec({ win[0], tok(p0 + 1, step + 50) }, true);
        }
        dec({ tok(pos, step + 70) }, true);
    }
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

static double max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = a.size() == b.size() && !a.empty() ? 0.0 : INFINITY;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        d = std::fmax(d, std::fabs((double) a[i] - b[i]));
    }
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s recurrent-model.gguf\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);
    std::vector<float> ref, snap, whole, whole0;
    int n0 = 0, n1 = 0, n2 = 0, n3 = 0;
    const bool ok = run(argv[1], 0, ref, n0) && run(argv[1], 1, snap, n1) && run(argv[1], 2, whole, n2) && run(argv[1], 3, whole0, n3);
    const double d1 = ok ? max_diff(ref, snap) : INFINITY;
    const double d2 = ok ? max_diff(whole, whole0) : INFINITY;
    printf("%zu logits, %d rollbacks by the host copy: max |no window - window rolled back| = %g\n", ref.size(), n1, d1);
    printf("%zu logits of whole windows, %d copies taken: max |with copies - without| = %g\n", whole.size(), n2, d2);
    std::vector<float> bad; int n4 = 0;
    const bool ok4 = ok && run(argv[1], 4, bad, n4);
    const double d4 = ok4 ? max_diff(ref, bad) : 0.0;
    printf("negative control (stale copy restored): max diff %g (must be > 0)\n", d4);
    const bool pass = ok && ok4 && d1 == 0.0 && d2 == 0.0 && n1 == 6 && n2 == 6 && d4 > 0.0;
    printf("%s\n", pass ? "OK" : "FAIL");
    llama_backend_free();
    return pass ? 0 : 1;
}
