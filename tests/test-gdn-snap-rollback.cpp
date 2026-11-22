// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// the delta-net snapshot rows of a prompt ubatch (n_rs_seq > 0) after a partial seq_rm into that ubatch: prefill N tokens in one
// ubatch, remove the last R (R <= n_rs_seq), decode one token, and compare its logits with a fresh context that prefilled N - R
// tokens and decoded the same token. A snapshot row left unwritten by the prompt kernel restores a wrong state and the logits differ.
// usage: test-gdn-snap-rollback <model.gguf> [N = 600] [R = 2] [n_rs_seq = 2]. Exit 0: max |diff| <= 1e-3 * max |logit| (+ 1e-4)
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool run(llama_model * model, int n_pre, int n_rm, int n_rs_seq, std::vector<float> & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 1024;
    cp.n_ubatch        = 1024;
    cp.n_seq_max       = 1;
    cp.n_rs_seq        = n_rs_seq;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto tok = [&](int p) { return (llama_token) ((p * 31 + 7) % (n_vocab - 10) + 5); };
    bool ok = true;

    llama_batch b = llama_batch_init(n_pre, 0, 1);
    for (int i = 0; i < n_pre; i++) {
        b.token[i] = tok(i); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == n_pre - 1;
    }
    b.n_tokens = n_pre;
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "prefill of %d tokens failed\n", n_pre);
        ok = false;
    }
    llama_batch_free(b);

    const int n_keep = n_pre - n_rm;
    if (ok && n_rm > 0 && !llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, -1)) {
        fprintf(stderr, "seq_rm of the last %d positions refused\n", n_rm);
        ok = false;
    }
    if (ok) {
        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = tok(n_keep); d.pos[0] = n_keep; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0; d.logits[0] = 1;
        d.n_tokens = 1;
        if (llama_decode(ctx, d) != 0) {
            fprintf(stderr, "decode after the rollback failed\n");
            ok = false;
        } else {
            const float * l = llama_get_logits_ith(ctx, 0);
            out.assign(l, l + n_vocab);
        }
        llama_batch_free(d);
    }
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [N] [R] [n_rs_seq]\n", argv[0]);
        return 2;
    }
    const int n   = argc > 2 ? atoi(argv[2]) : 600;
    const int r   = argc > 3 ? atoi(argv[3]) : 2;
    const int nrs = argc > 4 ? atoi(argv[4]) : 2;
    // the snapshot rows under test, not the replay rollback
    setenv("LLAMA_RS_REPLAY", "0", 0);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    std::vector<float> a, ref;
    const bool ok = run(model, n, r, nrs, a) && run(model, n - r, 0, nrs, ref);
    llama_model_free(model);
    llama_backend_free();
    if (!ok || a.size() != ref.size() || a.empty()) {
        fprintf(stderr, "FAIL: run error\n");
        return 1;
    }
    double dmax = 0.0, lmax = 0.0, sa = 0.0, sr = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dmax = std::max(dmax, (double) std::fabs(a[i] - ref[i]));
        lmax = std::max(lmax, (double) std::fabs(ref[i]));
        sa += a[i] * (double) (i + 1);
        sr += ref[i] * (double) (i + 1);
    }
    const double tol = 1e-3 * lmax + 1e-4;
    printf("N %d R %d n_rs_seq %d: max |diff| %.3e (tol %.3e, max |logit| %.3e), checksum rollback %.9e fresh %.9e\n",
            n, r, nrs, dmax, tol, lmax, sa, sr);
    if (!(dmax <= tol)) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
