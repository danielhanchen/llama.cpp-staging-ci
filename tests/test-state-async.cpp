// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// Checks LLAMA_STATE_SEQ_FLAGS_ASYNC against a synchronous save of the same sequence.
// The saved bytes must be complete at every point the API documents (the next llama_decode(), llama_synchronize() or
// llama_state_* call), also when the destination is reused by the next save, when the next call only queries a size or
// fails, and when the memory is cleared right after the save. Before each asynchronous save the device is kept busy
// with a decode of another sequence, so the copy is still pending when the next call is made.
// The asynchronous path needs a GPU backend (CUDA, ROCm, MUSA); elsewhere the save is synchronous and every check passes.
// Usage: test-state-async -m MODEL [-ngl 99] [--repeat N]

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_staging_logs = 0;
static int n_sync_logs    = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (strstr(text, "staging buffer")) {
        n_staging_logs++;
    }
    if (strstr(text, "saving the state synchronously")) {
        n_sync_logs++;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    common_params params;

    params.kv_unified = true;
    params.n_parallel = 3; // seq 0 and 1 are saved, seq 2 keeps the device busy
    params.n_ctx      = 4096;
    params.n_batch    = 1024;
    params.n_ubatch   = 512;

    int n_repeat = 10;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            n_repeat = atoi(argv[i + 1]);
            for (int j = i; j + 2 <= argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_log_set(log_cb, nullptr);

    const char * env_async = getenv("LLAMA_STATE_ASYNC");
    printf("LLAMA_STATE_ASYNC=%s repeat=%d\n", env_async ? env_async : "(unset)", n_repeat);

    llama_memory_t mem = llama_get_memory(ctx);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // deterministic token ids
    uint32_t rng = 12345;
    auto next_tok = [&]() {
        rng = rng*1664525u + 1013904223u;
        return (llama_token) (100 + (rng >> 8) % (uint32_t) std::max(1, n_vocab/2 - 100));
    };

    const int n0 = 300; // seq 0: more cells than seq 1, so the two states have different layouts
    const int n1 = 120;
    const int nb = 1024; // busy decode

    std::vector<llama_token> toks0(n0), toks1(n1), toks_busy(nb);
    for (auto & t : toks0)     { t = next_tok(); }
    for (auto & t : toks1)     { t = next_tok(); }
    for (auto & t : toks_busy) { t = next_tok(); }
    const llama_token tok_probe = next_tok();

    common_batch batch(ctx);

    auto decode = [&](const std::vector<llama_token> & toks, llama_pos pos0, llama_seq_id seq, bool out_last) {
        batch.clear();
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.add(toks[i], pos0 + (llama_pos) i, seq, out_last && i + 1 == toks.size());
        }
        const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            fprintf(stderr, "decode failed: %d\n", ret);
            exit(1);
        }
    };

    // queue work on the device that does not touch seq 0 and 1, and return without waiting for it
    auto busy = [&]() {
        decode(toks_busy, 0, 2, true);
    };
    auto busy_done = [&]() {
        llama_synchronize(ctx);
        llama_memory_seq_rm(mem, 2, -1, -1);
    };

    auto fill = [&]() {
        decode(toks0, 0, 0, true);
        decode(toks1, 0, 1, true);
        llama_synchronize(ctx);
    };

    fill();

    int n_fail_total = 0;

    for (const llama_state_seq_flags base_flags : { (llama_state_seq_flags) LLAMA_STATE_SEQ_FLAGS_NONE, (llama_state_seq_flags) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY }) {
        const llama_state_seq_flags async_flags = base_flags | LLAMA_STATE_SEQ_FLAGS_ASYNC;

        // synchronous reference states
        std::vector<uint8_t> ref0(llama_state_seq_get_size_ext(ctx, 0, base_flags));
        std::vector<uint8_t> ref1(llama_state_seq_get_size_ext(ctx, 1, base_flags));
        if (llama_state_seq_get_data_ext(ctx, ref0.data(), ref0.size(), 0, base_flags) != ref0.size() ||
            llama_state_seq_get_data_ext(ctx, ref1.data(), ref1.size(), 1, base_flags) != ref1.size()) {
            fprintf(stderr, "reference save failed\n");
            return 1;
        }
        printf("\nflags=%u: state sizes seq0 %zu, seq1 %zu bytes\n", base_flags, ref0.size(), ref1.size());
        if (ref0.size() <= 64 || ref0.size() == ref1.size()) {
            printf("flags=%u: no state of this kind, skipped\n", base_flags);
            continue;
        }

        // restores seq 0 and 1 from the references (the cleared case starts from an empty memory)
        auto restore_refs = [&]() {
            llama_memory_clear(mem, true);
            if (base_flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
                // a partial state cannot be restored alone: rebuild the full state
                fill();
                return;
            }
            if (llama_state_seq_set_data_ext(ctx, ref0.data(), ref0.size(), 0, base_flags) != ref0.size() ||
                llama_state_seq_set_data_ext(ctx, ref1.data(), ref1.size(), 1, base_flags) != ref1.size()) {
                fprintf(stderr, "restore failed\n");
                exit(1);
            }
        };

        // logits of one probe token after restoring a seq 0 state into an empty memory
        auto probe_logits = [&](const std::vector<uint8_t> & st) {
            llama_memory_clear(mem, true);
            std::vector<float> res;
            if (llama_state_seq_set_data_ext(ctx, st.data(), st.size(), 0, base_flags) != st.size()) {
                return res;
            }
            decode({ tok_probe }, n0, 0, true);
            const float * lg = llama_get_logits_ith(ctx, -1);
            res.assign(lg, lg + n_vocab);
            return res;
        };

        std::vector<float> logits_ref;
        if (!(base_flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
            logits_ref = probe_logits(ref0);
            restore_refs();
        }

        std::vector<uint8_t> buf(std::max(ref0.size(), ref1.size()));
        std::vector<uint8_t> tiny(16);

        auto same = [](const std::vector<uint8_t> & a, size_t n, const std::vector<uint8_t> & ref) {
            return n == ref.size() && memcmp(a.data(), ref.data(), ref.size()) == 0;
        };
        auto first_diff = [](const std::vector<uint8_t> & a, const std::vector<uint8_t> & ref) {
            for (size_t i = 0; i < ref.size(); ++i) {
                if (a[i] != ref[i]) {
                    return (long) i;
                }
            }
            return -1L;
        };

        struct result { const char * name; int n_fail; long diff; };
        std::vector<result> results;

        auto run = [&](const char * name, auto && fn) {
            int n_fail = 0;
            long diff = -1;
            for (int r = 0; r < n_repeat; ++r) {
                long d = -1;
                if (!fn(d)) {
                    n_fail++;
                    if (diff < 0) {
                        diff = d;
                    }
                }
            }
            results.push_back({ name, n_fail, diff });
            printf("flags=%u %-44s mismatches %d / %d%s\n", base_flags, name, n_fail, n_repeat,
                    n_fail ? (" (first differing byte " + std::to_string(diff) + ")").c_str() : "");
            fflush(stdout);
            n_fail_total += n_fail;
        };

        // 0. documented: complete after llama_synchronize()
        run("save, synchronize", [&](long & d) {
            memset(buf.data(), 0xcc, buf.size());
            busy();
            const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            busy_done();
            d = first_diff(buf, ref0);
            return same(buf, n, ref0);
        });

        // 1. documented: complete after the next llama_decode()
        run("save, decode", [&](long & d) {
            memset(buf.data(), 0xcc, buf.size());
            busy();
            const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            decode(toks_busy, nb, 2, true);
            d = first_diff(buf, ref0);
            const bool ok = same(buf, n, ref0);
            busy_done();
            return ok;
        });

        // 2. claim 2: two saves into one destination, the second one smaller
        run("save seq0, save seq1 into the same buffer", [&](long & d) {
            memset(buf.data(), 0xcc, buf.size());
            busy();
            llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 1, async_flags);
            busy_done();
            d = first_diff(buf, ref1);
            return same(buf, n, ref1);
        });

        // 3. claim 1: complete after a size query
        run("save, llama_state_seq_get_size_ext", [&](long & d) {
            memset(buf.data(), 0xcc, buf.size());
            busy();
            const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            llama_state_seq_get_size_ext(ctx, 0, base_flags);
            d = first_diff(buf, ref0);
            const bool ok = same(buf, n, ref0);
            busy_done();
            return ok;
        });

        run("save, llama_state_get_size", [&](long & d) {
            memset(buf.data(), 0xcc, buf.size());
            busy();
            const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            llama_state_get_size(ctx);
            d = first_diff(buf, ref0);
            const bool ok = same(buf, n, ref0);
            busy_done();
            return ok;
        });

        // 4. claim 1: complete after a save that fails (destination too small)
        run("save, failing save", [&](long & d) {
            memset(buf.data(), 0xcc, buf.size());
            busy();
            const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            const size_t n_tiny = llama_state_seq_get_data_ext(ctx, tiny.data(), tiny.size(), 0, async_flags);
            d = first_diff(buf, ref0);
            const bool ok = same(buf, n, ref0) && n_tiny == 0;
            busy_done();
            return ok;
        });

        // 5. claim 3: the memory is cleared right after the save (idle device, and busy device)
        for (const bool with_busy : { false, true }) {
            run(with_busy ? "save, llama_memory_clear (busy device)" : "save, llama_memory_clear (idle device)", [&](long & d) {
                memset(buf.data(), 0xcc, buf.size());
                if (with_busy) {
                    busy();
                } else {
                    llama_synchronize(ctx);
                }
                const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
                llama_memory_clear(mem, true);
                llama_synchronize(ctx);
                d = first_diff(buf, ref0);
                bool ok = same(buf, n, ref0);
                if (!logits_ref.empty()) {
                    const std::vector<float> lg = probe_logits(buf);
                    const bool same_logits = lg.size() == logits_ref.size() && memcmp(lg.data(), logits_ref.data(), lg.size()*sizeof(float)) == 0;
                    if (!same_logits && ok) {
                        printf("  bytes equal but logits differ\n");
                    }
                    ok = ok && same_logits;
                }
                restore_refs();
                return ok;
            });
        }

        // the state of seq 0 and 1 is unchanged by the busy decodes
        {
            std::vector<uint8_t> chk0(ref0.size()), chk1(ref1.size());
            llama_state_seq_get_data_ext(ctx, chk0.data(), chk0.size(), 0, base_flags);
            llama_state_seq_get_data_ext(ctx, chk1.data(), chk1.size(), 1, base_flags);
            if (chk0 != ref0 || chk1 != ref1) {
                printf("flags=%u: SETUP ERROR: the reference states changed\n", base_flags);
                n_fail_total++;
            }
        }

        // the save returns before the device is done (the asynchronous path ran)
        {
            busy();
            int64_t t0 = now_us();
            llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, async_flags);
            const int64_t t_async = now_us() - t0;
            busy_done();
            busy();
            t0 = now_us();
            llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, base_flags);
            const int64_t t_sync = now_us() - t0;
            busy_done();
            printf("flags=%u: save call with a busy device: async %lld us, sync %lld us\n", base_flags, (long long) t_async, (long long) t_sync);
        }
    }

    printf("\nstaging buffer logs %d, synchronous fallback logs %d\n", n_staging_logs, n_sync_logs);
    printf("%s: %d mismatches\n", n_fail_total ? "FAIL" : "PASS", n_fail_total);

    return n_fail_total ? 1 : 0;
}
