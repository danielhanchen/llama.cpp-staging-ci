// recurrent state cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// roll back multiple sequences, then replay them in a single batch whose per-seq token count
// exceeds n_ubatch: each seq's replay spans several ubatches while its rollback restore is
// still pending. Compared against a reference context that never advanced past the rollback
// point and decodes the identical replay batch.
struct case_multi_seq_split_replay : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "replay"; }

    const char * description() const override {
        return "replay rolled-back sequences in a batch wider than n_ubatch, against a reference context";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t  n_seqs     = 2;
        constexpr uint32_t  n_ubatch   = 16;
        constexpr uint32_t  n_prompt   = 20;
        constexpr uint32_t  n_rollback = 3;
        constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
        constexpr llama_pos p0         = n_prompt - n_rollback;

        // both contexts have the same layout, only what they decode differs
        const auto make_ctx_multi = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 256;
            opts.n_batch    = 256;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_roll = make_ctx_multi();
        llama_context_ptr ctx_ref  = make_ctx_multi();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        const auto tok = [&](uint32_t seq, llama_pos pos) {
            return gen_token(n_vocab, (llama_seq_id) seq, pos);
        };

        bool ok = true;

        // keep the anchor at p0 - 1 when rolling back the tail
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);

            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) p0);

            ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

            // a second partial removal while one is pending must be refused
            ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
        }
        CTX_REQUIRE(ok, "the multi-seq prefill / rollback failed");

        // all seqs replay in a single batch
        const auto decode_replay = [&](llama_context * ctx) {
            common_batch batch(ctx);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (uint32_t i = 0; i < n_replay; ++i) {
                    const llama_pos pos = p0 + (llama_pos) i;
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
                }
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        ok = decode_replay(ctx_roll.get()) && decode_replay(ctx_ref.get());
        CTX_REQUIRE(ok, "the multi-seq replay decode failed");

        // both contexts decode identical batches, so the logits should match;
        // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
        constexpr float nmse_eps = 1e-4f;

        logits_diff diff;
        for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "multi-seq split replay logits mismatch (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
        //   in ctx_ref only, then compare identical seq-1-only continuations
        constexpr uint32_t n_tail = 4;

        {
            common_batch batch_tail(ctx_ref.get());
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = p0 + (llama_pos) (n_replay + i);
                batch_tail.add(tok(0, pos + 7), pos, 0, false);
            }
            ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
        }

        logits_diff diff_tail;
        for (uint32_t i = 0; i < n_tail && ok; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1) &&
                 decode_one(ctx_ref.get(),  tok(1, pos), pos, 1);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing tail logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff_tail.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(ok && diff_tail.nmse() <= nmse_eps,
                "a seq-1-only decode leaked the state of seq 0 (nmse " + std::to_string(diff_tail.nmse()) + ")");

        TLOG_INF("%s: seq-1-only decode is independent of seq 0 (max diff %g, nmse %g)\n",
                __func__, (double) diff_tail.diff_max, diff_tail.nmse());

        return case_result::pass();
    }
};

// save a rolled-back single-seq state, restore it into a fresh and into a dirty context, and
// verify exact logit matches on replay
struct case_rollback : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback"; }

    const char * description() const override {
        return "a rolled-back recurrent state restores into fresh and dirty contexts with identical replay";
    }

    case_result run(model_run & mr) const override {
        const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
        const int           n_vocab = llama_vocab_n_tokens(vocab);

        const auto make_rs_ctx = [&]() {
            ctx_opts opts;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_src = make_rs_ctx();
        llama_context_ptr ctx_dst = make_rs_ctx();
        if (!ctx_src || !ctx_dst) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        constexpr uint32_t n_rollback = 3;
        if (n_rs_seq < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_tokens tokens;
        if (!mr.has_vocab()) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        } else {
            tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
        }
        if (tokens.empty()) {
            TLOG_ERR("%s: not enough prompt tokens\n", __func__);
            return case_result::fail();
        }
        tokens.resize(n_rs_seq + 1, tokens.back());

        const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

        // Decode the full prompt on the source, then roll back three positions.
        // Replaying them crosses DSV4's ratio-4 compressor boundary.
        // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
        if (!decode_tokens(ctx_src.get(), tokens)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: rollback failed\n", __func__);
            return case_result::fail();
        }

        // Save the rolled-back state and restore it into a fresh context.
        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_src.get(), 0, 0);
        ckpt.load_tgt(ctx_dst.get(), 0, 0);

        // exact match - the same context configuration replaying the same tokens
        constexpr float nmse_eps = 0.0;

        std::vector<std::vector<float>> logits_src_replay(n_rollback);
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
        //       this is not the case here. add asserts and guardrails to prevent such attempts
        //if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) ||
        //    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1)) {
        //    TLOG_ERR("%s: partial rollback failed\n", __func__);
        //    return case_result::fail();
        //}

        //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        //common_prompt_checkpoint ckpt_partial;
        //ckpt_partial.update_tgt(ctx_src.get(), 0, partial_flags);
        //ckpt_partial.load_tgt(ctx_dst.get(), 0, partial_flags);

        // Repeat the load into a context that already has its own rollback state:
        // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
        // non-zero at load time. The restore must wipe that state and still match.
        llama_context_ptr ctx_dirty = make_rs_ctx();
        if (!ctx_dirty) {
            return case_result::fail();
        }

        llama_tokens noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!decode_tokens(ctx_dirty.get(), noise)) {
            TLOG_ERR("%s: failed to decode the dirty prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: dirty rollback failed\n", __func__);
            return case_result::fail();
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (logits_dirty == nullptr) {
                TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src_replay[i].data(), logits_dirty, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "dirty-ctx logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

        return case_result::pass();
    }
};

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
struct case_shared_seq_reserve : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "shared"; }

    const char * description() const override {
        return "sequences sharing cells keep decoding correctly after a seq_cp";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the shared-seq batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs     = 2;
        constexpr uint32_t n_prompt   = 128;
        constexpr uint32_t n_continue = 32;

        ctx_opts opts;
        opts.n_ctx      = 512;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1; // only a unified cache shares cells on seq_cp

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
            TLOG_ERR("%s: prompt decode failed\n", __func__);
            return case_result::fail();
        }

        // this is what llama-batched-bench does for -pps
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

        for (uint32_t i = 0; i < n_continue; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);

            common_batch batch(ctx.get());
            for (uint32_t s = 0; s < n_seqs; ++s) {
                batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "shared-seq decode failed at step " + std::to_string(i));

            for (uint32_t s = 0; s < n_seqs; ++s) {
                const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
                if (logits == nullptr) {
                    TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                    return case_result::fail();
                }

                bool finite = true;
                for (int t_ = 0; t_ < n_vocab; ++t_) {
                    finite = finite && std::isfinite(logits[t_]);
                }

                CTX_REQUIRE(finite, "non-finite shared-seq logit at step " + std::to_string(i) + ", seq " + std::to_string(s));
            }
        }

        TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);

        return case_result::pass();
    }
};

// k19-npmtp: two parallel requests with draft verify windows (llama-server --parallel N with an MTP / draft model). A batch holds the
// verify window of one sequence and a prompt chunk of another, so the window is not in the last ubatch of the decode; the window is
// then rolled back after the decode. A later batch places two windows so that the cells move. Run with the delta-net rollback by replay
// (LLAMA_RS_REPLAY=1) against the snapshot rows (LLAMA_RS_REPLAY=0), every logit compared. Before the fix the replay context committed
// every window before each ubatch (the rollback was refused: llama-server aborted in seq_rm), and a moved window read the old row
struct case_replay_overlap : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "replay_overlap"; }

    const char * description() const override {
        return "parallel verify windows roll back after later ubatches moved their cells, replay and snapshot rows against one sequence per ubatch";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the overlap batches are multi-seq graphs";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs   = 4;
        constexpr llama_pos n_pre   = 8;
        constexpr llama_pos n_chunk = 20;

        // the mode is read when the memory is made: set for each context, then the caller's value back
        const char * env_prev = getenv("LLAMA_RS_REPLAY");
        const std::string prev = env_prev ? env_prev : "";
        const auto make = [&](const char * replay) {
            setenv("LLAMA_RS_REPLAY", replay, 1);
            ctx_opts opts;
            opts.n_ctx      = 256;
            opts.n_batch    = 64;
            opts.n_ubatch   = 32;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 2;
            opts.kv_unified = 1;
            auto ctx = mr.ctx(opts);
            if (env_prev) {
                setenv("LLAMA_RS_REPLAY", prev.c_str(), 1);
            } else {
                unsetenv("LLAMA_RS_REPLAY");
            }
            return ctx;
        };

        llama_context_ptr ctx_rp   = make("1");
        llama_context_ptr ctx_snap = make("0");
        // the oracle: snapshot rows, every span of a batch decoded on its own (one sequence per ubatch: no cell is ever moved)
        llama_context_ptr ctx_ser  = make("0");
        if (!ctx_rp || !ctx_snap) {
            return case_result::fail();
        }
        if (llama_n_rs_seq(ctx_rp.get()) < 2) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_context * ctxs[2] = { ctx_rp.get(), ctx_snap.get() };

        // the same batch on both contexts, every output row compared
        logits_diff diff;
        const auto both = [&](const std::vector<std::array<llama_pos, 3>> & spans) {
            int32_t n_out = 0;
            for (llama_context * ctx : ctxs) {
                common_batch batch(ctx);
                n_out = 0;
                for (const auto & sp : spans) {
                    for (llama_pos p = sp[1]; p < sp[2]; ++p) {
                        batch.add(gen_token(n_vocab, (llama_seq_id) sp[0], p), p, (llama_seq_id) sp[0], true);
                        n_out++;
                    }
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    return false;
                }
            }
            for (const auto & sp : spans) {
                common_batch batch(ctx_ser.get());
                for (llama_pos p = sp[1]; p < sp[2]; ++p) {
                    batch.add(gen_token(n_vocab, (llama_seq_id) sp[0], p), p, (llama_seq_id) sp[0], true);
                }
                if (llama_process(ctx_ser.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    return false;
                }
            }
            for (int32_t i = 0; i < n_out; ++i) {
                const float * a = llama_get_logits_ith(ctx_rp.get(), i);
                const float * b = llama_get_logits_ith(ctx_snap.get(), i);
                if (a == nullptr || b == nullptr) {
                    return false;
                }
                diff.add(a, b, n_vocab);
            }
            return true;
        };
        const auto rollback = [&](llama_seq_id seq, llama_pos p0) {
            bool ok = true;
            for (llama_context * ctx : ctxs) {
                ok = ok && llama_memory_seq_rm(llama_get_memory(ctx), seq, p0, -1);
            }
            return ok && llama_memory_seq_rm(llama_get_memory(ctx_ser.get()), seq, p0, -1);
        };

        for (uint32_t s = 0; s < n_seqs; ++s) {
            CTX_REQUIRE(both({ { (llama_pos) s, 0, n_pre } }), "prefill failed");
        }

        // seq 0: a verify window (3 tokens) beside the prompt chunk of seq 1, which takes a later ubatch of the same decode
        CTX_REQUIRE(both({ { 0, n_pre, n_pre + 3 }, { 1, n_pre, n_pre + n_chunk } }), "overlap decode failed");
        CTX_REQUIRE(rollback(0, n_pre + 2), "the window of seq 0 cannot be rolled back after a later ubatch of seq 1");

        // windows of seqs 0 and 2 in one ubatch: their cells are gathered next to each other, the cell of seq 1 moves
        CTX_REQUIRE(both({ { 0, n_pre + 2, n_pre + 5 }, { 2, n_pre, n_pre + 3 } }), "two-window decode failed");
        CTX_REQUIRE(rollback(0, n_pre + 3) && rollback(2, n_pre + 1), "the windows of seqs 0 and 2 cannot be rolled back");

        // then all three, one token each, and a few more steps
        llama_pos pos[3] = { n_pre + 3, n_pre + n_chunk, n_pre + 1 };
        for (int step = 0; step < 4; ++step) {
            CTX_REQUIRE(both({ { 1, pos[1], pos[1] + 1 }, { 0, pos[0], pos[0] + 1 }, { 2, pos[2], pos[2] + 1 } }),
                    "decode failed at step " + std::to_string(step));
            for (llama_pos & p : pos) {
                p++;
            }
        }

        // the four requests of llama-server --parallel 4 (G4 k15kn_g4 par4): a verify window of seq 0 in its own ubatch, then a ubatch
        // of the three others whose cells are gathered next to each other, which moves the cell of seq 0 before its rollback
        CTX_REQUIRE(both({ { 0, pos[0], pos[0] + 3 }, { 1, pos[1], pos[1] + 4 }, { 2, pos[2], pos[2] + 4 }, { 3, n_pre, n_pre + 4 } }),
                "four-sequence decode failed");
        CTX_REQUIRE(rollback(0, pos[0] + 1), "the window of seq 0 cannot be rolled back after a ubatch that moved its cell");
        llama_pos pos4[4] = { pos[0] + 1, pos[1] + 4, pos[2] + 4, n_pre + 4 };
        for (int step = 0; step < 3; ++step) {
            CTX_REQUIRE(both({ { 2, pos4[2], pos4[2] + 1 }, { 0, pos4[0], pos4[0] + 1 }, { 3, pos4[3], pos4[3] + 1 }, { 1, pos4[1], pos4[1] + 1 } }),
                    "four-sequence step failed at " + std::to_string(step));
            for (llama_pos & p : pos4) {
                p++;
            }
        }

        // the recurrent states themselves, against the serial oracle: the logits of a random dummy model hardly depend on them (a moved
        // window that read the state of another sequence changed the logits by 1e-8)
        for (uint32_t s = 0; s < n_seqs; ++s) {
            llama_context * cs[3] = { ctx_rp.get(), ctx_snap.get(), ctx_ser.get() };
            const char *   nm[2] = { "replay", "snapshot rows" };
            std::vector<uint8_t> st[3];
            for (int c = 0; c < 3; ++c) {
                const size_t n = llama_state_seq_get_size_ext(cs[c], (llama_seq_id) s, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                st[c].resize(n);
                CTX_REQUIRE(n > 0 && llama_state_seq_get_data_ext(cs[c], st[c].data(), n, (llama_seq_id) s,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == n, "state save failed");
            }
            for (int c = 0; c < 2; ++c) {
                CTX_REQUIRE(st[c].size() == st[2].size(), std::string("the ") + nm[c] + " state of seq " + std::to_string(s) + " differs in size");
                float d_max = 0.0f;
                float a_max = 0.0f;
                for (size_t i = 0; i + sizeof(float) <= st[c].size(); i += sizeof(float)) {
                    float a, b;
                    memcpy(&a, st[c].data() + i, sizeof(float));
                    memcpy(&b, st[2].data() + i, sizeof(float));
                    if (std::isfinite(a) && std::isfinite(b)) {
                        d_max = std::max(d_max, std::fabs(a - b));
                        a_max = std::max(a_max, std::fabs(b));
                    }
                }
                TLOG_INF("%s: seq %u %s state: max diff %g (max %g) against one sequence per ubatch\n", __func__, s, nm[c], (double) d_max, (double) a_max);
                // the states of a dummy model are small (about 4e-3 here); a moved window rolled back from another cell's rows differed
                // by 2% to 100% of that
                CTX_REQUIRE(d_max <= 1e-3f*a_max + 1e-7f, std::string("the ") + nm[c] + " state of seq " + std::to_string(s) +
                        " differs from one sequence per ubatch (max diff " + std::to_string(d_max) + " of " + std::to_string(a_max) + ")");
            }
        }

        // random dummy models drift a little between the two rollback paths; see case_multi_seq_split_replay
        constexpr float nmse_eps = 1e-4f;
        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "replay logits differ from the snapshot rows (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: replay matched the snapshot rows (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_replay_overlap>());

    return cases;
}
