// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// common_speculative_types_from_gguf: the speculative type inferred from a draft GGUF when -md is given without --spec-type
// (metadata-only GGUFs written to the temp dir: architecture, block_count and the tensor names the inference looks at)

#include "ggml.h"
#include "gguf.h"
#include "speculative.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static std::string write_gguf(const std::string & name, const std::string & arch, uint32_t n_block,
                              const std::vector<std::string> & tensors) {
    ggml_init_params ip = { /* .mem_size = */ 1024 * 1024, /* .mem_buffer = */ nullptr, /* .no_alloc = */ false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * gctx = gguf_init_empty();
    gguf_set_val_str(gctx, "general.architecture", arch.c_str());
    gguf_set_val_u32(gctx, (arch + ".block_count").c_str(), n_block);
    for (const auto & tn : tensors) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_name(t, tn.c_str());
        gguf_add_tensor(gctx, t);
    }
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    const bool ok = gguf_write_to_file(gctx, path.c_str(), false);
    gguf_free(gctx);
    ggml_free(ctx);
    if (!ok) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        return {};
    }
    return path;
}

static int n_fail = 0;

static void check(const char * what, const std::string & path, const std::vector<common_speculative_type> & want) {
    const auto got = common_speculative_types_from_gguf(path);
    const bool ok = got == want;
    fprintf(stderr, "%s: %s (got %zu type(s)%s%s)\n", ok ? "PASS" : "FAIL", what, got.size(),
            got.empty() ? "" : ", first ", got.empty() ? "" : common_speculative_type_to_str(got[0]).c_str());
    n_fail += ok ? 0 : 1;
    std::filesystem::remove(path);
}

int main() {
    // gemma-4 assistant drafter (an MTP head with no nextn tensors, e.g. mtp-gemma-4-26B-A4B-it-Q8_0.gguf)
    check("gemma4-assistant -> draft-mtp",
          write_gguf("test-spec-gemma4-assistant.gguf", "gemma4-assistant", 4,
                     { "token_embd.weight", "blk.0.attn_q.weight", "blk.3.ffn_down.weight" }),
          { COMMON_SPECULATIVE_TYPE_DRAFT_MTP });

    // a model with a NextN layer (e.g. a Qwen3.6 / GLM MTP GGUF) -> draft-mtp
    check("nextn eh_proj -> draft-mtp",
          write_gguf("test-spec-nextn.gguf", "qwen35moe", 3, { "token_embd.weight", "blk.2.nextn.eh_proj.weight" }),
          { COMMON_SPECULATIVE_TYPE_DRAFT_MTP });

    // a plain draft model: no inference (the server keeps --spec-type as given)
    check("plain llama -> none",
          write_gguf("test-spec-plain.gguf", "llama", 2, { "token_embd.weight", "blk.1.attn_q.weight" }),
          {});

    // dflash / dspark
    check("dflash -> draft-dflash",
          write_gguf("test-spec-dflash.gguf", "dflash", 2, { "blk.0.attn_q.weight" }),
          { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH });
    check("dflash + markov head -> draft-dspark",
          write_gguf("test-spec-dspark.gguf", "dflash", 2, { "blk.0.attn_q.weight", "markov_w1.weight" }),
          { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK });

    // the server releases a -md draft model that no requested type drafts with (its test decode can abort on an assistant head)
    {
        const struct { std::vector<common_speculative_type> types; bool want; const char * what; } cases[] = {
            { { COMMON_SPECULATIVE_TYPE_NONE },                                          false, "none" },
            { { COMMON_SPECULATIVE_TYPE_NGRAM_MOD },                                     false, "ngram-mod" },
            { { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_NGRAM_CACHE }, false, "ngram-simple,ngram-cache" },
            { { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE },                                  true,  "draft-simple" },
            { { COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_DRAFT_MTP },  true,  "ngram-mod,draft-mtp" },
            { { COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 },                                  true,  "draft-eagle3" },
            { { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH },                                  true,  "draft-dflash" },
            { { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK },                                  true,  "draft-dspark" },
        };
        for (const auto & c : cases) {
            const bool ok = common_speculative_uses_draft_ctx(c.types) == c.want;
            fprintf(stderr, "%s: uses_draft_ctx(%s) == %d\n", ok ? "PASS" : "FAIL", c.what, (int) c.want);
            n_fail += ok ? 0 : 1;
        }
    }

    fprintf(stderr, "%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
