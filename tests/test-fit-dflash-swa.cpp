// test-fit-dflash-swa: the K/V cells --fit prices for a DFlash drafter whose layers all slide (common_fit_dflash_draft_bytes)
//
// A tiny DFlash GGUF (2 sliding layers, window 128) is written to a temporary file. Without --swa-full the cache holds the
// window plus one ubatch, padded to 256 (llama_kv_cache_iswa); with --swa-full the sliding layers hold the whole context.

#include "common.h"
#include "gguf.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

int main() {
    const int64_t n_layer = 2, n_embd = 64, n_head = 4, n_head_kv = 2, d_k = 16, n_swa = 128;

    ggml_init_params ip = { /*.mem_size =*/ 1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * fc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3*n_embd, n_embd);
    ggml_set_name(fc, "fc.weight");

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "dflash");
    gguf_set_val_u32(g, "dflash.block_count", n_layer);
    gguf_set_val_u32(g, "dflash.embedding_length", n_embd);
    gguf_set_val_u32(g, "dflash.attention.head_count", n_head);
    gguf_set_val_u32(g, "dflash.attention.head_count_kv", n_head_kv);
    gguf_set_val_u32(g, "dflash.attention.key_length", d_k);
    gguf_set_val_u32(g, "dflash.attention.sliding_window", n_swa);
    const bool pat[2] = { true, true };
    gguf_set_arr_data(g, "dflash.attention.sliding_window_pattern", GGUF_TYPE_BOOL, pat, 2);
    std::vector<std::string> toks;
    for (int i = 0; i < 100; ++i) {
        toks.push_back("t" + std::to_string(i));
    }
    std::vector<const char *> tok_ptrs;
    for (const auto & t : toks) {
        tok_ptrs.push_back(t.c_str());
    }
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tok_ptrs.data(), tok_ptrs.size());
    gguf_add_tensor(g, fc);

    const char * tmp = getenv("TMPDIR");
    const std::string path = std::string(tmp != nullptr ? tmp : "/tmp") + "/test-fit-dflash-swa-" + std::to_string((long long) ggml_time_us()) + ".gguf";
    if (!gguf_write_to_file(g, path.c_str(), false)) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return 1;
    }
    gguf_free(g);
    ggml_free(ctx);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 8192;
    cp.n_ubatch  = 512;
    cp.n_seq_max = 1;
    cp.type_k    = GGML_TYPE_F16;
    cp.type_v    = GGML_TYPE_F16;

    const size_t row = (size_t) n_head_kv*(d_k + d_k)*2; // K + V, f16

    size_t w = 0, kv = 0, comp = 0;
    cp.swa_full = false;
    CHECK(common_fit_dflash_draft_bytes(path.c_str(), cp, nullptr, 0, w, kv, comp));
    CHECK(kv == (size_t) n_layer*GGML_PAD(n_swa + 512, 256)*row);

    cp.swa_full = true;
    CHECK(common_fit_dflash_draft_bytes(path.c_str(), cp, nullptr, 0, w, kv, comp));
    CHECK(kv == (size_t) n_layer*cp.n_ctx*row);

    remove(path.c_str());

    if (n_fail == 0) {
        printf("OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
