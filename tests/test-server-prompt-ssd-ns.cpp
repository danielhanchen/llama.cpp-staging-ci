// the namespace of the SSD tier of the server prompt cache (server_prompt_ssd_namespace) covers the load options that change the
// logits of the same tokens: a restart with another --override-kv on the same --cache-ssd-dir restored the state computed without it
//   - every metadata override (key, type and value) gives another namespace, the same overrides the same one
//   - a default configuration keeps its namespace (the options are added only off their defaults)

#include "../tools/server/server-prompt-ssd.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    printf("%-64s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail = 1;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    auto ns = [&](const std::vector<std::string> & overrides, auto && edit) {
        common_params params;
        params.model.path = argv[1];
        for (const auto & o : overrides) {
            if (!string_parse_kv_override(o.c_str(), params.kv_overrides)) {
                fprintf(stderr, "bad override %s\n", o.c_str());
                exit(1);
            }
        }
        if (!params.kv_overrides.empty()) {
            params.kv_overrides.emplace_back();
            params.kv_overrides.back().key[0] = 0;
        }
        edit(params);
        return server_prompt_ssd_namespace(params, model, ctx, nullptr, nullptr);
    };
    auto none = [](common_params &) {};

    const std::string base = ns({}, none);
    const std::string rope = ns({ "llama.rope.freq_base=float:500000" }, none);

    check("override changes the namespace", rope != base);
    check("same override, same namespace", ns({ "llama.rope.freq_base=float:500000" }, none) == rope);
    check("another float value", ns({ "llama.rope.freq_base=float:500001" }, none) != rope);
    check("float values closer than %g digits", ns({ "llama.rope.freq_base=float:500000.00001" }, none) != rope);
    check("another key", ns({ "llama.rope.scaling.factor=float:500000" }, none) != rope);
    check("int override", ns({ "llama.expert_used_count=int:1" }, none) != base);
    check("int value", ns({ "llama.expert_used_count=int:1" }, none) != ns({ "llama.expert_used_count=int:2" }, none));
    check("int vs float of the same value", ns({ "llama.context_length=int:1" }, none) != ns({ "llama.context_length=float:1" }, none));
    check("bool value", ns({ "llama.use_parallel_residual=bool:true" }, none) != ns({ "llama.use_parallel_residual=bool:false" }, none));
    check("str value", ns({ "tokenizer.chat_template=str:a" }, none) != ns({ "tokenizer.chat_template=str:b" }, none));
    check("two overrides vs one", ns({ "llama.rope.freq_base=float:500000", "llama.expert_used_count=int:1" }, none) != rope);
    // free text is escaped: a str value with a newline cannot forge a second override line, a ':' cannot move the key / value split
    const std::string forged = "a\nkv_override=llama.expert_used_count:" + std::to_string((int) LLAMA_KV_OVERRIDE_TYPE_INT) + ":1";
    check("newline in a str value vs two overrides", ns({ "tokenizer.chat_template=str:" + forged }, none) !=
        ns({ "tokenizer.chat_template=str:a", "llama.expert_used_count=int:1" }, none));
    const std::string t = std::to_string((int) LLAMA_KV_OVERRIDE_TYPE_STR);
    check("':' in a key vs in a str value", ns({ "k=str:a:" + t + ":b" }, none) != ns({ "k:" + t + ":a=str:b" }, none));
    check("non-causal attention", ns({}, [](common_params & p) { p.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }) != base);

    // defaults: no line is added, so the namespace of a default configuration is what it was
    check("no override, the default namespace", base.find("kv_override") == std::string::npos);
    check("default attention type adds nothing", base.find("attention_type") == std::string::npos);
    check("--lora-init-without-apply alone adds nothing", ns({}, [](common_params & p) { p.lora_init_without_apply = true; }) == base);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", g_fail ? "FAIL" : "OK");
    return g_fail;
}
