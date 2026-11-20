// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// --moe-cache-mib on a GPU that is not ggml-cuda (Metal, Vulkan): the MoE cache takes only CUDA / ROCm / MUSA devices, so a model whose
// host experts belong to layers of such a GPU has nothing to cache, and the context must run without the cache instead of failing
// ("MoE cache requires a CUDA / ROCm / MUSA GPU backend" from the constructor). The GPU here is a mock: the CPU device under another
// registry name with the GPU type, so every weight is a host weight of a layer on a non-ggml-cuda GPU.
#include "llama.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>

static ggml_backend_device mock_dev;
static ggml_backend_reg    mock_reg;
static ggml_backend_reg_t  cpu_reg;

static const char * mock_dev_name(ggml_backend_dev_t) { return "MockGPU0"; }
static const char * mock_dev_desc(ggml_backend_dev_t) { return "mock GPU (the CPU under another registry)"; }
static enum ggml_backend_dev_type mock_dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void mock_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    cpu_reg->iface.get_device(cpu_reg, 0)->iface.get_props(dev, props);
    props->name        = mock_dev_name(dev);
    props->description = mock_dev_desc(dev);
    props->type        = GGML_BACKEND_DEVICE_TYPE_GPU;
}
static const char * mock_reg_name(ggml_backend_reg_t) { return "Mock"; }
static size_t mock_reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t mock_reg_dev(ggml_backend_reg_t, size_t) { return &mock_dev; }
static void * mock_reg_proc(ggml_backend_reg_t, const char * name) { return cpu_reg->iface.get_proc_address(cpu_reg, name); }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <MoE model.gguf>\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        fprintf(stderr, "no CPU device\n");
        return 1;
    }
    cpu_reg = ggml_backend_dev_backend_reg(cpu);

    mock_reg = *cpu_reg;
    mock_reg.iface.get_name         = mock_reg_name;
    mock_reg.iface.get_device_count = mock_reg_count;
    mock_reg.iface.get_device       = mock_reg_dev;
    mock_reg.iface.get_proc_address = mock_reg_proc;
    mock_dev = *cpu;
    mock_dev.reg                    = &mock_reg;
    mock_dev.iface.get_name         = mock_dev_name;
    mock_dev.iface.get_description  = mock_dev_desc;
    mock_dev.iface.get_type         = mock_dev_type;
    mock_dev.iface.get_props        = mock_dev_props;

    llama_backend_init();
    ggml_backend_dev_t devs[2] = { &mock_dev, nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.devices      = devs;
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "FAIL: could not load %s\n", argv[1]);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx          = 256;
    cp.n_batch        = 64;
    cp.n_ubatch       = 64;
    cp.moe_cache_size = (size_t) 64 << 20;
    llama_context * ctx = llama_init_from_model(model, cp);
    int rc = 0;
    if (ctx == nullptr) {
        fprintf(stderr, "FAIL: --moe-cache-mib 64 on a non-ggml-cuda GPU failed the context\n");
        rc = 1;
    } else {
        // and it decodes
        llama_token tok[4] = { 1, 2, 3, 4 };
        const int r = llama_decode(ctx, llama_batch_get_one(tok, 4));
        if (r != 0) {
            fprintf(stderr, "FAIL: decode returned %d\n", r);
            rc = 1;
        } else {
            fprintf(stderr, "OK: the context runs without the MoE cache\n");
        }
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
