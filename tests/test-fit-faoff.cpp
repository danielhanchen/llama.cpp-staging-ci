// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// test-fit-faoff: the margin beside a MoE cache without flash attention (common_fit_moe_margin_fa_plan, common/fit.h) and the flash
// attention the fit's measure reads back from the context it created (common_fit_last_flash_attn, llama_get_flash_attn).
//
// The numbers are those of the run that ran out of memory: Kaggle 2x T4, DeepSeek-V4.1 fake dsf8 (24.1 GiB), -fa off, ctx 16384,
// --moe-cache-mib auto. The fit projected CUDA1 at 13507 MiB used of 14804 MiB free (1297 MiB free at the 1024 MiB margin). The reduced
// margin beside the cache is 256 MiB, so the cache took +768 MiB on each device, leaving 529 MiB. The first 2100-token prompt then took
// F16 copies of its KQ matrix from the CUDA pool, outside the measured compute buffer, and aborted in cuMemCreate.
//
// With a model path in LLAMA_TEST_FIT_MODEL (any small GGUF), the measure itself runs on the CPU for -fa off / on / auto and the flash
// attention it reads back must follow the context (off, on, and auto as the context resolved it).

#include "fit.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

int main() {
    constexpr int64_t MiB = 1024*1024;
    const int64_t m         = 1024*MiB;   // the default margin
    const int64_t m_reduced =  256*MiB;   // LLAMA_MOE_FIT_MARGIN beside a cache (k17-models)

    // the plan: the full margin without flash attention unless switched off, the reduced one with it
    CHECK(common_fit_moe_margin_fa_plan(m, m_reduced, /*flash_attn =*/ false, /*keep_faoff =*/ true)  == m);
    CHECK(common_fit_moe_margin_fa_plan(m, m_reduced, /*flash_attn =*/ true,  /*keep_faoff =*/ true)  == m_reduced);
    CHECK(common_fit_moe_margin_fa_plan(m, m_reduced, /*flash_attn =*/ false, /*keep_faoff =*/ false) == m_reduced);
    CHECK(common_fit_moe_margin_fa_plan(m, m,         /*flash_attn =*/ false, /*keep_faoff =*/ true)  == m);

    // the dsf8 placement: free after the cache took the margin gain, and the KQ copies of the 2100-token prompt
    {
        const int64_t free_cuda1   = 14804*MiB - 13507*MiB;   // projected at the full margin
        const int64_t gain         = m - m_reduced;           // per device
        const int64_t free_reduced = free_cuda1 - gain;
        const int64_t free_kept    = free_cuda1 - (m - common_fit_moe_margin_fa_plan(m, m_reduced, false, true));
        CHECK(free_reduced == 529*MiB);
        CHECK(free_kept    == 1297*MiB);
        // the full margin was not enough either: with it kept (k19-fitfaoff first step) the 7986-token prompt still aborted in a pool
        // allocation. One F16 copy of the KQ matrix at the full context (64 heads x 2048 x 16384) is what the margin has to hold
        const int64_t kq = common_fit_faoff_kq_bytes(64, 2048, 16384);
        CHECK(kq == int64_t(4096)*MiB);
        CHECK(kq > free_kept);
        // half the F32 KQ the measured compute buffer holds (CUDA1 10768 MiB, of which the F32 KQ is 8 GiB)
        CHECK(2*kq <= 10768*MiB);
        CHECK(common_fit_faoff_kq_bytes(-1, 2048, 16384) == 0);
    }

    // the n_ubatch choice without flash attention (common_fit_faoff_ubatch_plan): halve while the whole model plus the KQ copy reserve
    // at that ubatch misses a GPU's margin, not below 512; then layers move. dsf8 on 2x T4 (24.1 GiB of weights) fits at no ubatch: 512
    {
        std::vector<uint32_t> tried;
        auto never = [&](uint32_t ub) { tried.push_back(ub); return false; };
        CHECK(common_fit_faoff_ubatch_plan(2048, 512, never) == 512);
        CHECK((tried == std::vector<uint32_t>{ 2048, 1024 }));
        tried.clear();
        // a model that fits once the reserve is 2 GiB (64 heads x 1024 x 16384 x 2 B) keeps 1024
        const int64_t room = int64_t(2048)*MiB;
        auto room_fits = [&](uint32_t ub) { tried.push_back(ub); return common_fit_faoff_kq_bytes(64, ub, 16384) <= room; };
        CHECK(common_fit_faoff_ubatch_plan(2048, 512, room_fits) == 1024);
        CHECK((tried == std::vector<uint32_t>{ 2048, 1024 }));
        tried.clear();
        // one that fits at the given ubatch keeps it (one measure)
        CHECK(common_fit_faoff_ubatch_plan(2048, 512, [&](uint32_t ub) { tried.push_back(ub); return true; }) == 2048);
        CHECK(tried.size() == 1);
        tried.clear();
        // at or below the floor nothing is measured or changed; an odd ubatch stops at the floor
        CHECK(common_fit_faoff_ubatch_plan(512, 512, never) == 512);
        CHECK(common_fit_faoff_ubatch_plan(256, 512, never) == 256);
        CHECK(tried.empty());
        CHECK(common_fit_faoff_ubatch_plan(3000, 512, never) == 512);
        CHECK((tried == std::vector<uint32_t>{ 3000, 1500, 750 }));
        // the rule applies only without -ub, without flash attention, with both switches on and a GPU
        CHECK( common_fit_faoff_pin_ubatch(false, true,  true,  true,  true));
        CHECK(!common_fit_faoff_pin_ubatch(true,  true,  true,  true,  true));   // -ub given: kept, the reserve grows with it
        CHECK(!common_fit_faoff_pin_ubatch(false, false, true,  true,  true));   // flash attention on: unchanged
        CHECK(!common_fit_faoff_pin_ubatch(false, true,  false, true,  true));   // LLAMA_FIT_FAOFF_MARGIN=0
        CHECK(!common_fit_faoff_pin_ubatch(false, true,  true,  false, true));   // LLAMA_FIT_FAOFF_UB=0
        CHECK(!common_fit_faoff_pin_ubatch(false, true,  true,  true,  false));  // no GPU (no CUDA pool copies)
    }

    // the measure reads back the flash attention of the context it created
    const char * path = getenv("LLAMA_TEST_FIT_MODEL");
    if (path != nullptr && path[0] != '\0') {
        llama_backend_init();
        const llama_flash_attn_type types[3] = { LLAMA_FLASH_ATTN_TYPE_DISABLED, LLAMA_FLASH_ATTN_TYPE_ENABLED, LLAMA_FLASH_ATTN_TYPE_AUTO };
        bool got[3] = {};
        for (int i = 0; i < 3; i++) {
            llama_model_params   mp = llama_model_default_params();
            llama_context_params cp = llama_context_default_params();
            mp.n_gpu_layers    = 0;
            cp.n_ctx           = 512;
            cp.flash_attn_type = types[i];
            std::vector<ggml_backend_dev_t> devs;
            uint32_t ngl = 0, nct = 0, nex = 0;
            common_get_device_memory_data(path, &mp, &cp, devs, ngl, nct, nex, GGML_LOG_LEVEL_ERROR);
            got[i] = common_fit_last_flash_attn();
            fprintf(stderr, "measure with flash_attn_type %s: flash attention %s\n", llama_flash_attn_type_name(types[i]), got[i] ? "on" : "off");
        }
        CHECK(!got[0]);
        CHECK(got[1]);
        // auto resolves at the context; the CPU backend runs FLASH_ATTN_EXT, so on (the point is that the measure follows the context)
        CHECK(got[2]);

        // the whole fit without flash attention and without -ub on the CPU: no GPU, so n_ubatch stays and no choice is recorded
        {
            llama_model_params   mp = llama_model_default_params();
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx           = 512;
            cp.n_batch         = 2048;
            cp.n_ubatch        = 2048;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            std::vector<float> ts(llama_max_devices(), 0.0f);
            std::vector<llama_model_tensor_buft_override> tbo(llama_max_tensor_buft_overrides(), { nullptr, nullptr });
            std::vector<size_t> margins(llama_max_devices(), size_t(1024)*MiB);
            common_fit_set_ubatch_user(false);
            common_fit_faoff_ubatch_reset();
            common_fit_params(path, &mp, &cp, ts.data(), tbo.data(), margins.data(), 512, nullptr, GGML_LOG_LEVEL_ERROR);
            fprintf(stderr, "fit -fa off without -ub on the CPU: n_ubatch %u, chosen %u\n", cp.n_ubatch, common_fit_faoff_ubatch());
            CHECK(cp.n_ubatch == 2048);
            CHECK(common_fit_faoff_ubatch() == 0);
        }
        llama_backend_free();
    } else {
        fprintf(stderr, "LLAMA_TEST_FIT_MODEL not set: the measure part is skipped\n");
    }

    if (n_fail == 0) {
        fprintf(stderr, "test-fit-faoff: OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
