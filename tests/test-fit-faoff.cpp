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
        llama_backend_free();
    } else {
        fprintf(stderr, "LLAMA_TEST_FIT_MODEL not set: the measure part is skipped\n");
    }

    if (n_fail == 0) {
        fprintf(stderr, "test-fit-faoff: OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
