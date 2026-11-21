// test-fit-small-reserve: the run-time reserve of a small-device --fit placement (common_fit_small_reserve_plan / _mib, common/fit.h)
//
// The planner itself needs a model file and a real device (it measures the placement through llama_model_load with no_alloc), so
// this checks the arithmetic it applies, with the numbers of the placement that ran out of memory: RTX PRO 6000 capped at 8 GB
// (7792 MiB free), Qwen3.8-Flash-Next 4-bit, -c 32768 -ub 2048, Q8_0 MTP drafter dropped by --fit:
//   - the small-device rule kept every layer at a margin of 256 MiB beside a 141 MiB cache, 277 MiB projected free
//   - 110.56 MiB were free at the first 8k layer-major group with the route tail skip on (166.4 MiB taken by buffers the measurement
//     does not have), none with it off (the CUDA graphs of the chat decode failed to instantiate, then a stream creation aborted)
//   - the scheduler merges layer-major ubatches only with 128 MiB of free memory left beside them
//   - the 141 MiB cache held 48 slots of 2.27 MiB in its 43-layer group (109.15 MiB), 14.19 MiB were kept for the group graphs
// The plan has to leave the measured run-time use plus the scheduler's headroom free, and the cache group at least n_expert_used
// (10) slots.

#include "fit.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

int main() {
    // the defaults are read once, at the first call
#ifdef _WIN32
    _putenv("LLAMA_FIT_SMALL_RESERVE_MIB=");
    _putenv("LLAMA_FIT_SMALL_RESERVE_MIN_CACHE_MIB=");
#else
    unsetenv("LLAMA_FIT_SMALL_RESERVE_MIB");
    unsetenv("LLAMA_FIT_SMALL_RESERVE_MIN_CACHE_MIB");
#endif

    // RTX PRO 6000 capped at 8 GB Flash-Next
    {
        const int64_t free_mib       = 7792;
        const int64_t used_mib       = 7514;   // model 4459 + context 742 + compute 2312, cache 141 included
        const int64_t margin_mib     = 256;
        const int64_t cache_mib      = 141;
        const double  runtime_mib    = 277.0 - 110.56;  // taken after the fit before the first 8k group (tail skip on)
        const double  sched_headroom = 128.0;
        const int64_t extra = common_fit_small_reserve_mib(cache_mib);
        CHECK(extra == 77);
        CHECK(extra == common_fit_small_reserve_plan(cache_mib, 128, 64));
        const int64_t free_after = free_mib - (used_mib - extra);   // the cache gives the reserve
        CHECK(free_after >= margin_mib + extra);
        CHECK((double) free_after >= runtime_mib + sched_headroom);
        // before: 277 MiB, less than what the run took plus the headroom
        CHECK((double) (free_mib - used_mib) < runtime_mib + sched_headroom);
        // the cache keeps a working group: 64 MiB less the group-graph share and the overhead seen at 141 MiB
        const double overhead = 141.0 - 14.19 - 109.15;
        const double slots    = (double) (cache_mib - extra - 14.19 - overhead) / (109.15/48.0);
        CHECK(slots >= 10.0);
        printf("G4 8 GB Flash-Next: reserve %lld MiB, cache %lld MiB (~%.1f slots), %lld MiB projected free (before %lld), need %.1f\n",
            (long long) extra, (long long) (cache_mib - extra), slots, (long long) free_after, (long long) (free_mib - used_mib),
            runtime_mib + sched_headroom);
    }

    // the plan: capped at the reserve, never below the floor, off at 0
    CHECK(common_fit_small_reserve_plan(1000, 128, 64) == 128);
    CHECK(common_fit_small_reserve_plan( 192, 128, 64) == 128);
    CHECK(common_fit_small_reserve_plan( 100, 128, 64) ==  36);
    CHECK(common_fit_small_reserve_plan(  64, 128, 64) ==   0);
    CHECK(common_fit_small_reserve_plan(  48, 128, 64) ==   0);   // the opt-in 48 MiB cap (LLAMA_FIT_SMALL_CACHE_MIB) stays as it is
    CHECK(common_fit_small_reserve_plan(   0, 128, 64) ==   0);
    CHECK(common_fit_small_reserve_plan( 141,   0, 64) ==   0);   // LLAMA_FIT_SMALL_RESERVE_MIB=0: the margin of before
    CHECK(common_fit_small_reserve_plan( 141, -5, 64) ==   0);
    CHECK(common_fit_small_reserve_plan( 141, 128,  0) == 128);
    for (int64_t c = 0; c <= 512; ++c) {
        const int64_t e = common_fit_small_reserve_plan(c, 128, 64);
        CHECK(e >= 0 && e <= 128);
        CHECK(e == 0 || c - e >= 64);
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
