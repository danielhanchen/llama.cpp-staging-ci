// test-fit-place8-grow: the MoE cache size of a decode-sized --fit placement (common_fit_place8_grow_of / _size, common/fit.h)
//
// The planner measures the placement through llama_model_load with no_alloc (a model file and a real device), so this simulates
// the measurements: total(cache) = rest + cache, over = free - total - margin.

#include "fit.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static constexpr int64_t MiB = 1024*1024;

struct sim {
    int64_t free;
    int64_t rest; // everything but the cache at the small ubatch
    int64_t margin;
    int     n_meas = 0;

    bool operator()(int64_t req, int64_t & over) {
        n_meas++;
        over = free - (rest + req) - margin;
        return true;
    }
};

static int64_t grow_size(int64_t size_ok, int64_t cap, int64_t grow, int64_t ctx_ub, sim & s) {
    return common_fit_place8_grow_size(size_ok, cap, grow, ctx_ub, [&](int64_t r, int64_t & o) { return s(r, o); });
}

int main() {
    const int64_t cap_auto = int64_t(1) << 40;

    // --moe-cache-mib 512 fitted to 136 MiB beside the prompt graphs, 1700 MiB of them freed at the small ubatch
    {
        sim s = { 8000*MiB, 5000*MiB, 384*MiB };
        const int64_t req = grow_size(136*MiB, 512*MiB, 1700*MiB, 0, s);
        printf("cap 512 MiB: %lld MiB\n", (long long) (req/MiB));
        CHECK(req == 512*MiB);
        // unbounded request: the whole difference
        sim s2 = { 8000*MiB, 5000*MiB, 384*MiB };
        CHECK(grow_size(136*MiB, cap_auto, 1700*MiB, 0, s2) == 1836*MiB);
    }
    // a request the cache already holds: no growth, the placement of before stays
    {
        sim s = { 8000*MiB, 5000*MiB, 384*MiB };
        CHECK(grow_size(512*MiB, 512*MiB, 1700*MiB, 0, s) == -1);
        CHECK(s.n_meas == 0);
        sim s2 = { 8000*MiB, 5000*MiB, 384*MiB };
        CHECK(grow_size(450*MiB, 512*MiB, 1700*MiB, 0, s2) == -1); // under 128 MiB of growth left
    }
    // the margin decides: the request comes down by the miss and 16 MiB
    {
        sim s = { 8000*MiB, 6000*MiB, 384*MiB };
        const int64_t req = grow_size(136*MiB, cap_auto, 1700*MiB, 0, s);
        CHECK(req == 8000*MiB - 6000*MiB - 384*MiB - 16*MiB);
        CHECK(s.n_meas == 2);
        int64_t over = 0;
        s(req, over);
        CHECK(over >= 0);
    }
    // context memory the measurement does not hold (ctx_ub) is charged in the check
    {
        sim s = { 8000*MiB, 5000*MiB, 384*MiB };
        const int64_t ctx_ub = 350*MiB;
        const int64_t req = grow_size(136*MiB, cap_auto, 1700*MiB, ctx_ub, s);
        CHECK(req > 0);
        int64_t over = 0;
        s(req, over);
        CHECK(over - ctx_ub >= 0);
    }
    // a failed measurement: no larger cache
    {
        const int64_t req = common_fit_place8_grow_size(136*MiB, cap_auto, 1700*MiB, 0, [](int64_t, int64_t &) { return false; });
        CHECK(req == -1);
    }

    // Gemma 4 26B-A4B (25 of 30 layers SWA, n_swa 1024, 8 KV heads x 256, f16 K/V: 200 KiB per SWA cell), one sequence, -c 32768 -ub 2048:
    // the SWA cache holds GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256) cells (llama_kv_cache_iswa), 3072 at the full ubatch, 1280
    // at the 31 tokens of the measurement. The context is created at the full ubatch, only the graphs are reserved at 31 tokens
    {
        auto swa_cells = [](int64_t n_ctx, int64_t n_swa, int64_t ub) {
            const int64_t n = std::min(n_ctx, n_swa + ub);
            return (n + 255) / 256 * 256;
        };
        const int64_t cell    = 25*2*8*256*2;
        const int64_t free    = 7264*MiB; // the full-ubatch placement meets the margin exactly
        const int64_t model   = 4200*MiB;
        const int64_t kv_full = 5*2*2*512*2*32768; // the 5 full-attention layers, 2 KV heads x 512
        const int64_t margin  = 384*MiB;
        auto ctx = [&](int64_t ub) { return kv_full + swa_cells(32768, 1024, ub)*cell; };
        auto cmp = [&](int64_t ub) { return ub > 31 ? int64_t(1300)*MiB : int64_t(80)*MiB; };
        const int64_t size_ok = 140*MiB;
        const int64_t ctx_full = ctx(2048) + size_ok; // the cache is counted as context
        const int64_t ctx_rel  = ctx(31)   + size_ok;
        const int64_t tot_full = model + ctx_full + cmp(2048);
        const int64_t tot_rel  = model + ctx_rel  + cmp(31);
        CHECK(ctx(2048) - ctx(31) == 1792*cell); // 350 MiB
        CHECK(free - tot_full >= margin);
        int64_t grow = 0, ctx_ub = 0;
        common_fit_place8_grow_of(tot_full, ctx_full, tot_rel, ctx_rel, grow, ctx_ub);
        CHECK(ctx_ub == 350*MiB);
        CHECK(grow == 1220*MiB);
        sim s = { free, model + ctx(31) + cmp(31), margin };
        const int64_t req = grow_size(size_ok, cap_auto, grow, ctx_ub, s);
        CHECK(req > size_ok);
        // at run time: the context at the full ubatch, the graphs at 31 tokens
        const int64_t used = model + ctx(2048) + req + cmp(31);
        printf("Gemma 4 SWA -ub 2048: cache %lld -> %lld MiB, %lld MiB free at run time (margin %lld)\n", (long long) (size_ok/MiB),
            (long long) (req/MiB), (long long) ((free - used)/MiB), (long long) (margin/MiB));
        CHECK(free - used >= margin);
        // without the SWA growth charged (the difference counted as compute) the run time leaves less than the margin
        sim s0 = { free, model + ctx(31) + cmp(31), margin };
        const int64_t req0 = grow_size(size_ok, cap_auto, tot_full - tot_rel, 0, s0);
        CHECK(free - (model + ctx(2048) + req0 + cmp(31)) < margin);
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
