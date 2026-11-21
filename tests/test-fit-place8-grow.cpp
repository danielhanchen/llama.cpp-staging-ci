// test-fit-place8-grow: the MoE cache size of a decode-sized --fit placement (common_fit_place8_grow_size, common/fit.h)
//
// The planner measures the placement through llama_model_load with no_alloc (a model file and a real device), so this simulates
// the measurements: total(cache) = rest + cache, over = free - total - margin.

#include "fit.h"

#include <cstdio>
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

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
