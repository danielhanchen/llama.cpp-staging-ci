// test-fit-placemid-lent: the mid-size --fit placement beside a lent drafter (common_fit_extra_model::lent, common/fit.h)
//
// The planner measures through llama_model_load with no_alloc (a model file and a real device), so this simulates the measurements
// of the mid-size branch with the same growth helpers it calls: total = rest + drafter + cache, over = free - total - margin. The cache
// it sizes is resident at decode beside the drafter's device weights and compute buffers, so a lent (context only) measurement
// grows the cache into them.

#include "common.h"
#include "fit.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static constexpr int64_t MiB = 1024*1024;

struct drafter {
    int64_t model;
    int64_t context;
    int64_t compute;

    // what one measurement adds for the drafter (the accounting of add_extra_memory)
    int64_t priced(bool lent, bool decode_sized) const {
        const bool excl = common_fit_extra_lent_excluded(lent, decode_sized);
        return context + (excl ? 0 : model + compute);
    }
    int64_t at_decode() const {
        return model + context + compute;
    }
};

// the mid-size branch: the full-ubatch placement (measured lent, as the fit before it), then the measurements at the small ubatch
static int64_t placemid(int64_t free, int64_t margin, int64_t rest_rel, int64_t prompt_compute, int64_t size_ok, const drafter & dr,
        bool decode_sized, int64_t & grow) {
    const int64_t tot_full = rest_rel + prompt_compute + dr.priced(true, false) + size_ok;
    const int64_t tot_rel  = rest_rel + dr.priced(true, decode_sized) + size_ok;
    int64_t ctx_ub = 0;
    common_fit_place8_grow_of(tot_full, 0, tot_rel, 0, grow, ctx_ub);
    return common_fit_place8_grow_size(size_ok, int64_t(1) << 40, grow, ctx_ub, [&](int64_t r, int64_t & over) {
        over = free - (rest_rel + dr.priced(true, decode_sized) + r) - margin;
        return true;
    });
}

int main() {
    // truth table of the lent accounting
    CHECK(!common_fit_extra_lent_excluded(false, false));
    CHECK(!common_fit_extra_lent_excluded(false, true));
    CHECK( common_fit_extra_lent_excluded(true,  false));
    CHECK(!common_fit_extra_lent_excluded(true,  true));

    // the switch: on by default, LLAMA_FIT_PLACEMID_DECODE_SIZED=0 measures lent as before
    common_set_env("LLAMA_FIT_PLACEMID_DECODE_SIZED", "");
    CHECK(common_fit_placemid_decode_sized());
    common_set_env("LLAMA_FIT_PLACEMID_DECODE_SIZED", "0");
    CHECK(!common_fit_placemid_decode_sized());
    common_set_env("LLAMA_FIT_PLACEMID_DECODE_SIZED", "1");
    CHECK(common_fit_placemid_decode_sized());
    common_set_env("LLAMA_FIT_PLACEMID_DECODE_SIZED", "");

    // RTX PRO 6000 at a 16 GB budget, Qwen3.8-Flash-Next 4-bit ctx 262144 -ub 2048 + its Q8_0 MTP drafter with host experts
    // (n_ubatch 16): 4154 MiB of prompt compute freed at 31 tokens, drafter 190 MiB weights, 340 MiB K/V, 176 MiB compute
    const int64_t free    = 15984*MiB;
    const int64_t margin  = 1024*MiB;
    const int64_t size_ok = 1233*MiB;
    const int64_t prompt  = 4154*MiB;
    const drafter dr      = { 190*MiB, 340*MiB, 176*MiB };
    const int64_t rest    = free - margin - size_ok - prompt - dr.context; // the full-ubatch placement meets the margin exactly

    for (const bool decode_sized : { false, true }) {
        int64_t grow = 0;
        const int64_t req = placemid(free, margin, rest, prompt, size_ok, dr, decode_sized, grow);
        CHECK(req > size_ok);
        const int64_t left = free - (rest + dr.at_decode() + req);
        printf("%s: grow %lld MiB, cache %lld -> %lld MiB, %lld MiB free at decode (margin %lld)\n",
            decode_sized ? "decode-sized" : "lent        ", (long long) (grow/MiB), (long long) (size_ok/MiB), (long long) (req/MiB),
            (long long) (left/MiB), (long long) (margin/MiB));
        if (decode_sized) {
            CHECK(grow == prompt - dr.model - dr.compute);
            CHECK(left >= margin);
            CHECK(left < margin + 128*MiB); // still grown up to the margin
        } else {
            CHECK(grow == prompt);
            CHECK(left < margin);
            CHECK(margin - left >= dr.model + dr.compute - 16*MiB); // the drafter's weights and compute come out of the margin
        }
    }

    // without a lent drafter both measurements agree
    {
        const drafter none = { 0, 0, 0 };
        int64_t g0 = 0, g1 = 0;
        const int64_t r0 = placemid(free, margin, rest + dr.context, prompt, size_ok, none, false, g0);
        const int64_t r1 = placemid(free, margin, rest + dr.context, prompt, size_ok, none, true,  g1);
        CHECK(g0 == g1 && r0 == r1);
    }

    if (n_fail != 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
