// test-fit-plain-spill: the margin step-down of a plain dense launch (common/fit.h): the step list, the trial margins it
// gives and the display gate. The fit itself needs a model file and a device; these are the rules it applies around the fit:
//   - the default list 512,448,256,192 MiB tried from the 1024 MiB default margin, never above a margin given
//   - the step-down acts only where every used device reports no display (1); a display (0) or an unknown signal (-1) keeps the margin
//   - LLAMA_FIT_PLAIN_HEADLESS=1 / 0 forces it either way

#include "fit.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static constexpr size_t MiB = 1024*1024;

int main() {
    // the step list
    {
        const std::vector<size_t> st = common_fit_margin_steps_mib("512,448,256,192");
        CHECK(st.size() == 4);
        CHECK(st.size() == 4 && st[0] == 512*MiB && st[1] == 448*MiB && st[2] == 256*MiB && st[3] == 192*MiB);
        CHECK(common_fit_margin_steps_mib("").empty());
        CHECK(common_fit_margin_steps_mib(nullptr).empty());
        const std::vector<size_t> st2 = common_fit_margin_steps_mib("256,,-5,");
        CHECK(st2.size() == 2 && st2[0] == 256*MiB && st2[1] == 0);
    }
    // trials from the 1024 MiB default (one device)
    {
        const std::vector<size_t> steps = common_fit_margin_steps_mib("512,448,256,192");
        const auto tr = common_fit_margin_trials({1024*MiB}, steps);
        CHECK(tr.size() == 4);
        CHECK(tr.size() == 4 && tr[0][0] == 512*MiB && tr[1][0] == 448*MiB && tr[2][0] == 256*MiB && tr[3][0] == 192*MiB);
        // never above a margin given: a 300 MiB margin only tries 256 and 192
        const auto tr2 = common_fit_margin_trials({300*MiB}, steps);
        CHECK(tr2.size() == 2 && tr2[0][0] == 256*MiB && tr2[1][0] == 192*MiB);
        // a margin at or below every step: no trial
        CHECK(common_fit_margin_trials({128*MiB}, steps).empty());
        // two devices: each lowered to the step, one already below stays; equal consecutive trials collapse
        const auto tr3 = common_fit_margin_trials({1024*MiB, 200*MiB}, steps);
        CHECK(tr3.size() == 4);
        CHECK(tr3.size() == 4 && tr3[2][0] == 256*MiB && tr3[2][1] == 200*MiB && tr3[3][0] == 192*MiB && tr3[3][1] == 192*MiB);
        const auto tr4 = common_fit_margin_trials({1024*MiB}, common_fit_margin_steps_mib("512,512,256"));
        CHECK(tr4.size() == 2);
    }
    // a reserve on top of the default margin (a 768 MiB projector loaded after the model) stays on top of every step
    {
        const std::vector<size_t> steps = common_fit_margin_steps_mib("512,448,256,192");
        const auto tr = common_fit_margin_trials({(1024 + 768)*MiB}, steps, {768*MiB});
        CHECK(tr.size() == 4);
        CHECK(tr.size() == 4 && tr[0][0] == (512 + 768)*MiB && tr[1][0] == (448 + 768)*MiB && tr[2][0] == (256 + 768)*MiB &&
              tr[3][0] == (192 + 768)*MiB);
        // no reserve: the same trials as without the argument
        const auto tr0 = common_fit_margin_trials({1024*MiB, 200*MiB}, steps, {0, 0});
        CHECK(tr0 == common_fit_margin_trials({1024*MiB, 200*MiB}, steps));
        // per device: only the device with the projector keeps it
        const auto tr2 = common_fit_margin_trials({(1024 + 300)*MiB, 1024*MiB}, steps, {300*MiB, 0});
        CHECK(tr2.size() == 4 && tr2[0][0] == (512 + 300)*MiB && tr2[0][1] == 512*MiB);
    }
    // the reserve is the part of each margin above the default; one step of the MTP drafter squeeze (512, then 448 MiB) keeps it
    {
        const std::vector<size_t> res = common_fit_margin_reserve({(1024 + 880)*MiB, 1024*MiB, 512*MiB}, {1024*MiB, 1024*MiB});
        CHECK(res.size() == 3 && res[0] == 880*MiB && res[1] == 0 && res[2] == 0);
        const auto sq = common_fit_margin_trials({(1024 + 880)*MiB}, {512*MiB}, common_fit_margin_reserve({(1024 + 880)*MiB}, {1024*MiB}));
        CHECK(sq.size() == 1 && sq[0][0] == (512 + 880)*MiB);
        const auto sq2 = common_fit_margin_trials({(1024 + 880)*MiB}, {448*MiB}, common_fit_margin_reserve({(1024 + 880)*MiB}, {1024*MiB}));
        CHECK(sq2.size() == 1 && sq2[0][0] == (448 + 880)*MiB);
        // without a projector the squeeze step is the plain cap, as before
        const auto sq0 = common_fit_margin_trials({1024*MiB}, {512*MiB}, common_fit_margin_reserve({1024*MiB}, {1024*MiB}));
        CHECK(sq0.size() == 1 && sq0[0][0] == 512*MiB);
        // a margin already at or below the step (a --fit-target below it) is not a step
        CHECK(common_fit_margin_trials({400*MiB}, {512*MiB}, common_fit_margin_reserve({400*MiB}, {1024*MiB})).empty());
    }
    // a GPU projector loaded after the fit: the step-down only where its reserve is in the margins (the server, default devices)
    {
        CHECK( common_fit_plain_spill_proj_ok(false, false, {}));          // no projector
        CHECK( common_fit_plain_spill_proj_ok(false, true,  {0}));         // no projector, a device list
        CHECK(!common_fit_plain_spill_proj_ok(true,  false, {0}));         // llama-mtmd-cli: nothing reserved
        CHECK(!common_fit_plain_spill_proj_ok(true,  false, {}));
        CHECK( common_fit_plain_spill_proj_ok(true,  false, {768*MiB}));   // the server: reserve on the fitted device
        CHECK(!common_fit_plain_spill_proj_ok(true,  true,  {0, 768*MiB})); // --device: the reserve may sit in another slot
    }
    // the display gate
    {
        CHECK( common_fit_plain_spill_gate({1}, nullptr));
        CHECK( common_fit_plain_spill_gate({1, 1}, nullptr));
        CHECK(!common_fit_plain_spill_gate({1, 0}, nullptr));
        CHECK(!common_fit_plain_spill_gate({0}, nullptr));
        CHECK(!common_fit_plain_spill_gate({-1}, nullptr));
        CHECK(!common_fit_plain_spill_gate({1, -1}, nullptr));
        CHECK(!common_fit_plain_spill_gate({}, nullptr));      // no device known: keep the margin
        CHECK( common_fit_plain_spill_gate({1}, ""));          // empty = unset
        CHECK(!common_fit_plain_spill_gate({0}, ""));
        CHECK( common_fit_plain_spill_gate({0}, "1"));
        CHECK( common_fit_plain_spill_gate({-1}, "1"));
        CHECK(!common_fit_plain_spill_gate({1}, "0"));
    }
    // a device the CPU backend owns has no display signal: unknown
    {
        std::vector<ggml_backend_dev_t> devs;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                devs.push_back(dev);
            }
        }
        const std::vector<int> h = common_fit_devices_headless(devs);
        CHECK(h.size() == devs.size());
        for (int x : h) {
            CHECK(x == -1);
        }
        CHECK(common_fit_devices_headless({nullptr}).empty());
    }
    if (n_fail == 0) {
        printf("test-fit-plain-spill: OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
