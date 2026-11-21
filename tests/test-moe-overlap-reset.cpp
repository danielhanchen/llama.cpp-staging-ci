// test-moe-overlap-reset: the fetch overlap rule of llama-moe-cache.cpp (ov_rule_tick) and the request reset of a fixed share
// (hybrid_reset), reduced. With the deferred admission (fetch_defer) a single token ubatch computes its admitted experts on the host
// only while the overlap is on, so the rule's verdict changes the output. A fixed-share request must give the same output whatever
// ran before it: after the reset the verdicts of a request are the ones of a fresh load. Checked here, on the CPU:
//   - a fresh load: a request with few missing bytes keeps the overlap off (every token computes its admitted experts on the device);
//   - without the overlap state in the reset, the same request after one with many missing bytes defers (the bug);
//   - with it, the same request gives the fresh load's verdicts token for token.

#include <cstdint>
#include <cstdio>
#include <vector>

static int fails = 0;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

struct ov_rule_sim {
    static constexpr int    ov_rule_skip = 8;
    static constexpr int    ov_rule_win  = 32;
    static constexpr int    ov_rule_win2 = 128;
    static constexpr double ov_rule_hyst = 0.75;

    bool     ov_rule  = true;
    double   ov_min   = 4.0*1024*1024; // the CUDA default from compute capability 9.0
    bool     ov_live  = !ov_rule;
    int      ov_seen  = 0;
    int      ov_win   = ov_rule_win;
    int      ov_n1    = 0;
    bool     ov_have0 = false;
    uint64_t ov_m0    = 0;
    size_t   n_layers = 40;

    uint64_t miss_bytes = 0; // the device lookup's miss counter (bytes); not cleared by the reset

    void tick(int64_t n_tokens) {
        if (n_tokens != 1) {
            ov_have0 = false;
            ov_n1    = 0;
            return;
        }
        if (ov_seen < ov_rule_skip) {
            ov_seen++;
            return;
        }
        if (!ov_have0) {
            ov_m0    = miss_bytes;
            ov_have0 = true;
            ov_n1    = 0;
            return;
        }
        if (++ov_n1 < ov_win) {
            return;
        }
        const double per = (double) (miss_bytes - ov_m0)/((double) ov_n1*n_layers);
        const bool   on  = ov_live ? per >= ov_rule_hyst*ov_min : per >= ov_min;
        ov_live = on;
        ov_m0   = miss_bytes;
        ov_n1   = 0;
        ov_win  = ov_rule_win2;
    }

    void reset(bool with_overlap) {
        if (with_overlap && ov_rule) {
            ov_live  = false;
            ov_seen  = 0;
            ov_win   = ov_rule_win;
            ov_n1    = 0;
            ov_have0 = false;
            ov_m0    = 0;
        }
    }

    // a request: a prompt ubatch, then n single tokens each missing per_tok bytes in every layer; returns the deferral
    // verdict of every single token (fetch_defer for n_ubatch == 1 needs overlap_batch, i.e. ov_live)
    std::vector<bool> request(int n, uint64_t per_tok) {
        std::vector<bool> d;
        tick(64);
        for (int i = 0; i < n; ++i) {
            tick(1);
            d.push_back(ov_live);
            miss_bytes += per_tok*n_layers;
        }
        return d;
    }
};

int main() {
    const uint64_t few  = 1u << 20;  // 1 MiB per layer and token: below the 3 MiB hysteresis floor
    const uint64_t many = 16u << 20; // 16 MiB: above the 4 MiB threshold

    ov_rule_sim fresh;
    const std::vector<bool> ref = fresh.request(200, few);
    bool any = false;
    for (bool b : ref) {
        any = any || b;
    }
    CHECK(!any, "a fresh load with few missing bytes defers");

    for (int with = 0; with < 2; ++with) {
        ov_rule_sim s;
        s.request(200, many);
        CHECK(s.ov_live, "the first request does not turn the overlap on");
        s.reset(with != 0);
        const std::vector<bool> got = s.request(200, few);
        if (with) {
            CHECK(got == ref, "after the reset the request's verdicts differ from a fresh load");
        } else {
            CHECK(got != ref, "control: without the overlap reset the history does not show");
        }
    }

    if (fails == 0) {
        printf("OK\n");
    }
    return fails == 0 ? 0 : 1;
}
