// test-moe-kick-wait: a kick wakes the idle host thread of the MoE cache's single-graph path (src/llama-kick-wait.h)
//
// The sleeper takes the kick stamp, then waits for a posted region or a kick. Checked without timing assumptions: with a long timeout
// every kick must end the wait as a kick (and one that lands before the wait must not be lost); the previous wait (no stamp in the
// predicate, LLAMA_MOE_SINGLE_KICK_GEN=0) only ends at its timeout. The wake latency at the real 1 ms timeout is printed for both.

#include "../src/llama-kick-wait.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

using clk = std::chrono::steady_clock;

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now().time_since_epoch()).count();
}

struct hub {
    std::mutex              mtx;
    std::condition_variable cv;
    std::atomic<int64_t>    stamp{0};
    std::atomic<int>        posted{0}; // the run_state == 1 of the real thread: never set here
};

struct trial {
    bool    woken   = false; // the wait ended by its predicate (a kick)
    int64_t latency = 0;     // kick -> the sleeper running again, us (negative: it timed out before the kick)
};

// one sleeper per trial; the kick comes after the sleeper took its stamp (early = before it waits, else while it waits)
static std::vector<trial> run(int n, std::chrono::microseconds timeout, bool use_stamp, bool early) {
    std::vector<trial> out(n);
    for (int i = 0; i < n; ++i) {
        hub h;
        std::atomic<int>     armed{0};
        std::atomic<int64_t> t_wake{0};
        std::atomic<int64_t> t_arm{0};
        bool                 woken = false;
        std::thread th([&] {
            const int64_t s0 = h.stamp.load(std::memory_order_acquire);
            t_arm.store(now_us(), std::memory_order_relaxed);
            armed.store(1, std::memory_order_release);
            if (early) {
                while (armed.load(std::memory_order_acquire) != 2) {
                    std::this_thread::yield();
                }
            }
            woken = llama_kick_wait_for(h.mtx, h.cv, h.stamp, s0, timeout, use_stamp, [&h] { return h.posted.load() == 1; });
            t_wake.store(now_us(), std::memory_order_release);
        });
        while (armed.load(std::memory_order_acquire) != 1) {
            std::this_thread::yield();
        }
        if (!early) {
            // 100 us after the sleeper took its stamp: it is (most likely) in the wait
            while (now_us() < t_arm.load(std::memory_order_relaxed) + 100) {
            }
        }
        const int64_t t_kick = now_us();
        h.stamp.store(t_kick, std::memory_order_release);
        llama_kick_notify(h.mtx, h.cv);
        if (early) {
            armed.store(2, std::memory_order_release);
        }
        th.join();
        out[i].woken   = woken;
        out[i].latency = t_wake.load() - t_kick;
    }
    return out;
}

static void report(const char * what, std::vector<trial> r) {
    int nw = 0;
    std::vector<int64_t> lat; // only the trials whose sleeper was still waiting at the kick
    for (const auto & t : r) {
        nw += t.woken;
        if (t.latency >= 0) {
            lat.push_back(t.latency);
        }
    }
    if (lat.empty()) {
        lat.push_back(-1);
    }
    std::sort(lat.begin(), lat.end());
    printf("%-44s woken by the kick %3d / %3zu, wake latency us (%3zu waiting at the kick): p50 %5lld p90 %5lld max %5lld\n", what, nw,
        r.size(), lat.size(), (long long) lat[lat.size()/2], (long long) lat[lat.size()*9/10], (long long) lat.back());
}

int main() {
    using std::chrono::milliseconds;

    // the checks: a long timeout, so only the predicate can end the wait early
    {
        const auto r = run(50, milliseconds(5000), true, false);
        int nw = 0;
        for (const auto & t : r) {
            nw += t.woken;
            CHECK(t.latency < 4000000);
        }
        CHECK(nw == 50);
        report("stamp in the predicate, kick during the wait", r);
    }
    {
        // a kick between the stamp and the wait is not lost
        const auto r = run(50, milliseconds(5000), true, true);
        int nw = 0;
        for (const auto & t : r) {
            nw += t.woken;
        }
        CHECK(nw == 50);
        report("stamp in the predicate, kick before the wait", r);
    }
    {
        // the previous wait: the kick is not part of the predicate, the wait runs to its timeout
        const auto r = run(5, milliseconds(50), false, false);
        for (const auto & t : r) {
            CHECK(!t.woken);
            CHECK(t.latency >= 10000);
        }
        report("previous wait (50 ms timeout)", r);
    }

    // the numbers at the real 1 ms timeout (printed, not checked: they depend on the scheduler)
    report("1 ms timeout, stamp in the predicate", run(200, milliseconds(1), true, false));
    report("1 ms timeout, previous wait", run(200, milliseconds(1), false, false));

    if (n_fail) {
        fprintf(stderr, "test-moe-kick-wait: %d failures\n", n_fail);
        return 1;
    }
    printf("test-moe-kick-wait: OK\n");
    return 0;
}
