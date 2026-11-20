// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-hostgap: server_queue::yield_to_queue with and without the fast end (LLAMA_SERVER_YIELD_FAST). The main loop runs many short
// yields (as a speculative pass does: draft, verify, the drafter's pass) while another thread posts tasks at random times. Checked:
// every task is processed exactly once; a task is processed by the worker only while the main thread is inside yield_to_queue() (the
// server's slot code outside a yield never runs beside a task); tasks queued when a yield starts are processed during that yield as
// before; the loop terminates. The yield-end time is printed (median, us).
// usage: test-server-queue-yield [iterations]   (LLAMA_SERVER_YIELD_FAST=0 / 1 set by ctest)
#include "server-queue.h"
#include "server-task.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void spin_us(int us) {
    const int64_t t_end = now_ns() + (int64_t) us * 1000;
    while (now_ns() < t_end) {
    }
}

int main(int argc, char ** argv) {
    const int n_iter = argc > 1 ? atoi(argv[1]) : 3000;

    server_queue q;

    std::atomic<bool> in_yield{false};     // the main thread is inside yield_to_queue()
    std::atomic<bool> outside{false};      // the main thread runs slot code outside a yield
    std::atomic<int>  violations{0};
    std::atomic<int>  n_worker_tasks{0};
    std::atomic<int>  n_main_tasks{0};
    std::atomic<int>  n_queued_at_start_missed{0};
    std::mutex        m_seen;
    std::vector<int>  seen;                // ids of processed tasks (the poster's tasks only)
    std::atomic<int>  n_posted{0};
    std::atomic<bool> stop_poster{false};
    std::atomic<int>  pending_mark{-1};    // a task posted right before a yield: it must be processed during that yield

    q.on_new_task([&](server_task && task, bool is_yielding) {
        if (outside.load()) {
            violations++; // a task processed while the slot code runs
        }
        if (is_yielding) {
            if (!in_yield.load()) {
                violations++; // the worker processed a task outside every yield
            }
            n_worker_tasks++;
        } else {
            n_main_tasks++;
        }
        if (task.type == SERVER_TASK_TYPE_METRICS) {
            std::lock_guard<std::mutex> lock(m_seen);
            seen.push_back(task.id);
        }
        if (task.id == pending_mark.load() && !is_yielding) {
            n_queued_at_start_missed++;
        }
        return true;
    });

    int iter = 0;
    std::vector<double> end_us;
    end_us.reserve(n_iter * 3);
    std::mt19937 rng(1234);
    q.on_update_slots([&]() {
        if (iter >= n_iter) {
            stop_poster = true;
            q.terminate();
            return;
        }
        ++iter;
        // keep the loop going (the server posts NEXT_RESPONSE each round)
        {
            server_task t(SERVER_TASK_TYPE_NEXT_RESPONSE);
            t.id = q.get_new_id();
            q.post(std::move(t));
        }
        for (int k = 0; k < 3; ++k) {
            outside = true;
            spin_us((int) (rng() % 40));
            // every 50th yield: a task queued right before the yield starts is processed by the worker during it
            if (iter % 50 == 0 && k == 1) {
                server_task t(SERVER_TASK_TYPE_METRICS);
                t.id = q.get_new_id();
                pending_mark = t.id;
                n_posted++;
                q.post(std::move(t));
            }
            outside = false;
            const int work_us = (int) (rng() % 300);
            int64_t t_end_work = 0;
            in_yield = true;
            q.yield_to_queue([&]() {
                spin_us(work_us);
                if (iter % 50 == 0 && k == 1) {
                    spin_us(500); // room for the worker to take the marked task
                }
                t_end_work = now_ns();
            });
            end_us.push_back((now_ns() - t_end_work) / 1e3);
            in_yield = false;
        }
        outside = true;
        spin_us((int) (rng() % 40));
        outside = false;
    });

    std::thread poster([&]() {
        std::mt19937 r2(99);
        while (!stop_poster.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(50 + r2() % 400));
            server_task t(SERVER_TASK_TYPE_METRICS);
            t.id = q.get_new_id();
            n_posted++;
            q.post(std::move(t));
        }
    });

    std::thread watchdog([&]() {
        for (int i = 0; i < 1200 && !stop_poster.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!stop_poster.load()) {
            fprintf(stderr, "FAIL: the loop did not finish in 120 s (deadlock?)\n");
            fflush(stderr);
            _Exit(1);
        }
    });

    q.start_loop();
    poster.join();
    watchdog.join();

    // the tasks posted after the last round are not processed (the loop ended): count only what was processed
    std::sort(seen.begin(), seen.end());
    const bool dup = std::adjacent_find(seen.begin(), seen.end()) != seen.end();
    std::sort(end_us.begin(), end_us.end());
    const double med = end_us.empty() ? 0.0 : end_us[end_us.size() / 2];
    const char * fast = getenv("LLAMA_SERVER_YIELD_FAST");
    printf("LLAMA_SERVER_YIELD_FAST=%s: %d rounds, %d yields, yield end median %.1f us; tasks posted %d, processed %zu (worker %d, main %d), "
           "violations %d, marked tasks missed by the yield %d, duplicates %d\n", fast ? fast : "(unset)", iter, (int) end_us.size(), med,
           n_posted.load(), seen.size(), n_worker_tasks.load(), n_main_tasks.load(), violations.load(), n_queued_at_start_missed.load(), (int) dup);
    const int n_lost = n_posted.load() - (int) seen.size();
    bool ok = violations.load() == 0 && !dup && n_queued_at_start_missed.load() == 0 && iter == n_iter && n_lost >= 0 && n_lost <= 20 &&
              n_worker_tasks.load() > 0;
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
