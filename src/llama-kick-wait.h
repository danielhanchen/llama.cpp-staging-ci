#pragma once

// a service thread that sleeps while idle and a producer that kicks it (the single-graph host thread of llama_moe_cache)
//
// The sleeper takes the kick stamp (any value that changes with every kick) before it decides to sleep, then waits until ready() or
// until the stamp moves. Without the stamp in the predicate a kick only woke the thread for it to fall asleep again until the timeout
// (up to 1 ms for the first pickup after an idle period), and a kick that landed before the wait was lost. The kick takes the sleep
// mutex before it notifies, so a sleeper between its check and the wait cannot miss it.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

// returns true when woken by ready() or a kick, false at the timeout. use_stamp false: the stamp is ignored (the previous wait)
template <typename Ready>
static inline bool llama_kick_wait_for(std::mutex & mtx, std::condition_variable & cv, const std::atomic<int64_t> & stamp, int64_t stamp0,
        std::chrono::microseconds timeout, bool use_stamp, Ready ready) {
    std::unique_lock<std::mutex> lock(mtx);
    return cv.wait_for(lock, timeout, [&] {
        return ready() || (use_stamp && stamp.load(std::memory_order_acquire) != stamp0);
    });
}

// after the stamp was changed
static inline void llama_kick_notify(std::mutex & mtx, std::condition_variable & cv) {
    {
        std::lock_guard<std::mutex> lock(mtx);
    }
    cv.notify_one();
}
