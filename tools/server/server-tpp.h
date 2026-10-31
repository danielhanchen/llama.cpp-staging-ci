#pragma once

// k19-tpp: the pre-fit pieces of the MTP window policy's on / off decision (server-context.cpp load_model), kept free of the backends so a
// unit test can drive them with mock devices

#include <algorithm>
#include <cstdint>
#include <vector>

// the GPUs the fit will use: --device's list when given (else every device), GPUs only, and with --split-mode none only the main GPU
// (main_gpu indexes that list, as the model's devices do)
template <typename D, typename IsGpu>
static std::vector<D> tpp_fit_gpus(const std::vector<D> & requested, const std::vector<D> & all, IsGpu is_gpu, bool split_none, int32_t main_gpu) {
    std::vector<D> gpus;
    for (const D & d : requested.empty() ? all : requested) {
        if (is_gpu(d)) {
            gpus.push_back(d);
        }
    }
    if (split_none && !gpus.empty()) {
        const D main_dev = gpus[std::clamp<int32_t>(main_gpu, 0, (int32_t) gpus.size() - 1)];
        gpus = { main_dev };
    }
    return gpus;
}

// experts placed on the host by hand: a named tensor override (--cpu-moe, --n-cpu-moe, -ot; the list is padded with { nullptr, nullptr }
// entries for the fit, which place nothing) or a fixed --moe-cache-mib
template <typename O>
static bool tpp_host_experts(const std::vector<O> & overrides, size_t moe_cache_size, bool moe_cache_auto) {
    return std::any_of(overrides.begin(), overrides.end(), [](const O & o) { return o.pattern != nullptr; }) ||
        (moe_cache_size > 0 && !moe_cache_auto);
}
