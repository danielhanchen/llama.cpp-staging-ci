// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

// k19-tpp: the pre-fit pieces of the MTP window policy's on / off decision (server-context.cpp load_model), kept free of the backends so a
// unit test can drive them with mock devices

#include <algorithm>
#include <cstdint>
#include <vector>

// the kind of a device for tpp_fit_gpus
enum tpp_gpu_kind { TPP_NOT_GPU = 0, TPP_GPU = 1, TPP_IGPU = 2 };

// the GPUs the fit will use: --device's list when given (discrete and integrated GPUs, in its order), else as llama_model_load picks
// them (src/llama.cpp): every discrete GPU, the integrated ones only without a discrete one, and of those only the first one's backend
// (an integrated GPU seen by two backends, e.g. HIP and Vulkan on Strix Halo); with --split-mode none only the main GPU (main_gpu
// indexes that list, as the model's devices do). k19-igpu: integrated GPUs (ggml-cuda's prop.integrated: Strix Halo gfx1151, Jetson,
// DGX Spark) were left out, so there n_gpu was 0: "off: no GPU" by default, and LLAMA_SPEC_TPP=1 forced the window past the MoE cache rule
template <typename D, typename Kind, typename SameBackend>
static std::vector<D> tpp_fit_gpus(const std::vector<D> & requested, const std::vector<D> & all, Kind kind, SameBackend same_backend,
        bool split_none, int32_t main_gpu) {
    std::vector<D> gpus;
    if (!requested.empty()) {
        for (const D & d : requested) {
            if (kind(d) != TPP_NOT_GPU) {
                gpus.push_back(d);
            }
        }
    } else {
        std::vector<D> igpus;
        for (const D & d : all) {
            const int k = kind(d);
            if (k == TPP_GPU) {
                gpus.push_back(d);
            } else if (k == TPP_IGPU && (igpus.empty() || same_backend(d, igpus.back()))) {
                igpus.push_back(d);
            }
        }
        if (gpus.empty()) {
            gpus = igpus;
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
