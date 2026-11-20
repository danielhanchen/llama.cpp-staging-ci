// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-igpu: tpp_fit_gpus (tools/server/server-tpp.h) with mock device lists: the GPUs the k19-tpp MTP window policy prices are the ones
// llama_model_load picks (src/llama.cpp): discrete GPUs, integrated ones only without a discrete one (of the first one's backend only),
// --device's list as given, --split-mode none the main GPU of that list

#include "server-tpp.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct mock_dev {
    std::string name;
    int         kind; // tpp_gpu_kind
    int         reg;  // backend registry
};

static int n_fail = 0;

static void check(const char * what, const std::vector<const mock_dev *> & got, const std::vector<std::string> & want) {
    std::string g;
    std::string w;
    for (const mock_dev * d : got) {
        g += d->name + " ";
    }
    for (const std::string & s : want) {
        w += s + " ";
    }
    if (g != w) {
        fprintf(stderr, "FAIL %s: got [%s] want [%s]\n", what, g.c_str(), w.c_str());
        n_fail++;
    } else {
        printf("ok   %s: [%s]\n", what, g.c_str());
    }
}

static std::vector<const mock_dev *> fit(const std::vector<const mock_dev *> & req, const std::vector<const mock_dev *> & all,
        bool split_none = false, int32_t main_gpu = 0) {
    return tpp_fit_gpus(req, all,
        [](const mock_dev * d) { return d->kind; },
        [](const mock_dev * a, const mock_dev * b) { return a->reg == b->reg; },
        split_none, main_gpu);
}

int main() {
    const mock_dev cpu   { "CPU",     TPP_NOT_GPU, 0 };
    const mock_dev cuda0 { "CUDA0",   TPP_GPU,     1 };
    const mock_dev cuda1 { "CUDA1",   TPP_GPU,     1 };
    const mock_dev rocm0 { "ROCm0",   TPP_IGPU,    2 }; // Strix Halo 8060S under HIP
    const mock_dev vk0   { "Vulkan0", TPP_IGPU,    3 }; // the same iGPU seen by Vulkan
    const mock_dev igpu1 { "ROCm1",   TPP_IGPU,    2 };

    // Strix Halo: the integrated GPU alone is the fit's GPU (was: none)
    check("igpu only", fit({}, { &cpu, &rocm0 }), { "ROCm0" });
    // the same iGPU seen by two backends: the first one's backend only, as llama_model_load
    check("igpu two backends", fit({}, { &rocm0, &vk0, &cpu }), { "ROCm0" });
    check("igpu same backend", fit({}, { &rocm0, &vk0, &igpu1 }), { "ROCm0", "ROCm1" });
    // a discrete GPU beside an integrated one: the discrete one only (unchanged)
    check("dgpu + igpu", fit({}, { &rocm0, &cuda0, &cpu }), { "CUDA0" });
    check("two dgpus", fit({}, { &cpu, &cuda0, &cuda1 }), { "CUDA0", "CUDA1" });
    check("cpu only", fit({}, { &cpu }), {});
    // --device: its list in its order, integrated GPUs included
    check("--device igpu", fit({ &rocm0 }, { &cuda0, &rocm0, &cpu }), { "ROCm0" });
    check("--device mixed", fit({ &rocm0, &cpu, &cuda0 }, { &cuda0, &rocm0, &cpu }), { "ROCm0", "CUDA0" });
    // --split-mode none: the main GPU of that list
    check("split none main 1", fit({}, { &cuda0, &cuda1 }, true, 1), { "CUDA1" });
    check("split none igpu", fit({}, { &cpu, &rocm0 }, true, 0), { "ROCm0" });
    check("split none clamp", fit({}, { &cpu, &rocm0 }, true, 5), { "ROCm0" });

    if (n_fail > 0) {
        fprintf(stderr, "%d failed\n", n_fail);
        return 1;
    }
    printf("all passed\n");
    return 0;
}
