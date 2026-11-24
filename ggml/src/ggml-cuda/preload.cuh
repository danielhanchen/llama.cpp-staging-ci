// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

// kernel preload (proc "ggml_backend_cuda_preload_graph", ggml-cuda.cu): while a graph is walked for the kernels its launches would
// take, a launch site that checks ggml_cuda_preload_on loads its function (cudaFuncGetAttributes) and does not run it
extern thread_local bool ggml_cuda_preload_active;

static __inline__ bool ggml_cuda_preload_on() {
    return ggml_cuda_preload_active;
}

void ggml_cuda_preload_func(const void * fn);

// the address a launch site takes for a tensor during a kernel preload: the graph walked is not allocated, and a site that branches
// on a present input (ids, candidates) must see one; nothing is read or written there
static __inline__ const void * ggml_cuda_preload_ptr(const ggml_tensor * t) {
    return t == nullptr ? nullptr : t->data != nullptr || !ggml_cuda_preload_on() ? t->data : reinterpret_cast<const void *>(uintptr_t(256));
}

// ggml_cuda_kernel_launch, or during a kernel preload the load of its function
template<typename Kernel, typename... Args>
static __inline__ void ggml_cuda_kernel_launch_pl(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {
    if (ggml_cuda_preload_on()) {
        ggml_cuda_preload_func(reinterpret_cast<const void *>(kernel));
        return;
    }
    ggml_cuda_kernel_launch(kernel, launch_params, std::forward<Args>(args)...);
}

// the kernels the launch of a flash attention node takes (fattn.cu)
void ggml_cuda_flash_attn_ext_preload(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
