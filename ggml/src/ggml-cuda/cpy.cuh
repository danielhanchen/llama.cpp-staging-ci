// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// device-to-device copy of height rows of width bytes (pitches in bytes) on stream, like cudaMemcpy2DAsync. A copy kernel
// rather than a memcpy: inside a CUDA graph a memcpy node costs more than a small kernel and breaks the programmatic
// dependent launch chain of the kernels around it. GGML_CUDA_DISABLE_COPY_KERNEL=1 uses cudaMemcpy(2D)Async again.
void ggml_cuda_memcpy_d2d(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaStream_t stream);

// up to GGML_CUDA_CPY_MULTI_MAX F32 -> F32 copies with the same shapes and strides that ggml_cuda_cpy runs with its
// generic element kernel (the recurrent conv-state snapshots of a delta-net layer) in one launch, each element copied as before.
// ggml_cuda_cpy_multi_ok: whether ggml_cuda_cpy takes that kernel for src0 -> src1. GGML_CUDA_CPY_MULTI=0: one launch each
#define GGML_CUDA_CPY_MULTI_MAX 4
bool ggml_cuda_cpy_multi_enabled();
bool ggml_cuda_cpy_multi_ok(const ggml_tensor * src0, const ggml_tensor * src1);
void ggml_cuda_cpy_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0, ggml_tensor * const * src1, int n);
