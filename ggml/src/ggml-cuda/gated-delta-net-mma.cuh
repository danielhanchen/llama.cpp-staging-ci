#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;
    // tokens per sequence of the output rows (0 = n_tokens): the kernel may run a leading part of the op's tokens
    int64_t n_tokens_dst = 0;
};

bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & args, cudaStream_t stream);
// false when ggml_cuda_gdn_mma_launch would refuse args for its shape, device or switches (the AR kernel runs)
bool ggml_cuda_gdn_mma_candidate(int device, const ggml_cuda_gdn_mma_args & args);
