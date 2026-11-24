// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

// GGML_CUDA_P1FUSE_STATS=1 counts the host-side launches of the fused paths and prints them at exit (engagement
// counts for A/B runs; a CUDA graph replay runs no host code, so these count evaluations that launched, not replays)
enum ggml_cuda_p1_stat_kind {
    GGML_CUDA_P1_GDN_FUSED_NEW,  // a gated_delta_net with an in-place state or fused gates that the shape-only test would have refused
    GGML_CUDA_P1_GDN_FUSED,      // every gated_delta_net with an in-place state or fused gates
    GGML_CUDA_P1_GATED_NORM_SILU, // fused gated norm launches with the silu gate
    GGML_CUDA_P1_GATED_NORM_SIG,  // ... with the sigmoid gate
    GGML_CUDA_P1_MMVF_ONEBAR,    // mul_mat_vec_f launches with the one-barrier epilogue (block > warp)
    GGML_CUDA_P1_MMVF_TWOBAR,    // ... with the per-column barriers (GGML_CUDA_MMVF_ONEBAR=0)
    GGML_CUDA_P1_COUNT,
};
void ggml_cuda_p1_stat(ggml_cuda_p1_stat_kind kind);
