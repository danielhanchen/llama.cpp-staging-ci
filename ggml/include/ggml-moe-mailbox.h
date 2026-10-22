#pragma once

// layout of a single-graph MoE mailbox (k11, LLAMA_MOE_SINGLE; see ggml_moe_mailbox_publish / ggml_moe_mailbox_merge): host memory mapped
// into the device. The device writes the host ids and the activations of a batch, then request (release, system scope); a host thread
// computes the experts of those ids into out and writes completed = request (release); the merge on the device waits for it.

#include <stddef.h>
#include <stdint.h>

#define GGML_MOE_MAILBOX_MAX_TOKENS 4

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_MOE_MAILBOX_FN static inline __host__ __device__
#else
#define GGML_MOE_MAILBOX_FN static inline
#endif

struct ggml_moe_mailbox_hdr {
    uint32_t request;   uint32_t pad0[31]; // device -> host: sequence of the last published batch
    uint32_t completed; uint32_t pad1[31]; // host -> device: sequence of the last computed batch
    uint32_t error;     uint32_t pad2[31]; // 1: the device gave up waiting, 2: the host failed
    int32_t  n_tokens;  int32_t  pad3[31]; // tokens of the published batch
};

// device memory of a mailbox (zeroed at allocation)
struct ggml_moe_mailbox_ctl {
    uint32_t seq;     // sequence of the last published batch
    uint32_t pending; // the last publish had host rows
    uint32_t failed;  // a merge gave up
    uint32_t pad[61];
};

GGML_MOE_MAILBOX_FN size_t ggml_moe_mailbox_align(size_t n) {
    return (n + 255) & ~(size_t) 255;
}

GGML_MOE_MAILBOX_FN size_t ggml_moe_mailbox_off_ids(void) {
    return ggml_moe_mailbox_align(sizeof(struct ggml_moe_mailbox_hdr));
}

GGML_MOE_MAILBOX_FN size_t ggml_moe_mailbox_off_x(int n_used) {
    return ggml_moe_mailbox_off_ids() + ggml_moe_mailbox_align(sizeof(int32_t)*(size_t) n_used*GGML_MOE_MAILBOX_MAX_TOKENS);
}

GGML_MOE_MAILBOX_FN size_t ggml_moe_mailbox_off_out(int n_used, int n_embd) {
    return ggml_moe_mailbox_off_x(n_used) + ggml_moe_mailbox_align(sizeof(float)*(size_t) n_embd*GGML_MOE_MAILBOX_MAX_TOKENS);
}

GGML_MOE_MAILBOX_FN size_t ggml_moe_mailbox_size(int n_used, int n_embd) {
    return ggml_moe_mailbox_off_out(n_used, n_embd) + ggml_moe_mailbox_align(sizeof(float)*(size_t) n_embd*n_used*GGML_MOE_MAILBOX_MAX_TOKENS);
}

// allocation through the device backend (CUDA: ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_mailbox_alloc")):
// returns the host pointer of the mailbox (zeroed), its device pointer and the device memory ctl (zeroed); NULL if not possible
typedef void * (*ggml_backend_moe_mailbox_alloc_t)(void * dev /* ggml_backend_dev_t */, size_t size, void ** mbox_dev, void ** ctl_dev);
typedef void   (*ggml_backend_moe_mailbox_free_t)(void * dev, void * mbox_host, void * ctl_dev);
