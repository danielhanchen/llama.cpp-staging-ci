#pragma once

// layout of a single-graph MoE mailbox (k11, LLAMA_MOE_SINGLE; see ggml_moe_mailbox_publish / ggml_moe_mailbox_merge): host memory mapped
// into the device. The device writes the host ids and the activations of a batch, then request (release, system scope); a host thread
// computes the experts of those ids into out and writes completed = request (release); the merge on the device waits for it.

#include <stddef.h>
#include <stdint.h>

// k17-models: capacity 8 (was 4), so an 8-row verify window (7 n-gram drafts) takes the mailbox instead of a graph split per layer;
// llama_moe_cache limits the batches it publishes at run time (LLAMA_MOE_SINGLE_MAX_TOKENS, default 4 = the previous behaviour; 8 opt-in)
#define GGML_MOE_MAILBOX_MAX_TOKENS 8

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
    uint32_t started;   uint32_t pad4[31]; // host -> device: sequence of the batch the host computes now (picked up)
    uint32_t busy;      uint32_t pad5[31]; // host -> device: nonzero while the host thread computes a batch of any mailbox
    // k19-overlap GGML_MOE_MAILBOX_TIMING=1 (diagnostic, off by default): running sums the device stores here (plain stores, never
    // read back by it): [0] batches, [1] ns from the end of the publish to the merge's start (the device work beside the host),
    // [2] ns the merge waited for the host, [3] batches whose merge waited over 2 us, [4] ns from the publish to the end of the wait,
    // [5] ns from the host's completion stamp ([7], host CLOCK_REALTIME) to the end of the wait; [6] the publish's globaltimer stamp,
    // stored before the request (the host reads it at the pickup); [7] the host's CLOCK_REALTIME stamp, stored before completed;
    // [8] the smallest (end of the wait - [7]) as int64, for the clock offset (see llama_moe_cache single_timing_report)
    uint64_t tm[12];    uint64_t pad6[4];
};

// device memory of a mailbox (zeroed at allocation)
struct ggml_moe_mailbox_ctl {
    uint32_t seq;     // sequence of the last published batch
    uint32_t pending; // the last publish had host rows
    uint32_t failed;  // a merge gave up
    uint32_t pad0;
    uint64_t t_pub;   // k19-overlap GGML_MOE_MAILBOX_TIMING: globaltimer at the end of the last publish with host rows
    uint64_t sums[6]; // k19-overlap GGML_MOE_MAILBOX_TIMING: the sums of ggml_moe_mailbox_hdr::tm, kept here and stored there
    int64_t  min_d2;  // k19-overlap GGML_MOE_MAILBOX_TIMING: hdr tm[8] (0 = none yet)
    uint32_t pad[44];
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
