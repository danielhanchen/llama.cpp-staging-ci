// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "allreduce.cuh"

#if !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

// ---------------------------------------------------------------------------
// AllReduce for tensor-parallel inference across two GPUs (CUDA or
// ROCm/HIP).
//
// Provides an in-place sum reduction over matching tensors on two GPUs
// in the same process.  Used by the tensor-split path alongside NCCL;
// targets setups without NVLink/xGMI, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#ifdef GGML_USE_HIP
            // Equals ~100ns at 2500 MHz (sleeps for n * [1,64] clock cycles)
            __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // GGML_USE_HIP
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

#if !defined(GGML_USE_HIP)
// ---------------------------------------------------------------------------
// Chunked kernel AllReduce, P2P variant -- same three phases as
// ggml_cuda_ar_kernel, but the data and the arrival token are written
// straight into the PEER's device memory (peer_inbox / peer_flag, reachable
// through cudaDeviceEnablePeerAccess), and each GPU reads its own inbox and
// polls its own flag in local device memory.  One PCIe (or NVLink) crossing
// per element instead of two (D2H into pinned host memory, then the peer's
// read of it), and the spin polls local memory instead of host memory.
//
// Release: every thread's peer stores are fenced (__threadfence_system),
// the block syncs, then thread 0 stores the token into the peer's flag.
// Acquire: thread 0 spins on its local flag (volatile), the block syncs and
// fences, then the inbox is read with cache-volatile loads (ld.global.cv) so
// no stale line from an earlier AR on this slot can be returned.
//
// spin_limit_cycles > 0 bounds the spin (integrity probe only): on timeout
// the kernel sets *timeout_flag and carries on with whatever is in the inbox,
// which the probe then reports as a mismatch.  0 = spin until the token
// arrives, like the host-staged kernel.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_p2p_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ peer_inbox,
        const T_wire * __restrict__ my_inbox,
        int                         count,
        int *                       peer_flag,
        const int *                 my_flag,
        int                         token,
        long long                   spin_limit_cycles,
        int *                       timeout_flag) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));
    static_assert(ELEMS_PER_VEC * sizeof(T_wire) == sizeof(int4), "P2P AllReduce expects 16-byte vectors");

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf -> T_wire and store into the peer's inbox.
    for (int i = gtid; i < count_vec; i += gnt) {
        const int off = i * ELEMS_PER_VEC;
        T_wire wire[ELEMS_PER_VEC];
#pragma unroll
        for (int k = 0; k < ELEMS_PER_VEC; ++k) {
            wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
        }
        ggml_cuda_memcpy_1<sizeof(wire)>(&peer_inbox[off], wire);
    }
    if (bid == 0 && tid < count - tail) {
        peer_inbox[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
    }

    __threadfence_system();
    __syncthreads();

    // Phase 2: signal the peer (remote store), wait for the peer's signal (local load).
    if (tid == 0) {
        ggml_cuda_ar_signal_set(peer_flag + bid * ARRIVAL_INTS, token);
        __threadfence_system();

        const int *     slot  = my_flag + bid * ARRIVAL_INTS;
        const long long start = spin_limit_cycles > 0 ? clock64() : 0;
        while (ggml_cuda_ar_signal_get(slot) != token) {
            __nanosleep(100);
            if (spin_limit_cycles > 0 && clock64() - start > spin_limit_cycles) {
                *(volatile int *) timeout_flag = 1;
                break;
            }
        }
    }

    __syncthreads();
    __threadfence_system();

    // Phase 3: read the peer's contribution from the local inbox and sum, both sides rounded through T_wire.
    for (int i = gtid; i < count_vec; i += gnt) {
        const int off = i * ELEMS_PER_VEC;
        const int4 v = __ldcv(reinterpret_cast<const int4 *>(&my_inbox[off]));
        T_wire wire[ELEMS_PER_VEC];
        memcpy(wire, &v, sizeof(v));
#pragma unroll
        for (int k = 0; k < ELEMS_PER_VEC; ++k) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
        }
    }
    if (bid == 0 && tid < count - tail) {
        const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        const T_wire other = *(const volatile T_wire *) &my_inbox[tail + tid];
        recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(other));
    }
#else
    GGML_UNUSED_VARS(sendbuf, recvbuf, peer_inbox, my_inbox, count, peer_flag, my_flag, token, spin_limit_cycles, timeout_flag);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
}

// Integrity probe helpers: deterministic, exactly representable test values
// (multiples of 0.25 below 2^11) so that every fp32 sum is exact.
static __device__ __forceinline__ float ggml_cuda_ar_probe_value(int idx, int rank, int round) {
    return (float) ((idx * 7 + round * 131 + rank * 1009) % 2039) * 0.25f;
}

static __global__ void ggml_cuda_ar_probe_fill(float * dst, int count, int rank, int round) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        dst[i] = ggml_cuda_ar_probe_value(i, rank, round);
    }
}

// Counts elements of buf that differ from probe_value(i, 0, round) + probe_value(i, 1, round)
// (sum_both = true) or from probe_value(i, rank, round) (sum_both = false).
static __global__ void ggml_cuda_ar_probe_check(const float * buf, int count, int rank, int round, bool sum_both, unsigned int * n_bad) {
    unsigned int bad = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float want = sum_both ?
            ggml_cuda_ar_probe_value(i, 0, round) + ggml_cuda_ar_probe_value(i, 1, round) :
            ggml_cuda_ar_probe_value(i, rank, round);
        bad += buf[i] != want;
    }
    if (bad) {
        atomicAdd(n_bad, bad);
    }
}
#endif // !defined(GGML_USE_HIP)

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
// k19-2xt4: the ring depth is a runtime choice, GGML_CUDA_AR_SLOTS (default 8, 2..GGML_CUDA_AR_POOL_MAX). acquire_slot blocks the
// host until AR N - slots has finished on both devices, so with 2 slots the host ran at most ~one layer ahead of the GPUs and waited
// once per AllReduce per device (Kaggle 2x T4 nsys: 98 cudaEventSynchronize per q38s token, GPUs idle 10-12% (q38), 22-26% (DSV4.1
// fake), 41-45% (Qwen3.6-35B-A3B)). A deeper ring lets the host enqueue several layers; the kernels, buffers per AR and the order of
// the additions are the same, so the result is the same bytes. GGML_CUDA_AR_SLOTS=2 = the old ring (kill switch). Costs (slots - 2)
// x 1 MiB of pinned host staging and, on the P2P path, of device inbox per GPU.
static constexpr int GGML_CUDA_AR_POOL_MAX     = 16;
static constexpr int GGML_CUDA_AR_SLOTS_DEFAULT = 8;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;
    int      n_slots;      // k19-2xt4: ring depth (GGML_CUDA_AR_SLOTS)

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_MAX];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;

    // P2P direct-write path (set only when both directions report
    // cudaDeviceCanAccessPeer, peer access could be enabled and the
    // integrity probe passed).  p2p_inbox[i] / p2p_flags[i] live in device
    // i's memory and are written by the peer; dev_tmp[i] is filled by the
    // peer's cudaMemcpyPeerAsync on the copy-engine path.
    bool                     p2p;
    bool                     p2p_enabled_here[GGML_CUDA_MAX_DEVICES]; // holds a reference of ggml_cuda_ar_peer_refs (dropped on free)
    char *                   p2p_inbox[GGML_CUDA_MAX_DEVICES];        // POOL_SIZE * buf_bytes
    int *                    p2p_flags[GGML_CUDA_MAX_DEVICES];        // POOL_SIZE * KERNEL_BLOCKS * ARRIVAL_STRIDE bytes
    cudaEvent_t              p2p_pushed[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_MAX]; // my data is in the peer's dev_tmp
};

static int * ggml_cuda_ar_p2p_flag_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = (size_t) slot * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(reinterpret_cast<char *>(p->p2p_flags[rank]) + offset);
}

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % p->n_slots);
    const bool pool_lapped = p->call_count >= p->n_slots;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// P2P setup + integrity probe
// ---------------------------------------------------------------------------

// Peer access is a property of the CUDA context, shared by every pipeline of the process: the pipelines that found it disabled
// enable it and count references per (device, peer), and the last of them to be freed disables it again. Disabling it on the
// free of one pipeline revoked it under another live pipeline on the same pair (two -sm tensor contexts on the same 2 GPUs, the
// first freed while the second decodes: illegal address). Access enabled by someone else (ggml_cuda_init with peer copies,
// NCCL) is never counted and never disabled here.
static std::mutex ggml_cuda_ar_peer_mutex;
static int        ggml_cuda_ar_peer_refs[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};

// on the current device (dev): true and a reference taken when peer access to peer is on, else false with the error in rc
static bool ggml_cuda_ar_peer_acquire(int dev, int peer, cudaError_t & rc) {
    std::lock_guard<std::mutex> lock(ggml_cuda_ar_peer_mutex);
    if (ggml_cuda_ar_peer_refs[dev][peer] > 0) {
        ++ggml_cuda_ar_peer_refs[dev][peer];
        rc = cudaSuccess;
        return true;
    }
    rc = cudaDeviceEnablePeerAccess(peer, 0);
    if (rc == cudaSuccess) {
        ggml_cuda_ar_peer_refs[dev][peer] = 1;
        return true;
    }
    (void) cudaGetLastError();
    return false;
}

// on the current device (dev): drops a reference taken by ggml_cuda_ar_peer_acquire, the last one disables peer access
static void ggml_cuda_ar_peer_release(int dev, int peer) {
    std::lock_guard<std::mutex> lock(ggml_cuda_ar_peer_mutex);
    GGML_ASSERT(ggml_cuda_ar_peer_refs[dev][peer] > 0);
    if (--ggml_cuda_ar_peer_refs[dev][peer] == 0) {
        (void) cudaDeviceDisablePeerAccess(peer);
        (void) cudaGetLastError();
    }
}

static void ggml_cuda_ar_p2p_release(ggml_cuda_ar_pipeline * p) {
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->p2p_inbox[i]) { CUDA_CHECK(cudaFree(p->p2p_inbox[i])); p->p2p_inbox[i] = nullptr; }
        if (p->p2p_flags[i]) { CUDA_CHECK(cudaFree(p->p2p_flags[i])); p->p2p_flags[i] = nullptr; }
        for (int s = 0; s < p->n_slots; ++s) {
            if (p->p2p_pushed[i][s]) { CUDA_CHECK(cudaEventDestroy(p->p2p_pushed[i][s])); p->p2p_pushed[i][s] = nullptr; }
        }
        if (p->p2p_enabled_here[i]) {
            ggml_cuda_ar_peer_release(p->devices[i], p->devices[1 - i]);
            p->p2p_enabled_here[i] = false;
        }
    }
    p->p2p = false;
}

#if !defined(GGML_USE_HIP)
// Runs the production P2P kernel (bounded spin) for a few rounds on known data and checks
// every element of the result, then checks one full-size cudaMemcpyPeerAsync each way.
// Returns true when every value arrived exactly; otherwise fills why.
static bool ggml_cuda_ar_p2p_probe(ggml_cuda_ar_pipeline * p, char * why, size_t why_n) {
    const int    n_chunk   = (int) (p->buf_bytes / sizeof(float));
    const int    n_copy    = (int) (p->copy_bytes / sizeof(float));
    float *      buf[2]    = {};
    unsigned *   n_bad[2]  = {};
    int *        tmo[2]    = {};
    bool         ok        = true;
    const long long spin_limit = 4000000000LL; // ~2-3 s at T4 / B200 clocks

    for (int i = 0; i < 2 && ok; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        ok = cudaMalloc((void **) &buf[i], p->copy_bytes) == cudaSuccess &&
             cudaMalloc((void **) &n_bad[i], sizeof(unsigned)) == cudaSuccess &&
             cudaMalloc((void **) &tmo[i], sizeof(int)) == cudaSuccess &&
             cudaMemset(tmo[i], 0, sizeof(int)) == cudaSuccess;
    }
    if (!ok) {
        snprintf(why, why_n, "probe allocation failed");
    }

    // Chunked kernel: 4 rounds alternating the two slots, full chunk and a tail-sized count.
    for (int r = 0; r < 4 && ok; ++r) {
        const int slot  = r % p->n_slots;
        const int count = (r & 2) ? n_chunk - 3 : n_chunk;
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            ggml_cuda_ar_probe_fill<<<256, 256, 0, p->streams[i]>>>(buf[i], count, i, r);
        }
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
        }
        for (int i = 0; i < 2; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            ggml_cuda_ar_p2p_kernel<float, float><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, p->streams[i]>>>(
                buf[i], buf[i],
                reinterpret_cast<float *>(p->p2p_inbox[peer] + (size_t) slot * p->buf_bytes),
                reinterpret_cast<const float *>(p->p2p_inbox[i] + (size_t) slot * p->buf_bytes),
                count, ggml_cuda_ar_p2p_flag_ptr(p, slot, peer), ggml_cuda_ar_p2p_flag_ptr(p, slot, i),
                r + 1, spin_limit, tmo[i]);
        }
        for (int i = 0; i < 2 && ok; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            if (cudaStreamSynchronize(p->streams[i]) != cudaSuccess) {
                (void) cudaGetLastError();
                snprintf(why, why_n, "probe kernel failed on device %d", p->devices[i]);
                ok = false;
            }
        }
        for (int i = 0; i < 2 && ok; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaMemsetAsync(n_bad[i], 0, sizeof(unsigned), p->streams[i]));
            ggml_cuda_ar_probe_check<<<256, 256, 0, p->streams[i]>>>(buf[i], count, i, r, true, n_bad[i]);
            unsigned bad = 0;
            int      timed_out = 0;
            CUDA_CHECK(cudaMemcpyAsync(&bad, n_bad[i], sizeof(bad), cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(&timed_out, tmo[i], sizeof(timed_out), cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
            if (bad || timed_out) {
                snprintf(why, why_n, "probe round %d on device %d: %u of %d values wrong%s",
                         r, p->devices[i], bad, count, timed_out ? ", peer signal never arrived" : "");
                ok = false;
            }
        }
    }

    // Copy engine: one full copy_bytes cudaMemcpyPeerAsync each way into the peer's dev_tmp.
    for (int i = 0; i < 2 && ok; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        ggml_cuda_ar_probe_fill<<<1024, 256, 0, p->streams[i]>>>(buf[i], n_copy, i, 7);
        CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
    }
    for (int i = 0; i < 2 && ok; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        if (cudaMemcpyPeerAsync(p->dev_tmp[peer], p->devices[peer], buf[i], p->devices[i], p->copy_bytes, p->streams[i]) != cudaSuccess) {
            (void) cudaGetLastError();
            snprintf(why, why_n, "probe cudaMemcpyPeerAsync %d -> %d failed", p->devices[i], p->devices[peer]);
            ok = false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
    }
    for (int i = 0; i < 2 && ok; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaMemsetAsync(n_bad[i], 0, sizeof(unsigned), p->streams[i]));
        ggml_cuda_ar_probe_check<<<1024, 256, 0, p->streams[i]>>>(reinterpret_cast<const float *>(p->dev_tmp[i]), n_copy, peer, 7, false, n_bad[i]);
        unsigned bad = 0;
        CUDA_CHECK(cudaMemcpyAsync(&bad, n_bad[i], sizeof(bad), cudaMemcpyDeviceToHost, p->streams[i]));
        CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
        if (bad) {
            snprintf(why, why_n, "probe peer copy %d -> %d: %u of %d values wrong", p->devices[peer], p->devices[i], bad, n_copy);
            ok = false;
        }
    }

    // Reset the flags so production tokens (1, 2, ...) start from a clean ring.
    const size_t flag_bytes = (size_t) p->n_slots * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaMemset(p->p2p_flags[i], 0, flag_bytes));
        if (buf[i])   { CUDA_CHECK(cudaFree(buf[i])); }
        if (n_bad[i]) { CUDA_CHECK(cudaFree(n_bad[i])); }
        if (tmo[i])   { CUDA_CHECK(cudaFree(tmo[i])); }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return ok;
}
#endif // !defined(GGML_USE_HIP)

#if !defined(GGML_USE_HIP)
// Wall time (microseconds per AllReduce) of `iters` back-to-back decode-sized F32 AllReduces
// (`count` elements) on both GPUs, through the P2P kernel (p2p = true) or the host-staged
// kernel. Uses its own token range and resets the arrival / flag rings afterwards.
static double ggml_cuda_ar_time_decode(ggml_cuda_ar_pipeline * p, bool p2p, float * const buf[2], int count, int iters) {
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        const int slot  = it % p->n_slots;
        const int token = it + 1;
        for (int i = 0; i < 2; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            if (p2p) {
                ggml_cuda_ar_p2p_kernel<float, float><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, p->streams[i]>>>(
                    buf[i], buf[i],
                    reinterpret_cast<float *>(p->p2p_inbox[peer] + (size_t) slot * p->buf_bytes),
                    reinterpret_cast<const float *>(p->p2p_inbox[i] + (size_t) slot * p->buf_bytes),
                    count, ggml_cuda_ar_p2p_flag_ptr(p, slot, peer), ggml_cuda_ar_p2p_flag_ptr(p, slot, i), token, 0LL, nullptr);
            } else {
                ggml_cuda_ar_kernel<float, float><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, p->streams[i]>>>(
                    buf[i], buf[i],
                    reinterpret_cast<float *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes),
                    reinterpret_cast<const float *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes),
                    count, ggml_cuda_ar_arrival_ptr(p, slot, i), ggml_cuda_ar_arrival_ptr(p, slot, peer), token);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    const size_t arrival_bytes = (size_t) p->n_slots * p->n_devices * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    const size_t flag_bytes    = (size_t) p->n_slots * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    ggml_cuda_set_device(p->devices[0]);
    CUDA_CHECK(cudaMemset(p->arrival.dev, 0, arrival_bytes));
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaMemset(p->p2p_flags[i], 0, flag_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return us;
}

// Auto mode: P2P only when it is measurably faster than host staging for a decode-sized
// AllReduce on this machine (median of 3 trials each, P2P must win by >= 3%). On 2x T4
// over a PCIe host bridge (Kaggle) P2P measured no decode gain (plans/multi_gpu_dense.md).
static bool ggml_cuda_ar_p2p_faster(ggml_cuda_ar_pipeline * p, double * us_p2p, double * us_host) {
    const int count = 8192; // 32 KiB of F32: a decode-sized hidden state
    float * buf[2] = {};
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaMalloc((void **) &buf[i], count * sizeof(float)));
        CUDA_CHECK(cudaMemset(buf[i], 0, count * sizeof(float)));
    }
    double tp[3], th[3];
    ggml_cuda_ar_time_decode(p, true, buf, count, 16);   // warm-up
    ggml_cuda_ar_time_decode(p, false, buf, count, 16);
    for (int t = 0; t < 3; ++t) {
        tp[t] = ggml_cuda_ar_time_decode(p, true, buf, count, 200);
        th[t] = ggml_cuda_ar_time_decode(p, false, buf, count, 200);
    }
    std::sort(tp, tp + 3);
    std::sort(th, th + 3);
    *us_p2p  = tp[1];
    *us_host = th[1];
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaFree(buf[i]));
    }
    return tp[1] < 0.97 * th[1];
}
#endif // !defined(GGML_USE_HIP)

// Decides the P2P path: GGML_CUDA_AR_P2P=0 disables it, =1 uses it whenever the probe passes;
// unset (auto) also requires it to be faster than host staging (ggml_cuda_ar_p2p_faster). It is
// only considered when both
// directions report peer access, peer access can be enabled and the probe passes.
static bool ggml_cuda_ar_p2p_setup(ggml_cuda_ar_pipeline * p, char * why, size_t why_n) {
#if defined(GGML_USE_HIP)
    snprintf(why, why_n, "not implemented for HIP");
    GGML_UNUSED(p);
    return false;
#else
    const char * env = getenv("GGML_CUDA_AR_P2P");
    if (env != nullptr && strcmp(env, "0") == 0) {
        snprintf(why, why_n, "disabled by GGML_CUDA_AR_P2P=0");
        return false;
    }
    for (int i = 0; i < 2; ++i) {
        int can = 0;
        if (cudaDeviceCanAccessPeer(&can, p->devices[i], p->devices[1 - i]) != cudaSuccess) {
            (void) cudaGetLastError();
            can = 0;
        }
        if (!can) {
            snprintf(why, why_n, "cudaDeviceCanAccessPeer(%d -> %d) = 0", p->devices[i], p->devices[1 - i]);
            return false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cudaError_t rc = cudaSuccess;
        if (ggml_cuda_ar_peer_acquire(p->devices[i], p->devices[1 - i], rc)) {
            p->p2p_enabled_here[i] = true;
        } else if (rc == cudaErrorPeerAccessAlreadyEnabled) {
            // enabled outside the pipelines: not ours to disable
        } else {
            snprintf(why, why_n, "cudaDeviceEnablePeerAccess(%d -> %d): %s", p->devices[i], p->devices[1 - i], cudaGetErrorString(rc));
            ggml_cuda_ar_p2p_release(p);
            return false;
        }
    }
    const size_t inbox_bytes = (size_t) p->n_slots * p->buf_bytes;
    const size_t flag_bytes  = (size_t) p->n_slots * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        bool ok = cudaMalloc((void **) &p->p2p_inbox[i], inbox_bytes) == cudaSuccess &&
                  cudaMalloc((void **) &p->p2p_flags[i], flag_bytes) == cudaSuccess &&
                  cudaMemset(p->p2p_flags[i], 0, flag_bytes) == cudaSuccess;
        for (int s = 0; ok && s < p->n_slots; ++s) {
            ok = cudaEventCreateWithFlags(&p->p2p_pushed[i][s], cudaEventDisableTiming) == cudaSuccess;
        }
        if (!ok) {
            (void) cudaGetLastError();
            snprintf(why, why_n, "P2P buffer allocation failed on device %d", p->devices[i]);
            ggml_cuda_ar_p2p_release(p);
            return false;
        }
    }
    if (!ggml_cuda_ar_p2p_probe(p, why, why_n)) {
        ggml_cuda_ar_p2p_release(p);
        return false;
    }
    const bool forced = env != nullptr && strcmp(env, "1") == 0;
    if (forced) {
        snprintf(why, why_n, "probe passed (4 kernel rounds + %zu MiB peer copy each way), forced by GGML_CUDA_AR_P2P=1", p->copy_bytes >> 20);
        return true;
    }
    double us_p2p = 0.0, us_host = 0.0;
    if (!ggml_cuda_ar_p2p_faster(p, &us_p2p, &us_host)) {
        snprintf(why, why_n, "probe passed, but a 32 KiB AllReduce takes %.1f us P2P vs %.1f us host-staged; GGML_CUDA_AR_P2P=1 forces P2P",
                 us_p2p, us_host);
        ggml_cuda_ar_p2p_release(p);
        return false;
    }
    snprintf(why, why_n, "probe passed (4 kernel rounds + %zu MiB peer copy each way), 32 KiB AllReduce %.1f us P2P vs %.1f us host-staged",
             p->copy_bytes >> 20, us_p2p, us_host);
    return true;
#endif // defined(GGML_USE_HIP)
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep (NVIDIA, sm70+) or
    // __builtin_amdgcn_s_sleep (AMD).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    {
        const uint64_t ns = ggml_cuda_ar_env_u64("GGML_CUDA_AR_SLOTS", GGML_CUDA_AR_SLOTS_DEFAULT);
        p->n_slots = (int) std::min<uint64_t>(std::max<uint64_t>(ns, 2), GGML_CUDA_AR_POOL_MAX);
    }
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    // On the P2P path the default is 0 (FP32 on the wire, exact for 2 GPUs);
    // the value is fixed after the P2P decision below unless the env var is set.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    const bool bf16_threshold_from_env = getenv("GGML_CUDA_AR_BF16_THRESHOLD") != nullptr;
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < p->n_slots; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)p->n_slots * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) p->n_slots * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    char p2p_why[256] = {};
    p->p2p = ggml_cuda_ar_p2p_setup(p, p2p_why, sizeof(p2p_why));
    if (p->p2p && !bf16_threshold_from_env) {
        p->bf16_threshold = 0;
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU, %d-slot ring (GGML_CUDA_AR_SLOTS)\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20, p->n_slots);
    GGML_LOG_INFO("%s: AllReduce transport: %s (%s), wire %s\n", __func__,
                  p->p2p ? "P2P direct write" : "host-staged", p2p_why,
                  p->bf16_threshold > 0 ? "BF16 for F32" : "native (F32 exact)");

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaFree(p->dev_tmp[i]));
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < p->n_slots; ++s) {
            if (p->ev_pool[i][s].app) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].app)); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].cpy[c])); }
            }
            if (p->ev_pool[i][s].h2d) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].h2d)); }
            if (p->ev_pool[i][s].ker) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].ker)); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->host_large_read_done[i]));
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->dev_tmp_kernel_done[i]));
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamDestroy(p->streams[i]));
        }
    }
    ggml_cuda_ar_p2p_release(p);
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#if !defined(GGML_USE_HIP)
// P2P copy-engine variant: each GPU pushes its whole contribution straight into the
// peer's dev_tmp with one cudaMemcpyPeerAsync (no host bounce), then each GPU's
// compute stream adds dev_tmp into dst once the peer's push has landed.
//   AR stream i:      wait compute(i) -> wait add_done(peer, previous AR) -> push src(i) -> dev_tmp(peer) -> record pushed(i)
//   compute stream i: wait pushed(peer) (data arrived) + pushed(i) (src(i) read before the in-place add) -> add -> record add_done(i)
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl_p2p(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    ggml_backend_cuda_context * cuda_ctx[2] = {};

    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[peer]));
        }
        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }
        CUDA_CHECK(cudaMemcpyPeerAsync(p->dev_tmp[peer], p->devices[peer], src_buf[i], p->devices[i], nbytes, p->streams[i]));
        CUDA_CHECK(cudaEventRecord(p->p2p_pushed[i][slot], p->streams[i]));
    }

    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->p2p_pushed[peer][slot]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->p2p_pushed[i][slot]));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i], reinterpret_cast<const T_src *>(p->dev_tmp[i]), (int) ne);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->dev_tmp_kernel_done_valid = true;
    return true;
}
#endif // !defined(GGML_USE_HIP)

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

#if !defined(GGML_USE_HIP)
    if (p->p2p) {
        return ggml_cuda_ar_allreduce_copy_impl_p2p<T_src, T_dst>(p, backends, src_buf, dst_buf, compute, ne, nbytes);
    }
#endif // !defined(GGML_USE_HIP)

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Decide between copy-engine and chunked kernel paths based on the working
    // type's actual byte count.  No upper bound: copy_outer slices reductions
    // larger than copy_bytes into copy_bytes-sized pieces.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#if !defined(GGML_USE_HIP)
#define LAUNCH_AR_KERNEL_P2P(T_dst, T_wire) \
                ggml_cuda_ar_p2p_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->p2p_inbox[peer] + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->p2p_inbox[i] + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_p2p_flag_ptr(p, slot, peer), \
                    ggml_cuda_ar_p2p_flag_ptr(p, slot, i), \
                    token, 0LL, nullptr)
#else
#define LAUNCH_AR_KERNEL_P2P(T_dst, T_wire) GGML_ABORT("P2P AllReduce is not available on HIP")
#endif // !defined(GGML_USE_HIP)
#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                if (p->p2p) { \
                    LAUNCH_AR_KERNEL_P2P(T_dst, T_wire); \
                } else { \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token); \
                }

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
#undef LAUNCH_AR_KERNEL_P2P
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}

#else // defined(GGML_USE_MUSA)

// MUSA lacks the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and a device-side
// sleep intrinsic that this implementation relies on, so the internal
// AllReduce is unavailable there. The dispatcher in ggml-cuda.cu treats
// a nullptr pipeline as "init failed" and silently falls back to the meta
// backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_MUSA)
