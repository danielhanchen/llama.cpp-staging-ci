// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "ggml-moe-prefetch.h"
#include "common.h"
#include "tiled/tiled.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

// waits between the threads of a graph (barrier, MoE host team) spin for a bounded number of rounds, then give the core away with
// sched_yield every round: with more threads than free cores a pure spin keeps the core from the very thread it waits for until the
// time slice ends (tests at 4 threads on 2-3 vCPUs ran 50-100x slower). GGML_SPIN_YIELD=N: rounds before yielding (default 64),
// 0: spin only (the previous behaviour)
static int ggml_spin_yield_rounds(void) {
    static atomic_int rounds = -1;
    int v = atomic_load_explicit(&rounds, memory_order_relaxed);
    if (v < 0) {
        const char * e = getenv("GGML_SPIN_YIELD");
        v = e != NULL ? atoi(e) : 64;
        v = v < 0 ? 0 : v;
        atomic_store_explicit(&rounds, v, memory_order_relaxed);
    }
    return v;
}

// the CPUs this process may run on (affinity mask where the OS has one), counted once
static int ggml_spin_n_cpus(void) {
    static atomic_int n_cpus = 0;
    int v = atomic_load_explicit(&n_cpus, memory_order_relaxed);
    if (v <= 0) {
#if defined(__linux__) && defined(CPU_COUNT)
        cpu_set_t set;
        CPU_ZERO(&set);
        v = sched_getaffinity(0, sizeof(set), &set) == 0 ? CPU_COUNT(&set) : 0;
#endif
        if (v <= 0) {
#if defined(_WIN32)
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            v = (int) si.dwNumberOfProcessors;
#else
            v = (int) sysconf(_SC_NPROCESSORS_ONLN);
#endif
        }
        v = v > 0 ? v : 1;
        atomic_store_explicit(&n_cpus, v, memory_order_relaxed);
    }
    return v;
}

// the MoE host team's waits sit on the decode's critical path (the device waits for the host batch): they spin only, as before,
// unless the team has more threads than the process has CPUs (then a pure spin starves the waited-for thread and they yield as the
// barrier does). GGML_SPIN_YIELD_TEAM=0: always spin only, =1: always yield past GGML_SPIN_YIELD rounds
static int ggml_spin_yield_rounds_team(int n_threads) {
    static atomic_int mode = -2;
    int m = atomic_load_explicit(&mode, memory_order_relaxed);
    if (m == -2) {
        const char * e = getenv("GGML_SPIN_YIELD_TEAM");
        m = e != NULL ? atoi(e) : -1;
        atomic_store_explicit(&mode, m, memory_order_relaxed);
    }
    if (m == 0 || (m < 0 && n_threads <= ggml_spin_n_cpus())) {
        return 0;
    }
    return ggml_spin_yield_rounds();
}

static inline void ggml_thread_spin_wait(int * spins, int limit) {
    if (limit > 0 && *spins >= limit) {
        sched_yield();
        return;
    }
    ++*spins;
    ggml_thread_cpu_relax();
}

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    const int spin_limit = ggml_spin_yield_rounds();
    int spins = 0;
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_spin_wait(&spins, spin_limit);
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    // If tiled is supported, it will execute the full op here and we return
    if (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (!params->use_ref && src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        const bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        const int64_t bs = ggml_blck_size(vec_dot_type);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    const void * src1_row  = (const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    void       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    const int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    if (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else if (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else {
                        from_float((const float *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (!params->use_ref && src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    const bool use_ref) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    GGML_UNUSED(use_ref); // no multi-column kernels without AVX2

#if defined(__AVX2__)
    // an expert routed to 2 or more rows of the batch: each weight row is decoded once for up to 8 of them (results
    // bit-identical to the loop below; GGML_CPU_MMID_MULTI=0: the loop). The reference backend (use_ref) takes the loop.
    const ggml_vec_dot_multi_t vdm = !use_ref && ir1_end - ir1_start >= 2 ? ggml_cpu_vec_dot_multi(type) : NULL;
    if (vdm != NULL) {
        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += 8) {
            const int nc = (int) MIN(8, ir1_end - iir1);
            const void * cols[8];
            float      * dsts[8];
            for (int c = 0; c < nc; ++c) {
                const struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1 + c);
                const int     id  = row_mapping.i1;
                const int64_t i11 = id % ne11;
                const int64_t i12 = row_mapping.i2;
                cols[c] = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));
                dsts[c] = (float *) ((char *) dst->data + ((int64_t) id*nb1 + i12*nb2));
            }
            for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                float r[8];
                vdm((int) ne00, r, src0_cur + ir0*nb01, cols, nc);
                for (int c = 0; c < nc; ++c) {
                    dsts[c][ir0] = r[c];
                }
            }
        }
        return;
    }
#endif

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

// ---- the host experts of one single-graph MoE mailbox batch (llama_moe_cache, LLAMA_MOE_SINGLE) in one parallel region -------
// The graph it replaces is mul_mat_id(up, x, ids), mul_mat_id(gate, x, ids), swiglu_split(gate, up), mul_mat_id(down, act, ids): four
// nodes, seven barriers and a plan per batch, for 1-4 tokens and a few experts (Flash-Next 4-bit at 12 / 16 GB on an RTX PRO 6000: 2.2-2.9
// experts per batch, 76-95 us per batch, of which 20-35 us does not depend on the number of experts). Here every thread quantizes the
// activations itself (no barrier), computes its share of the up and gate rows (shares of equal weight: a row counts once per pair it serves), waits at the one barrier, then forms the SwiGLU rows
// (and their quantization) of the pairs its down rows need and computes those down rows. Each output is the same vec_dot (or, for an
// expert routed to 2+ rows, the same multi-column kernel, bit-identical to vec_dot per column) of the same quantized rows as in the graph,
// so the result is the graph's bit for bit. Rows of id -1 are left as they are (GGML_MOE_HOST_LEAN=0: 0, as the graph
// writes them): the mailbox merge reads only the host rows.
// false (not handled: no OpenMP, a type or layout the path does not take, the tiled kernels forced): the caller computes the graph.
#define GGML_MOE_HOST_MAX_PAIRS 256
#define GGML_MOE_HOST_COLS      8

static int64_t ggml_moe_host_pad(int64_t n) {
    return (n + 63) & ~(int64_t) 63;
}

// bytes of scratch: up and gate rows of every pair (shared), then per thread the quantized activations for up and gate, one SwiGLU
// row and up to 8 quantized SwiGLU rows
static int64_t ggml_moe_host_per_thread(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        int n_tokens) {
    const int64_t n_embd = up->ne[0];
    const int64_t n_ff   = up->ne[1];
    return ggml_moe_host_pad((int64_t) n_tokens*(int64_t) ggml_row_size(type_traits_cpu[up->type].vec_dot_type, n_embd)) +
           ggml_moe_host_pad((int64_t) n_tokens*(int64_t) ggml_row_size(type_traits_cpu[gate->type].vec_dot_type, n_embd)) +
           ggml_moe_host_pad(n_ff*(int64_t) sizeof(float)) +
           GGML_MOE_HOST_COLS*ggml_moe_host_pad((int64_t) ggml_row_size(type_traits_cpu[down->type].vec_dot_type, n_ff));
}

// item index at weight B, items laid out group by group (rows items per group), an item of group g weighing its columns (pairs):
// the threads take equal weight, not equal items (a group of an expert routed to 4 rows costs 4 single rows)
static int64_t ggml_moe_host_item_at(const int32_t * gstart, int n_grp, int64_t rows, int64_t b) {
    int64_t cum = 0;
    for (int g = 0; g < n_grp; ++g) {
        const int64_t nc = gstart[g + 1] - gstart[g];
        if (b < cum + nc*rows) {
            return (int64_t) g*rows + (b - cum)/nc;
        }
        cum += nc*rows;
    }
    return (int64_t) n_grp*rows;
}

size_t ggml_cpu_moe_host_ffn_wsize(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        int n_used, int n_tokens, int n_threads) {
    if (up == NULL || gate == NULL || down == NULL || n_used <= 0 || n_tokens <= 0 || n_threads <= 0) {
        return 0;
    }
    const int64_t rows = ggml_moe_host_pad((int64_t) n_used*n_tokens*up->ne[1]*(int64_t) sizeof(float));
    const int64_t qact = (int64_t) n_used*n_tokens*ggml_moe_host_pad((int64_t) ggml_row_size(type_traits_cpu[down->type].vec_dot_type, up->ne[1]));
    return (size_t) (2*rows + qact + ggml_moe_host_per_thread(up, gate, down, n_tokens)*n_threads + 64);
}

// the shared state of one batch, read by every thread that computes it (the OpenMP region or the host team below)
struct ggml_moe_host_job {
    const struct ggml_tensor * up;
    const struct ggml_tensor * gate;
    const struct ggml_tensor * down;
    const float * x;
    float       * out;
    int     n_tokens;
    int     n_pair;
    int     n_grp;
    int64_t n_embd;
    int64_t n_ff;
    const int32_t * pe;
    const int32_t * po;
    const int32_t * pt;
    const int32_t * gstart;
    const int32_t * glist;
    float * ub;
    float * gb;
    char  * thr0;
    int64_t per;
    size_t  rsu;
    size_t  rsg;
    size_t  rsd;
    ggml_vec_dot_t       dot_u, dot_g, dot_d;
    ggml_from_float_t    q_u, q_g, q_d;
    ggml_vec_dot_multi_t vdm_u, vdm_g, vdm_d;
    bool qshare; // the gate activations are the up activations' bytes (same type), quantized once
    // the up / gate rows and the down rows are each cut into n_task ranges of equal weight; a thread takes the next free range until
    // none is left. With n_task = the threads every thread gets one range when all of them run (the original ranges); with more, a thread
    // that is descheduled holds back less. Each output is computed once with the same kernel whichever thread takes it.
    int n_task;
    atomic_int next_gu;
    atomic_int next_dn;
    atomic_int done_gu;
    // GGML_MOE_HOST_PF (sliced form): every thread computes the same row slice of every expert (thread i: up / gate rows
    // [i*n_ff/n_thr, (i+1)*n_ff/n_thr), down rows [i*n_embd/n_thr, ...)), so the rows a thread reads are known before the batch and can
    // be read ahead into its caches (ggml_cpu_moe_host_team_prefetch); the SwiGLU rows are formed once per pair into qact. Each output
    // is the same kernel on the same bytes as in the ranges above
    bool   sliced;
    int    n_thr;
    char * qact;
    atomic_int done_act;
    // GGML_MOE_HOST_DEP (default on; 0 = the one barrier): per group, the up and gate rows written so far; a down range waits
    // only for the groups it reads instead of for every up and gate row of the batch
    atomic_int * gdone;
    // 1: a read-ahead job (ggml_moe_host_pf_thread), not a batch
    int    kind;
};

// one up or gate row r (matrix m) of group g for every pair of the group: the computation of the ranges below
static inline void ggml_moe_host_gu_row(const struct ggml_moe_host_job * j, int g, int m, int64_t r, const char * xu, const char * xg) {
    const int32_t * pe = j->pe; const int32_t * pt = j->pt;
    const int32_t * gstart = j->gstart; const int32_t * glist = j->glist;
    const int64_t n_embd = j->n_embd;
    const int64_t n_ff   = j->n_ff;
    const struct ggml_tensor * wt = m == 0 ? j->up : j->gate;
    const char * row = (const char *) wt->data + (int64_t) pe[glist[gstart[g]]]*wt->nb[2] + r*wt->nb[1];
    float      * dst = m == 0 ? j->ub : j->gb;
    const char * xq  = m == 0 ? xu : xg;
    const size_t rs  = m == 0 ? j->rsu : j->rsg;
    const int    nc  = gstart[g + 1] - gstart[g];
    ggml_vec_dot_multi_t const vdm = m == 0 ? j->vdm_u : j->vdm_g;
    if (nc >= 2 && vdm != NULL) {
        for (int c0 = 0; c0 < nc; c0 += GGML_MOE_HOST_COLS) {
            const int ncc = MIN(GGML_MOE_HOST_COLS, nc - c0);
            const void * cols[GGML_MOE_HOST_COLS];
            float res[GGML_MOE_HOST_COLS];
            for (int c = 0; c < ncc; ++c) {
                cols[c] = xq + (size_t) pt[glist[gstart[g] + c0 + c]]*rs;
            }
            vdm((int) n_embd, res, row, cols, ncc);
            for (int c = 0; c < ncc; ++c) {
                dst[(int64_t) glist[gstart[g] + c0 + c]*n_ff + r] = res[c];
            }
        }
    } else {
        ggml_vec_dot_t const dot = m == 0 ? j->dot_u : j->dot_g;
        for (int c = 0; c < nc; ++c) {
            const int p = glist[gstart[g] + c];
            dot((int) n_embd, dst + (int64_t) p*n_ff + r, 0, row, 0, xq + (size_t) pt[p]*rs, 0, 1);
        }
    }
}

// the sliced form (GGML_MOE_HOST_PF): two barriers instead of one, the SwiGLU rows formed once per pair
static void ggml_moe_host_ffn_thread_sliced(struct ggml_moe_host_job * j, int ith, char * xu, char * xg, float * act) {
    const int64_t n_embd = j->n_embd;
    const int64_t n_ff   = j->n_ff;
    const int     n_grp  = j->n_grp;
    const int     n_thr  = j->n_thr;
    const int32_t * po = j->po;
    const int32_t * gstart = j->gstart; const int32_t * glist = j->glist;
    const int64_t qstride = ggml_moe_host_pad((int64_t) j->rsd);

    const int64_t r0 = n_ff*ith/n_thr;
    const int64_t r1 = n_ff*(ith + 1)/n_thr;
    for (int g = 0; g < n_grp; ++g) {
        for (int m = 0; m < 2; ++m) {
            for (int64_t r = r0; r < r1; ++r) {
                ggml_moe_host_gu_row(j, g, m, r, xu, xg);
            }
        }
    }
    atomic_fetch_add_explicit(&j->done_gu, 1, memory_order_acq_rel);
    {
        const int spin_limit = ggml_spin_yield_rounds_team(n_thr);
        int spins = 0;
        while (atomic_load_explicit(&j->done_gu, memory_order_acquire) < n_thr) {
            ggml_thread_spin_wait(&spins, spin_limit);
        }
    }
    for (int p = ith; p < j->n_pair; p += n_thr) {
        ggml_vec_swiglu_f32((int) n_ff, act, j->gb + (int64_t) p*n_ff, j->ub + (int64_t) p*n_ff);
        j->q_d(act, j->qact + (int64_t) p*qstride, n_ff);
    }
    atomic_fetch_add_explicit(&j->done_act, 1, memory_order_acq_rel);
    {
        const int spin_limit = ggml_spin_yield_rounds_team(n_thr);
        int spins = 0;
        while (atomic_load_explicit(&j->done_act, memory_order_acquire) < n_thr) {
            ggml_thread_spin_wait(&spins, spin_limit);
        }
    }
    const int64_t d0 = n_embd*ith/n_thr;
    const int64_t d1 = n_embd*(ith + 1)/n_thr;
    for (int g = 0; g < n_grp; ++g) {
        const int    nc = gstart[g + 1] - gstart[g];
        const char * w0 = (const char *) j->down->data + (int64_t) j->pe[glist[gstart[g]]]*j->down->nb[2];
        for (int c0 = 0; c0 < nc; c0 += GGML_MOE_HOST_COLS) {
            const int ncc = MIN(GGML_MOE_HOST_COLS, nc - c0);
            const void * cols[GGML_MOE_HOST_COLS];
            for (int c = 0; c < ncc; ++c) {
                cols[c] = j->qact + (int64_t) glist[gstart[g] + c0 + c]*qstride;
            }
            if (ncc >= 2 && j->vdm_d != NULL) {
                for (int64_t r = d0; r < d1; ++r) {
                    float res[GGML_MOE_HOST_COLS];
                    j->vdm_d((int) n_ff, res, w0 + r*j->down->nb[1], cols, ncc);
                    for (int c = 0; c < ncc; ++c) {
                        j->out[(int64_t) po[glist[gstart[g] + c0 + c]]*n_embd + r] = res[c];
                    }
                }
            } else {
                for (int c = 0; c < ncc; ++c) {
                    float * o = j->out + (int64_t) po[glist[gstart[g] + c0 + c]]*n_embd;
                    for (int64_t r = d0; r < d1; ++r) {
                        j->dot_d((int) n_ff, o + r, 0, w0 + r*j->down->nb[1], 0, cols[c], 0, 1);
                    }
                }
            }
        }
    }
}

// GGML_MOE_HOST_WAIT_EACH=0: a down range waits for the up / gate rows of all its groups before its first row
static bool ggml_moe_host_wait_each(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GGML_MOE_HOST_WAIT_EACH");
        on = e == NULL || atoi(e) != 0;
    }
    return on != 0;
}

static void ggml_moe_host_ffn_thread_impl(struct ggml_moe_host_job * j, int ith, bool sliced);

static void ggml_moe_host_ffn_thread(struct ggml_moe_host_job * j, int ith) {
    ggml_moe_host_ffn_thread_impl(j, ith, j->sliced);
}

// sliced: every one of the j->n_thr threads runs (the team, or an OpenMP region that got all of its threads)
static void ggml_moe_host_ffn_thread_impl(struct ggml_moe_host_job * j, int ith, bool sliced) {
    const int64_t n_embd = j->n_embd;
    const int64_t n_ff   = j->n_ff;
    const int     n_grp  = j->n_grp;
    const int32_t * pe = j->pe; const int32_t * po = j->po; const int32_t * pt = j->pt;
    const int32_t * gstart = j->gstart; const int32_t * glist = j->glist;
    char  * xu  = j->thr0 + j->per*ith;
    char  * xg0 = xu + ggml_moe_host_pad((int64_t) j->n_tokens*(int64_t) j->rsu);
    float * act = (float *) (xg0 + ggml_moe_host_pad((int64_t) j->n_tokens*(int64_t) j->rsg));
    char  * qb  = (char *) act + ggml_moe_host_pad(n_ff*(int64_t) sizeof(float));
    const int64_t qstride = ggml_moe_host_pad((int64_t) j->rsd);
    char  * xg  = j->qshare ? xu : xg0;

    // 1. the activations of each token quantized for up and gate (whole rows: the graph quantizes block ranges, the same bytes)
    for (int t = 0; t < j->n_tokens; ++t) {
        j->q_u(j->x + (int64_t) t*n_embd, xu + (size_t) t*j->rsu, n_embd);
        if (!j->qshare) {
            j->q_g(j->x + (int64_t) t*n_embd, xg + (size_t) t*j->rsg, n_embd);
        }
    }

    if (sliced) {
        ggml_moe_host_ffn_thread_sliced(j, ith, xu, xg, act);
        return;
    }

    // 2. up and gate rows: items (group, matrix, row), ranges of equal weight
    {
        const int64_t w_all = (int64_t) j->n_pair*2*n_ff;
        for (int k = atomic_fetch_add_explicit(&j->next_gu, 1, memory_order_relaxed); k < j->n_task;
                 k = atomic_fetch_add_explicit(&j->next_gu, 1, memory_order_relaxed)) {
            const int64_t i0 = ggml_moe_host_item_at(gstart, n_grp, 2*n_ff, w_all*k/j->n_task);
            const int64_t i1 = ggml_moe_host_item_at(gstart, n_grp, 2*n_ff, w_all*(k + 1)/j->n_task);
            int dep_g = -1;
            int dep_n = 0;
            for (int64_t it = i0; it < i1; ++it) {
                const int     g = (int) (it/(2*n_ff));
                const int     m = (int) ((it/n_ff) % 2);
                const int64_t r = it % n_ff;
                if (j->gdone != NULL && g != dep_g) {
                    if (dep_g >= 0) {
                        atomic_fetch_add_explicit(&j->gdone[dep_g], dep_n, memory_order_release);
                    }
                    dep_g = g;
                    dep_n = 0;
                }
                dep_n++;
                const struct ggml_tensor * wt = m == 0 ? j->up : j->gate;
                const char * row = (const char *) wt->data + (int64_t) pe[glist[gstart[g]]]*wt->nb[2] + r*wt->nb[1];
                float      * dst = m == 0 ? j->ub : j->gb;
                const char * xq  = m == 0 ? xu : xg;
                const size_t rs  = m == 0 ? j->rsu : j->rsg;
                const int    nc  = gstart[g + 1] - gstart[g];
                ggml_vec_dot_multi_t const vdm = m == 0 ? j->vdm_u : j->vdm_g;
                if (nc >= 2 && vdm != NULL) {
                    for (int c0 = 0; c0 < nc; c0 += GGML_MOE_HOST_COLS) {
                        const int ncc = MIN(GGML_MOE_HOST_COLS, nc - c0);
                        const void * cols[GGML_MOE_HOST_COLS];
                        float res[GGML_MOE_HOST_COLS];
                        for (int c = 0; c < ncc; ++c) {
                            cols[c] = xq + (size_t) pt[glist[gstart[g] + c0 + c]]*rs;
                        }
                        vdm((int) n_embd, res, row, cols, ncc);
                        for (int c = 0; c < ncc; ++c) {
                            dst[(int64_t) glist[gstart[g] + c0 + c]*n_ff + r] = res[c];
                        }
                    }
                } else {
                    ggml_vec_dot_t const dot = m == 0 ? j->dot_u : j->dot_g;
                    for (int c = 0; c < nc; ++c) {
                        const int p = glist[gstart[g] + c];
                        dot((int) n_embd, dst + (int64_t) p*n_ff + r, 0, row, 0, xq + (size_t) pt[p]*rs, 0, 1);
                    }
                }
            }
            if (dep_g >= 0) {
                atomic_fetch_add_explicit(&j->gdone[dep_g], dep_n, memory_order_release);
            }
            atomic_fetch_add_explicit(&j->done_gu, 1, memory_order_acq_rel);
        }
    }

    // the one barrier: every up and gate row is written (with gdone: each down range waits for its own groups below)
    {
        const int spin_limit = ggml_spin_yield_rounds_team(j->n_thr);
        int spins = 0;
        while (j->gdone == NULL && atomic_load_explicit(&j->done_gu, memory_order_acquire) < j->n_task) {
            ggml_thread_spin_wait(&spins, spin_limit);
        }
    }

    // 3. down rows: items (group, row), ranges of equal weight. The SwiGLU rows of a group's pairs and their quantization are formed by
    //    every thread whose range touches the group (the graph's swiglu_split row and its down quantization, same bytes), once for
    //    consecutive ranges of the same group
    {
        const int64_t w_all = (int64_t) j->n_pair*n_embd;
        int have_g  = -1;
        int have_c0 = -1;
        for (int k = atomic_fetch_add_explicit(&j->next_dn, 1, memory_order_relaxed); k < j->n_task;
                 k = atomic_fetch_add_explicit(&j->next_dn, 1, memory_order_relaxed)) {
            const int64_t i0 = ggml_moe_host_item_at(gstart, n_grp, n_embd, w_all*k/j->n_task);
            const int64_t i1 = ggml_moe_host_item_at(gstart, n_grp, n_embd, w_all*(k + 1)/j->n_task);
            // every gate / up task was claimed before this one (a thread takes down ranges only when none is left), so the
            // groups waited for are being computed. Each group is waited for when its rows are reached, so the first group's rows
            // run while the later ones finish (GGML_MOE_HOST_WAIT_EACH=0: all groups of the range are waited for first)
            const bool wait_each = j->gdone != NULL && ggml_moe_host_wait_each();
            if (j->gdone != NULL && !wait_each && i1 > i0) {
                const int spin_limit = ggml_spin_yield_rounds_team(j->n_thr);
                for (int g = (int) (i0/n_embd); g <= (int) ((i1 - 1)/n_embd); ++g) {
                    int spins = 0;
                    while (atomic_load_explicit(&j->gdone[g], memory_order_acquire) < 2*n_ff) {
                        ggml_thread_spin_wait(&spins, spin_limit);
                    }
                }
            }
            int64_t it = i0;
            while (it < i1) {
                const int     g    = (int) (it/n_embd);
                if (wait_each) {
                    const int spin_limit = ggml_spin_yield_rounds_team(j->n_thr);
                    int spins = 0;
                    while (atomic_load_explicit(&j->gdone[g], memory_order_acquire) < 2*n_ff) {
                        ggml_thread_spin_wait(&spins, spin_limit);
                    }
                }
                const int64_t r0   = it - (int64_t) g*n_embd;
                const int64_t r1   = MIN(i1, (int64_t) (g + 1)*n_embd) - (int64_t) g*n_embd;
                const int     nc   = gstart[g + 1] - gstart[g];
                const char  * w0   = (const char *) j->down->data + (int64_t) pe[glist[gstart[g]]]*j->down->nb[2];
                for (int c0 = 0; c0 < nc; c0 += GGML_MOE_HOST_COLS) {
                    const int ncc = MIN(GGML_MOE_HOST_COLS, nc - c0);
                    const void * cols[GGML_MOE_HOST_COLS];
                    const bool form = g != have_g || c0 != have_c0;
                    for (int c = 0; c < ncc; ++c) {
                        const int p = glist[gstart[g] + c0 + c];
                        if (form) {
                            ggml_vec_swiglu_f32((int) n_ff, act, j->gb + (int64_t) p*n_ff, j->ub + (int64_t) p*n_ff);
                            j->q_d(act, qb + c*qstride, n_ff);
                        }
                        cols[c] = qb + c*qstride;
                    }
                    have_g  = g;
                    have_c0 = c0;
                    if (ncc >= 2 && j->vdm_d != NULL) {
                        for (int64_t r = r0; r < r1; ++r) {
                            float res[GGML_MOE_HOST_COLS];
                            j->vdm_d((int) n_ff, res, w0 + r*j->down->nb[1], cols, ncc);
                            for (int c = 0; c < ncc; ++c) {
                                j->out[(int64_t) po[glist[gstart[g] + c0 + c]]*n_embd + r] = res[c];
                            }
                        }
                    } else {
                        for (int c = 0; c < ncc; ++c) {
                            float * o = j->out + (int64_t) po[glist[gstart[g] + c0 + c]]*n_embd;
                            for (int64_t r = r0; r < r1; ++r) {
                                j->dot_d((int) n_ff, o + r, 0, w0 + r*j->down->nb[1], 0, cols[c], 0, 1);
                            }
                        }
                    }
                }
                it = (int64_t) g*n_embd + r1;
            }
        }
    }
}

// ---- the host team of ggml_cpu_moe_host_ffn_team (GGML_MOE_HOST_TEAM, default on; 0 = the OpenMP region) -------------
// L4 host (12 vCPUs, 6 cores) LFM2.5 4 GB: the host experts take most of a decode step (5 threads, 661 us per batch).
// A larger OpenMP team does not get there: libgomp counts every OpenMP thread of the process (the decode team as well) and,
// with more of them than CPUs, stops spinning between regions, so each batch waits for its threads to wake up (an 11-thread team
// lost 9-34%, 2.7-4.9 cores busy). These threads are the batch's own: the caller is thread 0, the others spin on a counter between
// batches and sleep after GGML_MOE_HOST_TEAM_SPIN_US (2000) without one. Up to GGML_MOE_TEAM_MAX threads; one batch at a time.
#define GGML_MOE_TEAM_MAX 64
// the batch word (gen) carries the batch's thread count in its low bits, so a worker the batch does not use reads
// both at once and leaves it alone (it does not count down, spin for it or wake up for it): the caller may run a batch on fewer
// threads than the team has without the idle ones taking CPU time
#define GGML_MOE_TEAM_NBITS 7
#define GGML_MOE_TEAM_NMASK ((1 << GGML_MOE_TEAM_NBITS) - 1)

struct ggml_moe_team {
    ggml_mutex_t call;         // one batch at a time
    ggml_mutex_t mtx;          // with cv: the sleep of an idle worker
    ggml_cond_t  cv;
    ggml_thread_t thr[GGML_MOE_TEAM_MAX];
    int           start_gen[GGML_MOE_TEAM_MAX];
    int           n;           // workers started (threads 1..n)
    struct ggml_moe_host_job * job;
    int                        job_n;   // threads of the batch, the caller included
    atomic_int  GGML_CACHE_ALIGN gen;   // one count per batch (high bits) and the batch's threads, the caller included (low bits)
    atomic_int  GGML_CACHE_ALIGN left;  // workers of the batch that have not finished it
    atomic_int  sleepers;
    atomic_bool stop;
};

static struct ggml_moe_team ggml_moe_team_g;
static atomic_int ggml_moe_team_init = 0; // 0: not yet, 1: ready

// GGML_MOE_HOST_PF (the team form opt-in with =1, the OpenMP form below on by default, =0 off): a read-ahead of the row slices of experts
// a later batch is expected to compute, run by the team's workers while the caller goes on (ggml_cpu_moe_host_team_prefetch). Worker i reads the slices it computes in the sliced form of a batch
// on n_thr threads, so they are in its caches when the batch comes; the next batch (or read-ahead) stops it first. Only bytes are read:
// what a batch computes does not depend on it
// 64-bit atomics of the MoE host path: the MSVC shim above covers LONG only (no atomic_llong, no 64-bit overloads)
#if defined(_MSC_VER) && !defined(__clang__)
typedef volatile LONG64 atomic_llong;
#define ggml_atomic_ll_load(p)           InterlockedCompareExchange64((p), 0, 0)
#define ggml_atomic_ll_store(p, v)       ((void) InterlockedExchange64((p), (LONG64) (v)))
#define ggml_atomic_ll_add_relaxed(p, v) InterlockedExchangeAdd64((p), (LONG64) (v))
#else
#define ggml_atomic_ll_load(p)           atomic_load(p)
#define ggml_atomic_ll_store(p, v)       atomic_store((p), (v))
#define ggml_atomic_ll_add_relaxed(p, v) atomic_fetch_add_explicit((p), (v), memory_order_relaxed)
#endif

struct ggml_moe_host_pf {
    const struct ggml_tensor * up;
    const struct ggml_tensor * gate;
    const struct ggml_tensor * down;
    int32_t exps[GGML_MOE_HOST_MAX_PAIRS];
    int     n_exp;
    int     n_thr;
    atomic_int   abort;
    atomic_llong sink;
};
static struct ggml_moe_host_pf  ggml_moe_pf_g;
static struct ggml_moe_host_job ggml_moe_pf_job;   // kind 1: the workers run ggml_moe_host_pf_thread
static bool                     ggml_moe_pf_pending = false; // under ggml_moe_team_g.call

// -1 (unset): the OpenMP form only (the batches of a host without the host team), 0: off, 1: both forms
static int ggml_moe_host_pf_mode(void) {
    static int mode = -2;
    if (mode == -2) {
        const char * e = getenv("GGML_MOE_HOST_PF");
        mode = e != NULL ? (atoi(e) != 0 ? 1 : 0) : -1;
    }
    return mode;
}

static inline uint64_t ggml_moe_host_pf_read(const char * p, size_t n) {
    uint64_t v = 0;
    for (size_t o = 0; o < n; o += 64) {
        v += *(const volatile unsigned char *) (p + o);
    }
    return v;
}

static void ggml_moe_host_pf_thread(int ith) {
    struct ggml_moe_host_pf * f = &ggml_moe_pf_g;
    const int64_t n_ff   = f->up->ne[1];
    const int64_t n_embd = f->up->ne[0];
    const int64_t r0 = n_ff*ith/f->n_thr,   r1 = n_ff*(ith + 1)/f->n_thr;
    const int64_t d0 = n_embd*ith/f->n_thr, d1 = n_embd*(ith + 1)/f->n_thr;
    uint64_t v = 0;
    for (int k = 0; k < f->n_exp; ++k) {
        if (atomic_load_explicit(&f->abort, memory_order_relaxed)) {
            break;
        }
        const int64_t e = f->exps[k];
        const struct ggml_tensor * w2[2] = { f->up, f->gate };
        for (int m = 0; m < 2; ++m) {
            const char * b = (const char *) w2[m]->data + e*w2[m]->nb[2];
            v += ggml_moe_host_pf_read(b + r0*w2[m]->nb[1], (size_t) ((r1 - r0)*w2[m]->nb[1]));
        }
        const char * b = (const char *) f->down->data + e*f->down->nb[2];
        v += ggml_moe_host_pf_read(b + d0*f->down->nb[1], (size_t) ((d1 - d0)*f->down->nb[1]));
    }
    ggml_atomic_ll_add_relaxed(&f->sink, (long long) v);
}

static int ggml_moe_host_team_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char * e = getenv("GGML_MOE_HOST_TEAM");
        mode = e == NULL || atoi(e) != 0 ? 1 : 0;
    }
    return mode;
}

// the default without GGML_MOE_HOST_TEAM_SPIN_US, set by the caller before the first batch (ggml_cpu_moe_host_team_set_spin_us)
static atomic_llong ggml_moe_host_team_spin_default = 2000;

static int64_t ggml_moe_host_team_spin_us(void) {
    static int64_t us = -1;
    if (us < 0) {
        const char * e = getenv("GGML_MOE_HOST_TEAM_SPIN_US");
        us = e != NULL && atoll(e) >= 0 ? atoll(e) : (int64_t) ggml_atomic_ll_load(&ggml_moe_host_team_spin_default);
    }
    return us;
}

void ggml_cpu_moe_host_team_set_spin_us(int64_t us) {
    if (us >= 0) {
        ggml_atomic_ll_store(&ggml_moe_host_team_spin_default, (long long) us);
    }
}

static int ggml_moe_host_team_tasks(void) {
    static int n = -1;
    if (n < 0) {
        const char * e = getenv("GGML_MOE_HOST_TEAM_TASKS");
        n = e != NULL && atoi(e) > 0 ? MIN(atoi(e), 64) : 4;
    }
    return n;
}

static thread_ret_t ggml_moe_team_worker(void * arg) {
    struct ggml_moe_team * t = &ggml_moe_team_g;
    const int i = (int) (intptr_t) arg;
    int seen = t->start_gen[i];
    int64_t last = ggml_time_us();
    int spins = 0;
    for (;;) {
        const int g = atomic_load_explicit(&t->gen, memory_order_acquire);
        if (atomic_load_explicit(&t->stop, memory_order_acquire)) {
            break;
        }
        if (g != seen) {
            seen = g;
            // a batch on fewer threads: this worker neither computes nor counts down, and its spin time runs from its last batch.
            // The poster writes the next batch only after the batch's own workers have finished this one
            if (i < (g & GGML_MOE_TEAM_NMASK)) {
                if (t->job->kind == 1) {
                    ggml_moe_host_pf_thread(i);
                } else {
                    ggml_moe_host_ffn_thread(t->job, i);
                }
                atomic_fetch_add_explicit(&t->left, -1, memory_order_acq_rel);
                last = ggml_time_us();
                spins = 0;
            }
            continue;
        }
        if (++spins < 256) {
            ggml_thread_cpu_relax();
            continue;
        }
        spins = 0;
        if (ggml_time_us() - last < ggml_moe_host_team_spin_us()) {
            continue;
        }
        // no batch for a while: sleep until the next one (the poster wakes the sleepers after it counts gen up; the shared lock as
        // in the threadpool's wait, which is what ggml_cond_wait takes on Windows)
        ggml_mutex_lock_shared(&t->mtx);
        atomic_fetch_add(&t->sleepers, 1);
        for (;;) {
            const int gw = atomic_load(&t->gen);
            if (atomic_load(&t->stop) || (gw != seen && i < (gw & GGML_MOE_TEAM_NMASK))) {
                break;
            }
            ggml_cond_wait(&t->cv, &t->mtx);
        }
        atomic_fetch_add(&t->sleepers, -1);
        ggml_mutex_unlock_shared(&t->mtx);
        last = ggml_time_us();
    }
    return (thread_ret_t) 0;
}

static void ggml_moe_team_ensure_init(void) {
    if (atomic_load_explicit(&ggml_moe_team_init, memory_order_acquire) == 1) {
        return;
    }
    ggml_critical_section_start();
    if (atomic_load_explicit(&ggml_moe_team_init, memory_order_relaxed) == 0) {
        struct ggml_moe_team * t = &ggml_moe_team_g;
        ggml_mutex_init(&t->call);
        ggml_mutex_init(&t->mtx);
        ggml_cond_init(&t->cv);
        t->n = 0;
        t->job = NULL;
        t->job_n = 0;
        atomic_store(&t->gen, 0);
        atomic_store(&t->left, 0);
        atomic_store(&t->sleepers, 0);
        atomic_store(&t->stop, false);
        atomic_store_explicit(&ggml_moe_team_init, 1, memory_order_release);
    }
    ggml_critical_section_end();
}

// under t->call: a read-ahead still running is stopped and waited for
static void ggml_moe_pf_finish(struct ggml_moe_team * t) {
    if (!ggml_moe_pf_pending) {
        return;
    }
    atomic_store_explicit(&ggml_moe_pf_g.abort, 1, memory_order_relaxed);
    {
        const int spin_limit = ggml_spin_yield_rounds_team(t->n + 1);
        int spins = 0;
        while (atomic_load_explicit(&t->left, memory_order_acquire) != 0) {
            ggml_thread_spin_wait(&spins, spin_limit);
        }
    }
    atomic_store_explicit(&ggml_moe_pf_g.abort, 0, memory_order_relaxed);
    ggml_moe_pf_pending = false;
}

// stops and joins the workers (the next batch starts them again)
void ggml_cpu_moe_host_team_stop(void) {
    if (atomic_load_explicit(&ggml_moe_team_init, memory_order_acquire) != 1) {
        return;
    }
    struct ggml_moe_team * t = &ggml_moe_team_g;
    ggml_mutex_lock(&t->call);
    ggml_moe_pf_finish(t);
    if (t->n > 0) {
        ggml_mutex_lock(&t->mtx);
        atomic_store(&t->stop, true);
        ggml_cond_broadcast(&t->cv);
        ggml_mutex_unlock(&t->mtx);
        for (int i = 1; i <= t->n; ++i) {
            ggml_thread_join(t->thr[i], NULL);
        }
        t->n = 0;
        atomic_store(&t->stop, false);
    }
    ggml_mutex_unlock(&t->call);
}

bool ggml_cpu_moe_host_team_on(void) {
    return ggml_moe_host_team_mode() == 1;
}

#if defined(__GNUC__) && !defined(_WIN32)
// the workers run code of this library: they are joined before it is unloaded (GGML_BACKEND_DL)
__attribute__((destructor)) static void ggml_moe_team_fini(void) {
    ggml_cpu_moe_host_team_stop();
}
#endif

// runs the batch on n_threads threads of the team (the caller is thread 0); false if the team could not start them
static bool ggml_moe_team_run(struct ggml_moe_host_job * j, int n_threads) {
    ggml_moe_team_ensure_init();
    struct ggml_moe_team * t = &ggml_moe_team_g;
    ggml_mutex_lock(&t->call);
    ggml_moe_pf_finish(t);
    while (t->n < n_threads - 1) {
        const int i = t->n + 1;
        t->start_gen[i] = atomic_load(&t->gen);
        if (ggml_thread_create(&t->thr[i], NULL, ggml_moe_team_worker, (void *) (intptr_t) i) != 0) {
            break;
        }
        t->n = i;
    }
    if (t->n < n_threads - 1) {
        ggml_mutex_unlock(&t->call);
        return false;
    }
    t->job   = j;
    t->job_n = n_threads;
    atomic_store_explicit(&t->left, n_threads - 1, memory_order_relaxed);
    const unsigned g0 = (unsigned) atomic_load(&t->gen);
    atomic_store(&t->gen, (int) ((((g0 >> GGML_MOE_TEAM_NBITS) + 1u) << GGML_MOE_TEAM_NBITS) | (unsigned) n_threads));
    if (atomic_load(&t->sleepers) > 0) {
        ggml_mutex_lock(&t->mtx);
        ggml_cond_broadcast(&t->cv);
        ggml_mutex_unlock(&t->mtx);
    }
    ggml_moe_host_ffn_thread(j, 0);
    {
        const int spin_limit = ggml_spin_yield_rounds_team(n_threads);
        int spins = 0;
        while (atomic_load_explicit(&t->left, memory_order_acquire) != 0) {
            ggml_thread_spin_wait(&spins, spin_limit);
        }
    }
    ggml_mutex_unlock(&t->call);
    return true;
}

// GGML_MOE_HOST_PF: starts a read-ahead of the slices of experts exps[0..n) on workers 1..n_threads-1 (the slices of a sliced batch on
// n_threads threads) and returns; false if it did not start (no team, the form off, the workers not running yet)
bool ggml_cpu_moe_host_team_prefetch(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const int32_t * exps, int n, int n_threads) {
    if (ggml_moe_host_pf_mode() != 1 || ggml_moe_host_team_mode() != 1 || up == NULL || gate == NULL || down == NULL || exps == NULL ||
            n <= 0 || n_threads < 2 || n_threads > GGML_MOE_TEAM_MAX) {
        return false;
    }
    ggml_moe_team_ensure_init();
    struct ggml_moe_team * t = &ggml_moe_team_g;
    ggml_mutex_lock(&t->call);
    ggml_moe_pf_finish(t);
    if (t->n < n_threads - 1) {
        ggml_mutex_unlock(&t->call);
        return false;
    }
    struct ggml_moe_host_pf * f = &ggml_moe_pf_g;
    f->up = up; f->gate = gate; f->down = down;
    f->n_exp = 0;
    for (int k = 0; k < n && f->n_exp < GGML_MOE_HOST_MAX_PAIRS; ++k) {
        if (exps[k] >= 0 && exps[k] < up->ne[2]) {
            f->exps[f->n_exp++] = exps[k];
        }
    }
    f->n_thr = n_threads;
    if (f->n_exp == 0) {
        ggml_mutex_unlock(&t->call);
        return false;
    }
    ggml_moe_pf_job.kind = 1;
    t->job   = &ggml_moe_pf_job;
    t->job_n = n_threads;
    atomic_store_explicit(&t->left, n_threads - 1, memory_order_relaxed);
    const unsigned g0 = (unsigned) atomic_load(&t->gen);
    atomic_store(&t->gen, (int) ((((g0 >> GGML_MOE_TEAM_NBITS) + 1u) << GGML_MOE_TEAM_NBITS) | (unsigned) n_threads));
    if (atomic_load(&t->sleepers) > 0) {
        ggml_mutex_lock(&t->mtx);
        ggml_cond_broadcast(&t->cv);
        ggml_mutex_unlock(&t->mtx);
    }
    ggml_moe_pf_pending = true;
    ggml_mutex_unlock(&t->call);
    return true;
}

// GGML_MOE_HOST_PF where the batches run in the OpenMP region: the caller runs the read-ahead itself after it signalled a batch's
// completion, as one region: thread 0 calls stop(stop_data) and ends the region once it returns true (a batch was published) or
// after max_us, the other threads read their slices of experts exps[0..n)
bool ggml_cpu_moe_host_prefetch_sync(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const int32_t * exps, int n, int n_threads, bool (*stop)(void *), void * stop_data, int64_t max_us) {
#if defined(GGML_USE_OPENMP)
    if (ggml_moe_host_pf_mode() == 0 || up == NULL || gate == NULL || down == NULL || exps == NULL || stop == NULL || n <= 0 ||
            n_threads < 2) {
        return false;
    }
    struct ggml_moe_host_pf * f = &ggml_moe_pf_g;
    f->up = up; f->gate = gate; f->down = down;
    f->n_exp = 0;
    for (int k = 0; k < n && f->n_exp < GGML_MOE_HOST_MAX_PAIRS; ++k) {
        if (exps[k] >= 0 && exps[k] < up->ne[2]) {
            f->exps[f->n_exp++] = exps[k];
        }
    }
    f->n_thr = n_threads;
    if (f->n_exp == 0) {
        return false;
    }
    atomic_store_explicit(&f->abort, 0, memory_order_relaxed);
    atomic_int done;
    atomic_store_explicit(&done, 0, memory_order_relaxed);
    const int64_t t0 = ggml_time_us();
    #pragma omp parallel num_threads(n_threads)
    {
        const int ith = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        if (ith == 0) {
            while (atomic_load_explicit(&done, memory_order_acquire) < nth - 1) {
                if (stop(stop_data) || ggml_time_us() - t0 > max_us) {
                    atomic_store_explicit(&f->abort, 1, memory_order_relaxed);
                    break;
                }
                ggml_thread_cpu_relax();
            }
        } else {
            if (nth == n_threads) {
                ggml_moe_host_pf_thread(ith);
            }
            atomic_fetch_add_explicit(&done, 1, memory_order_acq_rel);
        }
    }
    atomic_store_explicit(&f->abort, 0, memory_order_relaxed);
    return true;
#else
    GGML_UNUSED(up); GGML_UNUSED(gate); GGML_UNUSED(down); GGML_UNUSED(exps); GGML_UNUSED(n); GGML_UNUSED(n_threads);
    GGML_UNUSED(stop); GGML_UNUSED(stop_data); GGML_UNUSED(max_us);
    return false;
#endif
}

static bool ggml_moe_host_ffn_impl(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const float * x, const int32_t * ids, float * out, int n_used, int n_tokens, int n_threads, void * wbuf, size_t wsize, bool use_team) {
    const bool team = use_team && n_threads <= GGML_MOE_TEAM_MAX;
#if !defined(GGML_USE_OPENMP)
    if (!team) {
        GGML_UNUSED(up); GGML_UNUSED(gate); GGML_UNUSED(down); GGML_UNUSED(x); GGML_UNUSED(ids); GGML_UNUSED(out);
        GGML_UNUSED(n_used); GGML_UNUSED(n_tokens); GGML_UNUSED(n_threads); GGML_UNUSED(wbuf); GGML_UNUSED(wsize);
        return false;
    }
#endif
    if (up == NULL || gate == NULL || down == NULL || x == NULL || ids == NULL || out == NULL || wbuf == NULL || n_used <= 0 ||
            n_tokens <= 0 || n_threads <= 0 || (int64_t) n_used*n_tokens > GGML_MOE_HOST_MAX_PAIRS) {
        return false;
    }
    {
        const char * e = getenv("GGML_CPU_TILED_MM_FORCE");
        if (e != NULL && atoi(e) == 1) {
            return false; // the graph's mul_mat_id would take the tiled kernels
        }
    }
    const int64_t n_embd = up->ne[0];
    const int64_t n_ff   = up->ne[1];
    const int64_t n_exp  = up->ne[2];
    if (gate->ne[0] != n_embd || gate->ne[1] != n_ff || gate->ne[2] != n_exp || down->ne[0] != n_ff || down->ne[1] != n_embd ||
            down->ne[2] != n_exp) {
        return false;
    }
    const struct ggml_tensor * w[3] = { up, gate, down };
    for (int k = 0; k < 3; ++k) {
        const enum ggml_type vt = type_traits_cpu[w[k]->type].vec_dot_type;
        if (!ggml_is_quantized(w[k]->type) || w[k]->nb[0] != ggml_type_size(w[k]->type) || type_traits_cpu[w[k]->type].vec_dot == NULL ||
                type_traits_cpu[w[k]->type].nrows != 1 || vt == GGML_TYPE_F32 || type_traits_cpu[vt].from_float == NULL ||
                (w[k]->ne[0] % ggml_blck_size(vt)) != 0) {
            return false;
        }
    }
    if (wsize < ggml_cpu_moe_host_ffn_wsize(up, gate, down, n_used, n_tokens, n_threads)) {
        return false;
    }

    // GGML_MOE_HOST_LEAN (bits, default 3; 0 = the earlier form): 1 = rows of id -1 are not written (the mailbox merge reads
    // only the host rows; the earlier form zeroed up to 7 of every 8 rows of the mapped mailbox on the calling thread before the region),
    // 2 = the gate activations reuse the up activations' quantized bytes when both have the same vec_dot type (the same bytes). Every
    // host row keeps its bytes. RTX PRO 6000 Flash-Next 4-bit 24 / 32 GB: 106 -> 88 us per 3-row host batch, decode +0.5..+3.3%,
    // md5 equal in all 8 cells
    static int lean = -1;
    if (lean < 0) {
        const char * el = getenv("GGML_MOE_HOST_LEAN");
        lean = el != NULL ? atoi(el) : 3;
    }

    // the pairs (slot j of token t) with a host expert, tokens outer and slots inner (the order of the graph's row mapping), grouped
    // by expert in the order of first use
    int32_t pe[GGML_MOE_HOST_MAX_PAIRS];     // expert of pair p
    int32_t po[GGML_MOE_HOST_MAX_PAIRS];     // output row (t*n_used + j) of pair p
    int32_t pt[GGML_MOE_HOST_MAX_PAIRS];     // token of pair p
    int32_t pg[GGML_MOE_HOST_MAX_PAIRS];     // group of pair p
    int32_t gstart[GGML_MOE_HOST_MAX_PAIRS + 1];
    int32_t glist[GGML_MOE_HOST_MAX_PAIRS];  // pairs of group g: glist[gstart[g] .. gstart[g + 1]), in pair order
    int32_t gcnt[GGML_MOE_HOST_MAX_PAIRS];
    int n_pair = 0;
    int n_grp  = 0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int j = 0; j < n_used; ++j) {
            const int32_t e = ids[(int64_t) t*n_used + j];
            if (e < 0) {
                if (!(lean & 1)) {
                    memset(out + ((int64_t) t*n_used + j)*n_embd, 0, n_embd*sizeof(float));
                }
                continue;
            }
            if (e >= n_exp) {
                return false;
            }
            int g = -1;
            for (int q = 0; q < n_pair; ++q) {
                if (pe[q] == e) {
                    g = pg[q];
                    break;
                }
            }
            if (g < 0) {
                g = n_grp++;
                gcnt[g] = 0;
            }
            pe[n_pair] = e; po[n_pair] = t*n_used + j; pt[n_pair] = t; pg[n_pair] = g;
            gcnt[g]++;
            n_pair++;
        }
    }
    if (n_pair == 0) {
        return true;
    }
    gstart[0] = 0;
    for (int g = 0; g < n_grp; ++g) {
        gstart[g + 1] = gstart[g] + gcnt[g];
        gcnt[g] = 0;
    }
    for (int p = 0; p < n_pair; ++p) {
        glist[gstart[pg[p]] + gcnt[pg[p]]++] = p;
    }

    struct ggml_moe_host_job job;
    job.up = up; job.gate = gate; job.down = down; job.x = x; job.out = out;
    job.n_tokens = n_tokens; job.n_pair = n_pair; job.n_grp = n_grp; job.n_embd = n_embd; job.n_ff = n_ff;
    job.pe = pe; job.po = po; job.pt = pt; job.gstart = gstart; job.glist = glist;
    const enum ggml_type vu = type_traits_cpu[up->type].vec_dot_type;
    const enum ggml_type vg = type_traits_cpu[gate->type].vec_dot_type;
    const enum ggml_type vd = type_traits_cpu[down->type].vec_dot_type;
    job.rsu = ggml_row_size(vu, n_embd);
    job.rsg = ggml_row_size(vg, n_embd);
    job.rsd = ggml_row_size(vd, n_ff);
    const int64_t rows_b = ggml_moe_host_pad((int64_t) n_used*n_tokens*n_ff*(int64_t) sizeof(float));
    job.per = ggml_moe_host_per_thread(up, gate, down, n_tokens);
    char * base = (char *) GGML_PAD((uintptr_t) wbuf, 64);
    const int64_t qact_b = (int64_t) n_used*n_tokens*ggml_moe_host_pad((int64_t) job.rsd);
    job.ub   = (float *) base;
    job.gb   = (float *) (base + rows_b);
    job.qact = base + 2*rows_b;
    job.thr0 = base + 2*rows_b + qact_b;
    job.kind  = 0;
    // (by default only batches of 6+ tokens: the wide verify windows of a block drafter, whose host experts the read-ahead covers best;
    // at 5 tokens and fewer the sliced form's second barrier cost as much as the read-ahead saved on some hosts)
    job.sliced = n_threads > 1 && (ggml_moe_host_pf_mode() == 1 || (ggml_moe_host_pf_mode() < 0 && !team && n_threads >= 16 && n_tokens >= 6));
    job.n_thr  = n_threads;
    atomic_store_explicit(&job.done_act, 0, memory_order_relaxed);
    job.dot_u = type_traits_cpu[up->type].vec_dot;
    job.dot_g = type_traits_cpu[gate->type].vec_dot;
    job.dot_d = type_traits_cpu[down->type].vec_dot;
    job.q_u = type_traits_cpu[vu].from_float;
    job.q_g = type_traits_cpu[vg].from_float;
    job.q_d = type_traits_cpu[vd].from_float;
    job.qshare = (lean & 2) != 0 && vu == vg && job.q_u == job.q_g;
#if defined(__AVX2__)
    job.vdm_u = ggml_cpu_vec_dot_multi(up->type);
    job.vdm_g = ggml_cpu_vec_dot_multi(gate->type);
    job.vdm_d = ggml_cpu_vec_dot_multi(down->type);
#else
    job.vdm_u = NULL;
    job.vdm_g = NULL;
    job.vdm_d = NULL;
#endif
    atomic_store_explicit(&job.next_gu, 0, memory_order_relaxed);
    atomic_store_explicit(&job.next_dn, 0, memory_order_relaxed);
    atomic_store_explicit(&job.done_gu, 0, memory_order_relaxed);
    // GGML_MOE_HOST_DEP: the per-group counters (the barrier form: NULL). Only the range form uses them (the sliced form
    // keeps its barriers). A100 host (6 cores / 12 threads, 10 host threads), Flash-Next 7-row host batches: the barrier was 8% of
    // each thread's batch time
    static int dep_mode = -1;
    if (dep_mode < 0) {
        const char * e = getenv("GGML_MOE_HOST_DEP");
        dep_mode = e == NULL || atoi(e) != 0;
    }
    atomic_int gdone[GGML_MOE_HOST_MAX_PAIRS];
    job.gdone = NULL;
    if (dep_mode && !job.sliced) {
        for (int g = 0; g < n_grp; ++g) {
            atomic_store_explicit(&gdone[g], 0, memory_order_relaxed);
        }
        job.gdone = gdone;
    }

    if (team) {
        job.n_task = n_threads == 1 ? 1 : n_threads*ggml_moe_host_team_tasks();
        if (n_threads == 1) {
            ggml_moe_host_ffn_thread(&job, 0);
            return true;
        }
        if (ggml_moe_team_run(&job, n_threads)) {
            return true;
        }
#if !defined(GGML_USE_OPENMP)
        return false;
#endif
        atomic_store_explicit(&job.next_gu, 0, memory_order_relaxed);
        atomic_store_explicit(&job.next_dn, 0, memory_order_relaxed);
        atomic_store_explicit(&job.done_gu, 0, memory_order_relaxed);
        atomic_store_explicit(&job.done_act, 0, memory_order_relaxed);
        if (job.gdone != NULL) {
            for (int g = 0; g < n_grp; ++g) {
                atomic_store_explicit(&job.gdone[g], 0, memory_order_relaxed);
            }
        }
    }
#if defined(GGML_USE_OPENMP)
    // ranges: one per thread
    job.n_task = n_threads;
    #pragma omp parallel num_threads(n_threads)
    {
        // (an OpenMP region with fewer threads than asked takes the ranges, whose tasks any thread can take)
        ggml_moe_host_ffn_thread_impl(&job, omp_get_thread_num(), job.sliced && omp_get_num_threads() == job.n_thr);
    }
    return true;
#else
    return false;
#endif
}

// the OpenMP region
bool ggml_cpu_moe_host_ffn(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const float * x, const int32_t * ids, float * out, int n_used, int n_tokens, int n_threads, void * wbuf, size_t wsize) {
    return ggml_moe_host_ffn_impl(up, gate, down, x, ids, out, n_used, n_tokens, n_threads, wbuf, wsize, false);
}

// the host team (GGML_MOE_HOST_TEAM=0: the OpenMP region)
bool ggml_cpu_moe_host_ffn_team(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const float * x, const int32_t * ids, float * out, int n_used, int n_tokens, int n_threads, void * wbuf, size_t wsize) {
    return ggml_moe_host_ffn_impl(up, gate, down, x, ids, out, n_used, n_tokens, n_threads, wbuf, wsize, ggml_moe_host_team_mode() == 1);
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

// GGML_CPU_MMID_QROW=0: MUL_MAT_ID quantizes its activations as blocks split across the threads in every row again
static bool ggml_cpu_mmid_qrow_on(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GGML_CPU_MMID_QROW");
        on = e == NULL || atoi(e) != 0;
    }
    return on != 0;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    // the routed experts of a host expert tensor in a file mapping are read by the disk prefetch workers while the threads
    // below start on them (page cache only, same bytes; a no-op unless the engine is on, see ggml-moe-prefetch.h)
    ggml_moe_prefetch_kernel_hook(dst, ith);

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        // with at least one row per thread, whole rows round robin (a short row splits into fewer blocks than there are
        // threads); the blocks are quantized independently, so the bytes do not depend on who quantizes them
        const int64_t nrq = ne11*ne12*ne13;
        if (nrq >= nth && ggml_cpu_mmid_qrow_on()) {
            for (int64_t ir = ith; ir < nrq; ir += nth) {
                const int64_t i13 = ir/(ne11*ne12);
                const int64_t i12 = (ir/ne11) % ne12;
                const int64_t i11 = ir % ne11;
                from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                           ne10);
            }
        } else {
            for (int64_t i13 = 0; i13 < ne13; ++i13) {
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    for (int64_t i11 = 0; i11 < ne11; ++i11) {
                        size_t bs = ggml_blck_size(vec_dot_type);
                        int64_t ne10_block_start = (ith * ne10/bs) / nth;
                        int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                        from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                                   (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                                   (ne10_block_end - ne10_block_start) * bs);
                    }
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 == -1 || (i02 >= 0 && i02 < n_as));

                if (i02 == -1) {
                    memset((char *) dst->data + id*nb1 + iid1*nb2, 0, ne0*sizeof(float));
                    continue;
                }

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata, params->use_ref
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

// batch invariant nodes (speculative verify batches): the matrix multiplications pick their kernel by the number of src1 columns
// (llamafile sgemm, repacked gemm vs gemv, the tiled kernels from 8 rows per expert), so every column (token) is computed alone, as a
// batch of one token computes it; returns true when done
static bool ggml_compute_forward_batch_invariant(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    if (!(tensor->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
        return false;
    }
    struct ggml_tensor * src1 = tensor->src[1];
    // an attention node over several queries of one KV length (llama-graph, exact verify): one query at a time, as the
    // node of that query alone (the tiled kernel takes 8+ queries)
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        struct ggml_tensor * q    = tensor->src[0];
        struct ggml_tensor * mask = tensor->src[3];
        if (q->ne[1] < 2 || q->ne[3] != 1) {
            return false;
        }
        for (int64_t i = 0; i < q->ne[1]; ++i) {
            struct ggml_tensor q_i   = *q;
            struct ggml_tensor m_i;
            struct ggml_tensor dst_i = *tensor;
            q_i.ne[1] = 1;
            q_i.data  = (char *) q->data + i*q->nb[1];
            if (mask) {
                m_i = *mask;
                m_i.ne[1] = 1;
                m_i.data  = (char *) mask->data + i*mask->nb[1];
            }
            dst_i.ne[2]  = 1;
            dst_i.data   = (char *) tensor->data + i*tensor->nb[2];
            dst_i.src[0] = &q_i;
            dst_i.src[3] = mask ? &m_i : NULL;
            dst_i.flags &= ~GGML_TENSOR_FLAG_BATCH_INVARIANT;
            ggml_compute_forward(params, &dst_i);
            ggml_barrier(params->threadpool);
        }
        return true;
    }
    if (tensor->op == GGML_OP_MUL_MAT) {
        if (src1->ne[1] < 2 || src1->ne[1] > 16 || src1->ne[2]*src1->ne[3] != 1 || tensor->ne[2]*tensor->ne[3] != 1) {
            return false;
        }
        for (int64_t j = 0; j < src1->ne[1]; ++j) {
            struct ggml_tensor src1_j = *src1;
            struct ggml_tensor dst_j  = *tensor;
            src1_j.ne[1] = 1;
            src1_j.data  = (char *) src1->data + j*src1->nb[1];
            dst_j.ne[1]  = 1;
            dst_j.data   = (char *) tensor->data + j*tensor->nb[1];
            dst_j.src[1] = &src1_j;
            dst_j.flags &= ~GGML_TENSOR_FLAG_BATCH_INVARIANT;
            ggml_compute_forward(params, &dst_j);
            ggml_barrier(params->threadpool);
        }
        return true;
    }
    if (tensor->op == GGML_OP_MUL_MAT_ID) {
        struct ggml_tensor * ids = tensor->src[2];
        if (src1->ne[2] < 2 || src1->ne[2] > 16 || src1->ne[3] != 1 || ids->ne[1] != src1->ne[2]) {
            return false;
        }
        // the stock path is already exact while no expert gets enough rows for the tiled kernel (8): one vec_dot per weight row
        // and token, the same as for a single token; the extra buffer types (repacked gemm vs gemv) still need the column loop
        if (src1->ne[2] < 8 && !ggml_tiled_mm_forced() && !ggml_cpu_extra_has_op(tensor)) {
            return false;
        }
        for (int64_t j = 0; j < src1->ne[2]; ++j) {
            struct ggml_tensor src1_j = *src1;
            struct ggml_tensor ids_j  = *ids;
            struct ggml_tensor dst_j  = *tensor;
            src1_j.ne[2] = 1;
            src1_j.data  = (char *) src1->data + j*src1->nb[2];
            ids_j.ne[1]  = 1;
            ids_j.data   = (char *) ids->data + j*ids->nb[1];
            dst_j.ne[2]  = 1;
            dst_j.data   = (char *) tensor->data + j*tensor->nb[2];
            dst_j.src[1] = &src1_j;
            dst_j.src[2] = &ids_j;
            dst_j.flags &= ~GGML_TENSOR_FLAG_BATCH_INVARIANT;
            ggml_compute_forward(params, &dst_j);
            ggml_barrier(params->threadpool);
        }
        return true;
    }
    return false;
}

// GGML_OP_DFLASH2_CONV (see ggml_dflash2_conv), the rows split over the threads; every multiply and add is rounded on
// its own, as the add / mul ops it replaces (no contraction into a fused multiply-add)
#if defined(__clang__)
#define GGML_DFLASH2_NO_CONTRACT
#elif defined(__GNUC__)
#define GGML_DFLASH2_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define GGML_DFLASH2_NO_CONTRACT
#endif
GGML_DFLASH2_NO_CONTRACT
static void ggml_compute_forward_dflash2_conv(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const struct ggml_tensor * x    = dst->src[0];
    const struct ggml_tensor * dyn  = dst->src[1];
    const struct ggml_tensor * base = dst->src[2];
    const int32_t K    = ggml_get_op_params_i32(dst, 0);
    const int32_t gs   = ggml_get_op_params_i32(dst, 1);
    const int32_t nblk = ggml_get_op_params_i32(dst, 2);
    const int32_t side = ggml_get_op_params_i32(dst, 3);
    const int64_t H  = x->ne[0];
    const int64_t T  = x->ne[1];
    const int64_t ng = H / gs;
    const int64_t bs = T / nblk;
    const int64_t nt = K < bs ? K : bs;
    const float * xd = (const float *) x->data;
    const float * dd = (const float *) dyn->data;
    const float * bd = (const float *) base->data;
    float       * od = (float *) dst->data;
    const int64_t dr = dyn->ne[0];
    for (int64_t t = params->ith; t < T; t += params->nth) {
        const int64_t j = t % bs;
        const float * c = dd + t * dr + ng * K * side;
        for (int64_t h = 0; h < H; ++h) {
            const int64_t g = h / gs;
            float r = (c[g] + bd[h]) * xd[t * H + h];
            for (int64_t k = 1; k < nt; ++k) {
                const float w = c[g + ng * k] + bd[h + H * k];
                const float v = j >= k ? xd[(t - k) * H + h] : 0.0f;
                volatile float m = w * v; // rounded on its own on any compiler
                r = r + m;
            }
            od[t * H + h] = r;
        }
    }
}

// GGML_OP_KQ_MASK (see ggml_kq_mask), the rows split over the threads
static bool ggml_kq_mask_swa_masked(int32_t swa_type, int32_t n_swa, int32_t p0, int32_t p1) {
    switch (swa_type) {
        case 1: return p1 - p0 >= n_swa;                                    // LLAMA_SWA_TYPE_STANDARD
        case 2: return p0 < (p1 / n_swa) * n_swa;                           // LLAMA_SWA_TYPE_CHUNKED
        case 3: { const int32_t h = n_swa / 2; const int32_t d = p1 - p0;  // LLAMA_SWA_TYPE_SYMMETRIC
                  return d < -h || d > h; }
        default: return false;
    }
}

static void ggml_compute_forward_kq_mask(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * cells = dst->src[0];
    const struct ggml_tensor * toks  = dst->src[1];
    const int64_t n_kv = dst->ne[0], n_tps = dst->ne[1], n_stream = dst->ne[3];
    const int64_t n_rows = n_tps*n_stream;
    const int64_t dr = (n_rows + params->nth - 1)/params->nth;
    const int64_t r0 = dr*params->ith, r1 = MIN(r0 + dr, n_rows);
    const ggml_fp16_t keep16 = GGML_CPU_FP32_TO_FP16(0.0f), drop16 = GGML_CPU_FP32_TO_FP16(-INFINITY);
    // (ggml_kq_mask_win): 1 = without the window cells that have a ring copy (flags), 2 = the ring's window cells only
    const int32_t win_mode = ggml_get_op_params_i32(dst, 8), n_win = ggml_get_op_params_i32(dst, 9), n_sinks = ggml_get_op_params_i32(dst, 10);
    const struct ggml_tensor * flags = win_mode == 1 ? dst->src[2] : NULL;
    GGML_ASSERT(win_mode == GGML_KQ_MASK_KIND_PLAIN || win_mode == GGML_KQ_MASK_KIND_WIN_BODY || win_mode == GGML_KQ_MASK_KIND_WIN_RING);
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t s = r / n_tps, i = r % n_tps;
        const int32_t * t = (const int32_t *) ((const char *) toks->data + s*toks->nb[2] + i*toks->nb[1]);
        const int32_t p1 = t[0], seq = t[1], causal = t[2], swa_type = t[3], n_swa = t[4], is_2d = t[5], x1 = t[6], y1 = t[7];
        const int32_t * c = (const int32_t *) ((const char *) cells->data + s*cells->nb[2]);
        const uint32_t bit = 1u << (seq & 31);
        const int word = seq < 32 ? 1 : 2;
        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t * cj = c + 5*j;
            const int32_t p0 = cj[0];
            bool keep = p0 != INT32_MIN && (((uint32_t) cj[word]) & bit) != 0;
            if (keep && causal && p0 > p1) {
                keep = false;
            }
            // M-RoPE: a cell at the token's position whose 2D position is after the token's (llama_kv_cell_ext::is_2d_gt)
            if (keep && causal && is_2d && p0 == p1 && (cj[4] > y1 || (cj[4] == y1 && cj[3] > x1))) {
                keep = false;
            }
            if (keep && swa_type != 0 && ggml_kq_mask_swa_masked(swa_type, n_swa, p0, p1)) {
                keep = false;
            }
            if (keep && win_mode != 0) {
                const bool in_win = (p0 <= p1 && (int64_t) p0 > (int64_t) p1 - n_win) || p0 < n_sinks;
                if (win_mode == 1) {
                    keep = !(in_win && ((const int32_t *) ((const char *) flags->data + s*flags->nb[1]))[j] != 0);
                } else {
                    keep = in_win;
                }
            }
            if (dst->type == GGML_TYPE_F16) {
                ((ggml_fp16_t *) ((char *) dst->data + s*dst->nb[3] + i*dst->nb[1]))[j] = keep ? keep16 : drop16;
            } else {
                ((float *) ((char *) dst->data + s*dst->nb[3] + i*dst->nb[1]))[j] = keep ? 0.0f : -INFINITY;
            }
        }
    }
}

// GGML_OP_KQ_MASK kind GGML_KQ_MASK_KIND_SEL (ggml_sel_mask), the rows split over the threads: every cell kq_mask - INFINITY, then the selected
// cells kq_mask + 0, each rounded as the f16 add of the chain it replaces
static void ggml_compute_forward_sel_mask(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * idx = dst->src[0];
    const struct ggml_tensor * kq  = dst->src[1];
    const int64_t n_kv   = dst->ne[0];
    const int64_t n_rows = dst->ne[1]*dst->ne[2]*dst->ne[3];
    const int64_t n_sel  = idx->ne[0];
    const int64_t dr = (n_rows + params->nth - 1)/params->nth;
    const int64_t r0 = dr*params->ith, r1 = MIN(r0 + dr, n_rows);
    for (int64_t r = r0; r < r1; ++r) {
        const ggml_fp16_t * k = (const ggml_fp16_t *) ((const char *) kq->data + r*kq->nb[1]);
        ggml_fp16_t       * d = (ggml_fp16_t *) ((char *) dst->data + r*dst->nb[1]);
        const int32_t     * c = (const int32_t *) ((const char *) idx->data + r*idx->nb[1]);
        for (int64_t j = 0; j < n_kv; ++j) {
            d[j] = GGML_CPU_FP32_TO_FP16(-INFINITY + GGML_CPU_FP16_TO_FP32(k[j]));
        }
        for (int64_t s = 0; s < n_sel; ++s) {
            const int32_t j = c[s];
            if (j >= 0 && j < n_kv) {
                d[j] = GGML_CPU_FP32_TO_FP16(0.0f + GGML_CPU_FP16_TO_FP32(k[j]));
            }
        }
    }
}

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    if (ggml_compute_forward_batch_invariant(params, tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_MOE_CACHE_LOOKUP:
            {
                GGML_ABORT("%s: not supported by the CPU backend", ggml_op_name(tensor->op));
            }
        case GGML_OP_KQ_MASK:
            {
                if (ggml_get_op_params_i32(tensor, GGML_KQ_MASK_KIND_SLOT) == GGML_KQ_MASK_KIND_SEL) {
                    ggml_compute_forward_sel_mask(params, tensor);
                } else {
                    ggml_compute_forward_kq_mask(params, tensor);
                }
            } break;
        case GGML_OP_DFLASH2_CONV:
            {
                ggml_compute_forward_dflash2_conv(params, tensor);
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

// GET_ROWS with at least this many rows runs on all threads (GGML_CPU_GET_ROWS_MT_MIN, 0 = always one thread)
static int64_t ggml_cpu_get_rows_mt_min(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_CPU_GET_ROWS_MT_MIN");
        const int64_t x = e ? atoll(e) : 4096;
        v = x <= 0 ? INT64_MAX : x;
    }
    return v;
}

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_KQ_MASK:
        case GGML_OP_DFLASH2_CONV:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
        case GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // the cost of launching additional threads decreases performance with GPU offloading for small gathers,
                // but a large one (e.g. the per-layer embedding rows of a whole prompt, read from a mapped file) is
                // bound by its page faults and cache misses, which threads overlap
                // every row is dequantized on its own, so splitting the rows does not change the result
                n_tasks = ggml_nelements(node->src[1]) >= ggml_cpu_get_rows_mt_min() ? n_threads : 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#if defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;
                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In
                        const int64_t ne13 = node->src[1]->ne[3]; // Batch

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                        // the second converted Q of a two-source node (one per thread, the row loop's scratch)
                        if (node->src[6] != NULL) {
                            cur += sizeof(float)*n_tasks*(2*DK + 2*DV + CACHE_LINE_SIZE_F32);
                        }
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const bool    win = ggml_get_op_params_i32(node, 1) == 1; // ggml_gated_delta_net_replay
                        const int64_t per_thread = S_v + (K > 1 || win ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

static int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan) {

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            if (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == sizeof(float)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    return 0;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    // past the spin rounds of GGML_SPIN_YIELD the poll yields the core each round (a yield counts as 8 rounds, about its cost, so
    // the poll lasts about as long as before): idle workers then no longer hold the cores the thread that posts the next graph needs
    const uint64_t spin_limit = (uint64_t) ggml_spin_yield_rounds();
    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; ) {
        // No new work. Keep polling.
        if (spin_limit > 0 && i >= spin_limit) {
            sched_yield();
            i += 8;
        } else {
            ggml_thread_cpu_relax();
            i++;
        }
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

// ggml_cpu_set_region_runner: a runner for the OpenMP regions of the graph computes. When set, ggml_graph_compute hands its region to
// runner(fn, arg, user), which runs fn(arg, max_threads) on a thread of its choice (or on the caller) and returns once fn returned
// (max_threads > 0: the region runs on at most that many threads, the team the runner's thread keeps; 0: as planned). A process whose
// graphs are computed from several threads then has them share one thread's OpenMP team: with libgomp each thread that starts a
// region keeps its own team, whose threads busy-wait for the next region after each one, so two teams of the size of the cores keep
// every core busy and slow each other down. The region's threads, its work and its results are the same whichever thread starts it
typedef void (*ggml_cpu_region_runner_t)(void (*fn)(void *, int), void * arg, void * user);
static ggml_cpu_region_runner_t ggml_cpu_region_runner      = NULL;
static void *                   ggml_cpu_region_runner_user = NULL;

#if defined(__GNUC__) || defined(__clang__)
#define GGML_CPU_RUNNER_LOAD(v)     __atomic_load_n(&(v), __ATOMIC_ACQUIRE)
#define GGML_CPU_RUNNER_STORE(v, x) __atomic_store_n(&(v), (x), __ATOMIC_RELEASE)
#else
#define GGML_CPU_RUNNER_LOAD(v)     (v)
#define GGML_CPU_RUNNER_STORE(v, x) ((v) = (x))
#endif

void ggml_cpu_set_region_runner(ggml_cpu_region_runner_t runner, void * user);

void ggml_cpu_set_region_runner(ggml_cpu_region_runner_t runner, void * user) {
    // the user pointer first: a caller that sees the runner sees its user pointer
    GGML_CPU_RUNNER_STORE(ggml_cpu_region_runner_user, user);
    GGML_CPU_RUNNER_STORE(ggml_cpu_region_runner, runner);
}

#ifdef GGML_USE_OPENMP
struct ggml_graph_compute_region {
    struct ggml_threadpool * threadpool;
    int                      n_threads;
};

static void ggml_graph_compute_omp_region(void * arg, int max_threads) {
    struct ggml_graph_compute_region * r = (struct ggml_graph_compute_region *) arg;
    struct ggml_threadpool * threadpool = r->threadpool;
    // fewer threads than planned: the plan's work buffer and workers cover them, and the graph's threads take the count of the region
    int n_threads = max_threads > 0 && max_threads < r->n_threads ? max_threads : r->n_threads;

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp single
        {
            // update the number of threads from the actual number of threads that we got from OpenMP
            n_threads = omp_get_num_threads();
            atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
        }

        // Apply thread CPU mask and priority
        int ith = omp_get_thread_num();

        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
        }
        ggml_graph_compute_thread(&threadpool->workers[ith]);
    }
}
#endif

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        struct ggml_graph_compute_region region = { threadpool, n_threads };
        const ggml_cpu_region_runner_t runner = GGML_CPU_RUNNER_LOAD(ggml_cpu_region_runner);
        if (runner != NULL) {
            runner(ggml_graph_compute_omp_region, &region, GGML_CPU_RUNNER_LOAD(ggml_cpu_region_runner_user));
        } else {
            ggml_graph_compute_omp_region(&region, 0);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#elif defined(__VXE__) || defined(__VXE2__)
    for (; i + 7 < n; i += 8) {
        const uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        const uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        ggml_cpu_iq_kernels_log();
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
        ggml_cpu_arm_kernels_log();
#endif
        is_first_call = false;
    }

    ggml_critical_section_end();
}
