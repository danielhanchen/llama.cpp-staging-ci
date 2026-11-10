// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    // k18: the host experts of one single-graph MoE mailbox batch (see ggml-cpu.c): out[t*n_used + j] = down_e(swiglu(gate_e(x_t), up_e(x_t)))
    // for e = ids[t*n_used + j] >= 0, rows of -1 left as they are (GGML_MOE_HOST_LEAN=0: zeroed); bit-identical to the graph mul_mat_id
    // up / gate, swiglu_split, mul_mat_id down.
    // false: not handled (no OpenMP build, types, scratch too small); the caller computes the graph.
    GGML_BACKEND_API size_t ggml_cpu_moe_host_ffn_wsize(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
            int n_used, int n_tokens, int n_threads);
    GGML_BACKEND_API bool   ggml_cpu_moe_host_ffn(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
            const float * x, const int32_t * ids, float * out, int n_used, int n_tokens, int n_threads, void * wbuf, size_t wsize);
    // k19-l4b: the same batch on a team of threads of its own instead of an OpenMP region (bit-identical; GGML_MOE_HOST_TEAM=0: the
    // OpenMP region), whether that team is on, and a stop of the team (joins its threads)
    GGML_BACKEND_API bool   ggml_cpu_moe_host_ffn_team(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
            const float * x, const int32_t * ids, float * out, int n_used, int n_tokens, int n_threads, void * wbuf, size_t wsize);
    GGML_BACKEND_API bool   ggml_cpu_moe_host_team_on(void);
    GGML_BACKEND_API void   ggml_cpu_moe_host_team_stop(void);

    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

    // k19-plerow: a row source for the GET_ROWS of one host matrix (a lazily mapped embedding table): fn returns the address of the
    // bytes of row `row` (the tensor's own row, or a copy of it written to tmp, which holds GGML_CPU_ROW_SOURCE_MAX_BYTES); the gather
    // dequantizes those bytes as before, so a source that returns copies of the same bytes leaves the results bit-identical.
    // fn == NULL removes the source of t. At most 4 tensors; rows of more than GGML_CPU_ROW_SOURCE_MAX_BYTES bytes are refused.
    // Returns true when set (or removed). Also reachable as the proc address "ggml_backend_cpu_set_row_source".
#define GGML_CPU_ROW_SOURCE_MAX_BYTES 4096
    typedef const void * (*ggml_cpu_row_source_t)(void * ud, const struct ggml_tensor * t, int64_t row, void * tmp);
    GGML_BACKEND_API bool ggml_backend_cpu_set_row_source(const struct ggml_tensor * t, ggml_cpu_row_source_t fn, void * ud);

#ifdef __cplusplus
}
#endif
