#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// dst = sum over the experts of ((experts + experts2) * expert_scale) * weights [+ extra]
void ggml_cuda_op_moe_weighted_reduction_add2(ggml_backend_cuda_context & ctx,
                                              const ggml_tensor *         experts,
                                              const ggml_tensor *         experts2,
                                              const ggml_tensor *         expert_scale,
                                              const ggml_tensor *         weights,
                                              const ggml_tensor *         extra,
                                              ggml_tensor *               dst);
