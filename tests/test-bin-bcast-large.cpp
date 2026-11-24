// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// A broadcast ADD on the GPU backend whose destination holds 2^32 elements: an attention-score sized f16 tensor
// [n_kv 65536, n_tokens 2048, n_head 32] plus a mask [65536, 2048, 1] broadcast over the heads, in place (one 8 GiB buffer).
// The broadcast kernel collapses the contiguous leading dims, which leaves dims of size 1 whose stride is the whole tensor
// (2^32 elements). Such a stride is never applied (the index of a size-1 dim is 0), so it must not trip the 32-bit asserts.
// Checks a sample of outputs (every head, rows at both ends) against the CPU sum. Needs about 8.3 GiB of device memory.
// No GPU, or not enough free memory: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr int64_t N_KV  = 65536;
static constexpr int64_t N_TOK = 2048;
static constexpr int64_t N_HEAD = 32;

static float val_a(int64_t i) { return (float) ((i * 2654435761u) % 61) * 0.125f; }
static float val_m(int64_t i) { return (float) ((i * 40503u) % 17) - 8.0f; }

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        printf("test-bin-bcast-large: SKIP (no GPU)\n");
        return 0;
    }
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    const size_t need = (size_t) (N_KV * N_TOK * N_HEAD + N_KV * N_TOK) * sizeof(ggml_fp16_t) + (256u << 20);
    if (free_mem < need) {
        printf("test-bin-bcast-large: SKIP (%zu MiB free, %zu MiB needed)\n", free_mem >> 20, need >> 20);
        return 0;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be == nullptr) {
        printf("test-bin-bcast-large: SKIP (backend init failed)\n");
        return 0;
    }

    ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, N_KV, N_TOK, N_HEAD);
    ggml_tensor * m = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, N_KV, N_TOK, 1);
    ggml_tensor * out = ggml_add_inplace(ctx, a, m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) {
        printf("test-bin-bcast-large: SKIP (allocation of %zu MiB failed)\n", need >> 20);
        ggml_free(ctx);
        ggml_backend_free(be);
        return 0;
    }

    // a: one row of N_KV values per (token, head), uploaded row by row; m: the whole mask
    {
        std::vector<ggml_fp16_t> row(N_KV);
        for (int64_t r = 0; r < N_TOK * N_HEAD; ++r) {
            for (int64_t c = 0; c < N_KV; ++c) {
                row[c] = ggml_fp32_to_fp16(val_a(r * N_KV + c));
            }
            ggml_backend_tensor_set(a, row.data(), (size_t) r * N_KV * sizeof(ggml_fp16_t), N_KV * sizeof(ggml_fp16_t));
        }
        std::vector<ggml_fp16_t> mv((size_t) N_KV * N_TOK);
        for (int64_t i = 0; i < N_KV * N_TOK; ++i) {
            mv[i] = ggml_fp32_to_fp16(val_m(i));
        }
        ggml_backend_tensor_set(m, mv.data(), 0, mv.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("test-bin-bcast-large: FAIL (graph compute)\n");
        return 1;
    }

    int64_t n_checked = 0, n_bad = 0;
    std::vector<ggml_fp16_t> row(N_KV);
    const int64_t toks[] = { 0, 1, N_TOK / 2, N_TOK - 1 };
    for (int64_t h = 0; h < N_HEAD; ++h) {
        for (int64_t t : toks) {
            const int64_t r = h * N_TOK + t;
            ggml_backend_tensor_get(out, row.data(), (size_t) r * N_KV * sizeof(ggml_fp16_t), N_KV * sizeof(ggml_fp16_t));
            for (int64_t c = 0; c < N_KV; ++c) {
                const float ref = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    ggml_fp16_to_fp32(ggml_fp32_to_fp16(val_a(r * N_KV + c))) + val_m(t * N_KV + c)));
                const float got = ggml_fp16_to_fp32(row[c]);
                n_checked++;
                if (got != ref) {
                    if (n_bad < 5) {
                        printf("mismatch head %lld token %lld col %lld: %g vs %g\n", (long long) h, (long long) t,
                               (long long) c, got, ref);
                    }
                    n_bad++;
                }
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);

    printf("test-bin-bcast-large: %lld of %lld sampled outputs differ: %s\n", (long long) n_bad, (long long) n_checked,
           n_bad == 0 ? "OK" : "FAIL");
    return n_bad == 0 ? 0 : 1;
}
