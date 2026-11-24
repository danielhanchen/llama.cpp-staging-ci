// bit identity of the MoE expert reduction (MUL of the expert outputs by their weights, one VIEW per expert, then the ADD chain, as
// build_moe_ffn builds it) on the Metal devices against the CPU backend. Metal runs the chain as one fused kernel (2 to 16 experts,
// GGML_METAL_FUSION_DISABLE=1: the unfused ops); its output must be the bytes of the unfused chain. SKIP without a Metal device.
// Finite random inputs, shapes of the routed MoEs (2 to 16 experts, 1 to 33 tokens, odd and lm-sized rows). A negative control
// changes one weight on the device: its output must differ. Prints the number of cases and of differing ones; exit 1 on any difference.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct reduce_case {
    int64_t n_embd, k, T;
};

static std::vector<float> run(ggml_backend_t backend, const reduce_case & c, const std::vector<float> & ex, const std::vector<float> & wt) {
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_embd, c.k, c.T);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c.k, c.T);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * weighted = ggml_mul(ctx, experts, weights);
    std::vector<ggml_tensor *> views(c.k);
    for (int64_t e = 0; e < c.k; ++e) {
        views[e] = ggml_view_2d(ctx, weighted, c.n_embd, c.T, weighted->nb[2], e * weighted->nb[1]);
        ggml_build_forward_expand(gf, views[e]);
    }
    ggml_tensor * out = views[0];
    for (int64_t e = 1; e < c.k; ++e) {
        out = ggml_add(ctx, out, views[e]);
        ggml_build_forward_expand(gf, out);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(experts, ex.data(), 0, ex.size() * sizeof(float));
    ggml_backend_tensor_set(weights, wt.data(), 0, wt.size() * sizeof(float));
    std::vector<float> res;
    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main() {
    ggml_backend_load_all();

    std::vector<reduce_case> cases;
    for (int64_t k : { 2, 4, 8, 9, 10, 12, 16 }) {
        for (int64_t T : { 1, 7, 33 }) {
            cases.push_back({ 2048, k, T });
        }
        cases.push_back({ 63, k, 5 });
    }
    cases.push_back({ 4096, 10, 1 });

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        printf("test-moe-reduce-exact: no CPU backend\n");
        return 1;
    }

    int n_cases = 0;
    int n_diff  = 0;
    int n_ctl   = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if ((type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) ||
                strncmp(ggml_backend_dev_name(dev), "MTL", 3) != 0) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) {
            continue;
        }
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        for (const auto & c : cases) {
            std::vector<float> ex(c.n_embd * c.k * c.T);
            std::vector<float> wt(c.k * c.T);
            for (auto & x : ex) {
                x = nd(rng);
            }
            for (auto & x : wt) {
                x = ud(rng);
            }
            const std::vector<float> a = run(cpu, c, ex, wt);
            const std::vector<float> b = run(be, c, ex, wt);
            const int r = a.size() != b.size() || memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0 ? 1 : 0;
            int n_el = 0;
            if (r && a.size() == b.size()) {
                for (size_t i = 0; i < a.size(); ++i) {
                    n_el += memcmp(&a[i], &b[i], sizeof(float)) != 0;
                }
            }
            printf("  %s n_embd %lld k %lld T %lld: %s", ggml_backend_dev_name(dev), (long long) c.n_embd, (long long) c.k,
                    (long long) c.T, r ? "DIFFER" : "equal");
            if (r) {
                printf(" (%d of %zu elements)", n_el, a.size());
            }
            printf("\n");
            n_cases++;
            n_diff += r;
            if (c.k == 10 && c.T == 7) {
                // negative control: one weight changed, the device output must change
                std::vector<float> wt2 = wt;
                wt2[3] += 0.25f;
                const std::vector<float> b2 = run(be, c, ex, wt2);
                const bool ctl_ok = b2.size() == b.size() && memcmp(b.data(), b2.data(), b.size() * sizeof(float)) != 0;
                printf("  %s negative control k 10: %s\n", ggml_backend_dev_name(dev), ctl_ok ? "differs (ok)" : "SAME (broken)");
                n_ctl += ctl_ok ? 0 : 1;
            }
        }
        ggml_backend_free(be);
    }
    ggml_backend_free(cpu);

    if (n_cases == 0) {
        printf("test-moe-reduce-exact: SKIP (no Metal device)\n");
        return 0;
    }
    printf("test-moe-reduce-exact: %d cases, %d differ, %d broken controls\n", n_cases, n_diff, n_ctl);
    printf("%s\n", n_diff == 0 && n_ctl == 0 ? "OK" : "FAIL");
    return n_diff == 0 && n_ctl == 0 ? 0 : 1;
}
