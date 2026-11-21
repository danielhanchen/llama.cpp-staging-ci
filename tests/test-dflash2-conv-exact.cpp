// bit identity of ggml_dflash2_conv (one op) and the add / mul / pad chain of build_dflash2_conv (src/models/dflash.cpp)
// it replaces, on every backend device that supports the op (the CPU always). Random inputs with negative zeros, infinities and NaNs;
// shapes of the Flash-Next DFlash2 drafter (H 2560, group 16, K 2: noise blocks of 7 / 8 rows, 2048-row feature batches) and corners
// (K up to 4, blocks shorter than K, two blocks). A negative control flips one input bit of the fused op: its output must differ.
// Prints the number of cases and of differing ones; exit 1 on any difference.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct conv_case {
    int64_t H, T;
    int     K, gs, n_blocks, side;
};

// the chain of build_dflash2_conv, verbatim (hidden / dynamic are contiguous here)
static ggml_tensor * conv_chain(ggml_context * ctx0, ggml_tensor * hidden, ggml_tensor * dynamic, ggml_tensor * base,
        int64_t kernel_size, int64_t group_size, int64_t n_blocks, int side) {
    const int64_t hidden_size = hidden->ne[0];
    const int64_t n_tokens    = hidden->ne[1];
    const int64_t n_groups    = hidden_size / group_size;
    const int64_t block_size  = n_tokens / n_blocks;

    ggml_tensor * blocks = ggml_reshape_3d(ctx0, hidden, hidden_size, block_size, n_blocks);
    ggml_tensor * coeffs = ggml_reshape_4d(ctx0, dynamic, n_groups, kernel_size, 2, n_tokens);
    ggml_tensor * coeffs_side = ggml_view_3d(ctx0, coeffs, n_groups, kernel_size, n_tokens,
            coeffs->nb[1], coeffs->nb[3], side * coeffs->nb[2]);

    ggml_tensor * coeff_all = ggml_cont(ctx0, coeffs_side);
    coeff_all = ggml_reshape_4d(ctx0, coeff_all, 1, n_groups, kernel_size, n_tokens);
    coeff_all = ggml_repeat_4d(ctx0, coeff_all, group_size, n_groups, kernel_size, n_tokens);

    ggml_tensor * base_side = ggml_reshape_4d(ctx0,
            ggml_view_1d(ctx0, base, hidden_size * kernel_size, side * base->nb[2]),
            group_size, n_groups, kernel_size, 1);

    ggml_tensor * weight_all = ggml_add(ctx0, coeff_all, base_side);

    const int64_t n_taps = std::min(kernel_size, block_size);

    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < n_taps; ++tap) {
        ggml_tensor * values = blocks;
        if (tap > 0) {
            ggml_tensor * previous = ggml_view_3d(ctx0, blocks, hidden_size, block_size - tap, n_blocks,
                    blocks->nb[1], blocks->nb[2], 0);
            values = ggml_pad_ext(ctx0, previous, 0, 0, tap, 0, 0, 0, 0, 0);
        }
        values = ggml_reshape_2d(ctx0, values, hidden_size, n_tokens);

        ggml_tensor * weight = ggml_reshape_2d(ctx0,
                ggml_cont(ctx0, ggml_view_4d(ctx0, weight_all, group_size, n_groups, 1, n_tokens,
                        weight_all->nb[1], weight_all->nb[2], weight_all->nb[3], tap * weight_all->nb[2])),
                hidden_size, n_tokens);

        ggml_tensor * term = ggml_mul(ctx0, weight, values);
        result = result ? ggml_add(ctx0, result, term) : term;
    }
    return result;
}

static void fill(std::vector<float> & v, std::mt19937 & rng, bool specials) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 63);
    for (auto & x : v) {
        x = nd(rng);
        if (specials) {
            switch (pick(rng)) {
                case 0: x = -0.0f; break;
                case 1: x = 0.0f; break;
                case 2: x = INFINITY; break;
                case 3: x = -INFINITY; break;
                case 4: x = NAN; break;
                case 5: x = 1e-40f; break; // denormal
                default: break;
            }
        }
    }
}

// returns: 0 equal, 1 differ, -1 skipped (op not supported); neg: the negative control (the fused op's x differs in one bit)
static int run_case(ggml_backend_t backend, ggml_backend_dev_t dev, const conv_case & c, uint32_t seed, bool specials, bool neg) {
    const int64_t ng = c.H / c.gs;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.H, c.T);
    ggml_tensor * x2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.H, c.T);
    ggml_tensor * dyn  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ng * c.K * 2, c.T);
    ggml_tensor * base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.H, c.K, 2);
    ggml_tensor * ref  = conv_chain(ctx, x, dyn, base, c.K, c.gs, c.n_blocks, c.side);
    ggml_tensor * fus  = ggml_dflash2_conv(ctx, x2, dyn,
            ggml_view_1d(ctx, base, c.H * c.K, c.side * base->nb[2]), c.K, c.gs, c.n_blocks, c.side);
    if (!ggml_backend_dev_supports_op(dev, fus)) {
        ggml_free(ctx);
        return -1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, fus);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(seed);
    std::vector<float> hx(c.H * c.T), hd(ng * c.K * 2 * c.T), hb(c.H * c.K * 2);
    fill(hx, rng, specials); fill(hd, rng, specials); fill(hb, rng, specials);
    std::vector<float> hx2 = hx;
    if (neg) {
        // x[0, 0] read by the output (h 0, t 0) through tap 0 (weight set to 1 below): one mantissa bit flipped for the fused op
        if (!std::isfinite(hx[0]) || hx[0] == 0.0f) {
            hx[0] = hx2[0] = 0.5f;
        }
        uint32_t u; memcpy(&u, &hx2[0], 4); u ^= 1u; memcpy(&hx2[0], &u, 4);
    }
    ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(x2, hx2.data(), 0, ggml_nbytes(x2));
    ggml_backend_tensor_set(dyn, hd.data(), 0, ggml_nbytes(dyn));
    ggml_backend_tensor_set(base, hb.data(), 0, ggml_nbytes(base));
    if (neg) {
        // make the perturbed value matter: the weight of tap 0 at (h 0, t 0) finite and non-zero
        float one = 1.0f;
        ggml_backend_tensor_set(dyn, &one, (size_t) c.side * ng * c.K * 4, 4);
        float zero = 0.0f;
        ggml_backend_tensor_set(base, &zero, (size_t) c.side * c.H * c.K * 4, 4);
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(2);
    }
    std::vector<float> a(c.H * c.T), b(c.H * c.T);
    ggml_backend_tensor_get(ref, a.data(), 0, ggml_nbytes(ref));
    ggml_backend_tensor_get(fus, b.data(), 0, ggml_nbytes(fus));
    // bit-exact, except that two NaNs count as equal (their payload / sign may differ with the compiler's vector code)
    int r = 0;
    for (size_t i = 0; i < a.size() && r == 0; ++i) {
        if (std::isnan(a[i]) && std::isnan(b[i])) {
            continue;
        }
        r = memcmp(&a[i], &b[i], sizeof(float)) != 0 ? 1 : 0;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main() {
    ggml_backend_load_all();
    const std::vector<conv_case> cases = {
        { 2560,    7, 2, 16, 1, 0 }, { 2560,    7, 2, 16, 1, 1 }, { 2560,    8, 2, 16, 1, 0 }, { 2560,    8, 2, 16, 1, 1 },
        { 2560, 2048, 2, 16, 1, 0 }, { 2560, 2048, 2, 16, 1, 1 }, { 2560,   14, 2, 16, 2, 1 }, { 2560,   16, 2, 16, 2, 0 },
        {  256,    7, 4, 16, 1, 0 }, {  256,    8, 4, 32, 2, 1 }, {  256,    3, 4, 16, 1, 0 }, {  256,    6, 4, 16, 3, 1 },
        {  128,    1, 2, 16, 1, 0 }, {  128,    5, 3,  8, 5, 1 }, {  512,  512, 3, 64, 4, 0 }, {   64,    9, 1, 16, 1, 1 },
    };
    int n_cases = 0, n_diff = 0, n_neg = 0, n_neg_same = 0, n_dev = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_CPU && t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        int dc = 0, dd = 0;
        for (size_t k = 0; k < cases.size(); ++k) {
            for (int sp = 0; sp < 2; ++sp) {
                const int r = run_case(backend, dev, cases[k], 1234 + 17 * (uint32_t) k + sp, sp != 0, false);
                if (r < 0) {
                    continue;
                }
                dc++; dd += r;
                if (r) {
                    const auto & c = cases[k];
                    printf("  DIFF %s H %lld T %lld K %d gs %d blocks %d side %d specials %d\n", ggml_backend_dev_name(dev),
                            (long long) c.H, (long long) c.T, c.K, c.gs, c.n_blocks, c.side, sp);
                }
            }
            const int r = run_case(backend, dev, cases[k], 99 + (uint32_t) k, false, true);
            if (r >= 0) {
                n_neg++; n_neg_same += r == 0;
            }
        }
        printf("%s: %d cases, %d differ\n", ggml_backend_dev_name(dev), dc, dd);
        n_cases += dc; n_diff += dd; n_dev += dc > 0;
        ggml_backend_free(backend);
    }
    printf("test-dflash2-conv-exact: %d devices, %d cases, %d differ; negative control %d cases, %d not detected\n",
            n_dev, n_cases, n_diff, n_neg, n_neg_same);
    const bool ok = n_cases > 0 && n_diff == 0 && n_neg > 0 && n_neg_same == 0;
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
