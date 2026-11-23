// The CPU backend's Q8_K quantization (the activations of every K-quant / IQ dot product) is quantize_row_q8_K_ref byte for byte:
// scale, values and 16-value sums, over random rows with ties of the largest magnitude (both signs), zero blocks, scales from 1e-6 to
// 1e6, values whose products are exact half integers, and a NaN block. Any byte difference fails.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

int main() {
    ggml_cpu_init();
    if (ggml_cpu_has_riscv_v() || ggml_cpu_has_wasm_simd()) {
        // their zero blocks also clear the 16-value sums, which the reference leaves as they were
        printf("test-q8k-exact: SKIP (RVV / WASM SIMD quantizer)\n");
        return 0;
    }
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const int n = 2560;
    const size_t rs = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0, rows = 0;
    for (int r = 0; r < 4000; ++r) {
        std::vector<float> x(n);
        const float sc = std::pow(10.f, (float) (r % 13) - 6.f);
        for (auto & v : x) {
            v = nd(rng)*sc;
        }
        if (r % 7 == 1) { x[17] = 3.0f*sc; x[100] = -3.0f*sc; x[3] = 3.0f*sc; }
        if (r % 11 == 2) { for (int j = 256; j < 512; ++j) { x[j] = (r & 1) ? 0.0f : -0.0f; } }
        if (r % 5 == 3) { for (int j = 0; j < n; ++j) { x[j] = std::round(x[j]*4.f/sc)*0.25f; } }
        if (r % 9 == 4) { x[600] = 127.f; for (int j = 512; j < 768; ++j) { if (j != 600) { x[j] = (float) ((j % 255) - 127) + 0.5f; } } }
#ifdef NDEBUG
        if (r % 97 == 5) { x[1000] = NAN; } // a debug build asserts on a NaN in the reference (nearest_int)
#endif
        std::vector<uint8_t> a(rs, 0xAB), b(rs, 0xAB);
        tr->from_float(x.data(), a.data(), n);
        quantize_row_q8_K_ref(x.data(), b.data(), n);
        ++rows;
        if (memcmp(a.data(), b.data(), rs) != 0) {
            if (bad < 5) {
                printf("row %d differs\n", r);
            }
            ++bad;
        }
    }
    printf("test-q8k-exact: %d rows, %d differ\n", rows, bad);
    return bad != 0;
}
