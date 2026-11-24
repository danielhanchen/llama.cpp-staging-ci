// The exact int-to-float product of the CUDA q8_0 MMQ vec_dot (ggml_cuda_mmq_cvt_mul, GGML_CUDA_MMQ_EXACT_CVT) on the CPU:
// for an MMA accumulator x (|x| <= 32*128*128 = 2^19 for one k32 tile, the kernel needs |x| < 2^22) and an x scale d of the
// types that use it, fmaf(bits(0x4B400000 + x), d, -(1.5 * 2^23) * d) must equal (float) x * d bit for bit.
// The scales checked: every finite fp16 value (Q1_0 / Q2_0 / Q4_0 / Q5_0 / Q8_0 / IQ4_NL: d itself) and the IQ2_XXS / IQ3_XXS /
// IQ3_S / IQ4_XS forms (d * ls / 8, (ls*d + d/2)/2, ls*d, d * (ls - 32)) over all their ls, each against a set of x, and every
// x in [-2^19, 2^19] against a sample of those scales. Also the upper end of the valid range (|x| up to 2^22 - 1).
#include "ggml.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float bits_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t float_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

// the kernel's arithmetic: the accumulator as the low bits of 1.5 * 2^23, one fused multiply-add with the scaled bias
static float cvt_mul(int32_t x, float d) {
    const float nd = -12582912.0f * d;
    return std::fmaf(bits_to_float(0x4B400000u + (uint32_t) x), d, nd);
}

// the reference: the conversion, then the product
static float ref_mul(int32_t x, float d) {
    return (float) x * d;
}

// equal bits, except that a zero product may differ in sign (x = 0 with d < 0: -0 vs +0). The kernel adds the product to an
// accumulator that starts at +0 and never becomes -0 (+0 + -0 = +0 in round to nearest), so the sums are equal either way.
static bool same(float a, float b) {
    return float_to_bits(a) == float_to_bits(b) || (a == 0.0f && b == 0.0f);
}

int main() {
    std::vector<float> scales;
    for (uint32_t h = 0; h < 0x10000; ++h) {
        const float d = ggml_fp16_to_fp32((ggml_fp16_t) h);
        if (std::isfinite(d)) {
            scales.push_back(d);
        }
    }
    const size_t n_fp16 = scales.size();
    std::vector<float> derived; // the IQ scale forms, from a spread of fp16 d
    for (size_t i = 0; i < n_fp16; i += 97) {
        const float d = scales[i];
        for (int ls = 1; ls < 32; ls += 2) {
            derived.push_back(d * ls / 8);         // IQ2_XXS (ls = 2*scale + 1)
            derived.push_back(ls * d);             // IQ3_S (ls = 1 + 2*nibble)
        }
        for (int ls = 0; ls < 16; ++ls) {
            derived.push_back((ls*d + d/2)/2);     // IQ3_XXS
        }
        for (int ls = 0; ls < 64; ++ls) {
            derived.push_back(d * (ls - 32));      // IQ4_XS
        }
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> ux(-(1 << 19), 1 << 19);
    std::vector<int32_t> xs = { 0, 1, -1, 2, -2, 127, -128, 16129, -16384, (1 << 19), -(1 << 19), (1 << 22) - 1, -(1 << 22) };
    for (int i = 0; i < 48; ++i) {
        xs.push_back(ux(rng));
    }

    int64_t n = 0, bad = 0;
    auto check = [&](int32_t x, float d) {
        ++n;
        const float a = cvt_mul(x, d);
        const float b = ref_mul(x, d);
        if (!same(a, b)) {
            if (bad < 10) {
                printf("MISMATCH x %" PRId32 " d %a: exact %a (0x%08x) conversion %a (0x%08x)\n", x, (double) d, (double) a,
                    float_to_bits(a), (double) b, float_to_bits(b));
            }
            ++bad;
        }
    };
    for (const float d : scales) {
        for (const int32_t x : xs) {
            check(x, d);
        }
    }
    for (const float d : derived) {
        for (const int32_t x : xs) {
            check(x, d);
        }
    }
    std::uniform_int_distribution<size_t> us(0, n_fp16 - 1), ud(0, derived.size() - 1);
    std::vector<float> few = { 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f, ggml_fp16_to_fp32((ggml_fp16_t) 1) /* smallest subnormal */ };
    for (int i = 0; i < 10; ++i) {
        few.push_back(scales[us(rng)]);
        few.push_back(derived[ud(rng)]);
    }
    for (int32_t x = -(1 << 19); x <= (1 << 19); ++x) {
        for (const float d : few) {
            check(x, d);
        }
    }
    printf("%" PRId64 " products checked (%zu fp16 scales, %zu derived IQ scales), %" PRId64 " mismatches\n", n, n_fp16, derived.size(), bad);
    if (bad != 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
