#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
int main() {
    ggml_cpu_init();
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const int n = 4096, rows = 64;
    std::vector<float> x((size_t) n*rows);
    std::mt19937 rng(1); std::normal_distribution<float> nd;
    for (auto & v : x) v = nd(rng);
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, n)*rows);
    std::vector<double> t;
    for (int rep = 0; rep < 31; ++rep) {
        auto a = std::chrono::steady_clock::now();
        for (int it = 0; it < 50; ++it) tr->from_float(x.data(), y.data(), (int64_t) n*rows);
        auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(b - a).count()/50);
    }
    std::sort(t.begin(), t.end());
    printf("median %.1f us per %d floats (p10 %.1f p90 %.1f)\n", t[15], n*rows, t[3], t[27]);
}
