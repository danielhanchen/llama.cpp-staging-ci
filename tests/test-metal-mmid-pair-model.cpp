// the list of paired routes of the Metal mat-vec MUL_MAT_ID (ggml-metal-mmid-pair.h, the reference of kernel_mul_mv_id_pair_map)
// against an independent sequential pairing, on random ids of verify windows (1 to 10 experts used, 2 to 16 tokens, 4 to 512
// experts, few-expert pools so routes collide, repeated experts within a token). Checks: the list equals the sequential one entry
// for entry, every route is covered exactly once, the routes of an entry share its expert, a pair's routes are in increasing order,
// the count is at most the number of routes (the grid bound of the paired kernel), and a second build gives the same list.
// CPU only: no device needed.

#include "../ggml/src/ggml-metal/ggml-metal-mmid-pair.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

// pair the routes of each expert in route order, the entries ordered by their first route
static std::vector<int32_t> sequential(const std::vector<int32_t> & ids, int32_t n_used, int32_t n_tokens, int64_t stride) {
    const int32_t n = n_used*n_tokens;
    std::map<int32_t, std::vector<int32_t>> by_expert;
    for (int32_t p = 0; p < n; ++p) {
        by_expert[ids[(p/n_used)*stride + p%n_used]].push_back(p);
    }
    std::map<int32_t, std::pair<int32_t, int32_t>> entries; // first route -> (expert, second route)
    for (const auto & [e, rs] : by_expert) {
        for (size_t i = 0; i < rs.size(); i += 2) {
            entries[rs[i]] = { e, i + 1 < rs.size() ? rs[i + 1] : -1 };
        }
    }
    std::vector<int32_t> out(ggml_metal_mmid_pair_list_len(n), 0);
    out[0] = (int32_t) entries.size();
    int32_t k = 0;
    for (const auto & [a, eb] : entries) {
        out[1 + 3*k + 0] = eb.first;
        out[1 + 3*k + 1] = a;
        out[1 + 3*k + 2] = eb.second;
        k++;
    }
    return out;
}

int main() {
    std::mt19937 rng(42);

    int n_cases = 0, n_bad = 0;
    long long n_pairs = 0, n_singles = 0;

    for (int it = 0; it < 4000; ++it) {
        const int32_t n_used   = 1 + (int32_t) (rng() % 10);
        const int32_t n_tokens = 2 + (int32_t) (rng() % 15);
        const int32_t n_expert = std::max<int32_t>(n_used, 4 + (int32_t) (rng() % 509));
        const int32_t pool     = std::min<int32_t>(n_expert, n_used + (int32_t) (rng() % 6));
        const bool    dup      = rng() % 4 == 0;
        const int64_t stride   = n_used + (int64_t) (rng() % 3); // row stride of the ids in int32 (padded rows)

        if (n_used*n_tokens > GGML_METAL_MMID_PAIR_MAX_ROUTES) {
            continue;
        }

        std::vector<int32_t> experts(pool);
        for (int32_t j = 0; j < pool; ++j) {
            experts[j] = (int32_t) (rng() % n_expert);
        }

        std::vector<int32_t> ids(n_tokens*stride, -7);
        for (int32_t t = 0; t < n_tokens; ++t) {
            for (int32_t s = 0; s < n_used; ++s) {
                ids[t*stride + s] = dup && s > 0 && rng() % 3 == 0 ? ids[t*stride + s - 1] : experts[rng() % pool];
            }
        }

        const int32_t n = n_used*n_tokens;
        std::vector<int32_t> a(ggml_metal_mmid_pair_list_len(n), 0);
        std::vector<int32_t> b(ggml_metal_mmid_pair_list_len(n), 0);
        const int32_t na = ggml_metal_mmid_pair_build(ids.data(), stride, n_used, n_tokens, a.data());
        const int32_t nb = ggml_metal_mmid_pair_build(ids.data(), stride, n_used, n_tokens, b.data());

        bool ok = na == nb && a == b && na == a[0] && na <= n;

        const std::vector<int32_t> ref = sequential(ids, n_used, n_tokens, stride);
        ok = ok && ref == a;

        std::vector<int> cover(n, 0);
        for (int32_t e = 0; e < na && ok; ++e) {
            const int32_t ex = a[1 + 3*e], pa = a[1 + 3*e + 1], pb = a[1 + 3*e + 2];
            ok = ok && pa >= 0 && pa < n && ids[(pa/n_used)*stride + pa%n_used] == ex;
            if (ok) {
                cover[pa]++;
            }
            if (pb >= 0) {
                ok = ok && pb > pa && pb < n && ids[(pb/n_used)*stride + pb%n_used] == ex;
                if (ok) {
                    cover[pb]++;
                }
                n_pairs++;
            } else {
                n_singles++;
            }
            ok = ok && (e == 0 || a[1 + 3*(e - 1) + 1] < pa);
        }
        for (int32_t p = 0; p < n && ok; ++p) {
            ok = cover[p] == 1;
        }

        n_cases++;
        if (!ok) {
            n_bad++;
            if (n_bad <= 5) {
                printf("  FAIL: used %d tokens %d experts %d pool %d dup %d stride %lld\n", n_used, n_tokens, n_expert, pool, (int) dup, (long long) stride);
            }
        }
    }

    printf("test-metal-mmid-pair-model: %d cases, %d bad, entries %lld paired + %lld single\n", n_cases, n_bad, n_pairs, n_singles);

    return n_bad == 0 && n_cases > 0 ? 0 : 1;
}
