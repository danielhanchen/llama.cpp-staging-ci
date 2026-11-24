// bit identity of the paired-route mat-vec MUL_MAT_ID on Metal (GGML_METAL_MMID_PAIR=1) against the single-route kernel
// (GGML_METAL_MMID_PAIR=0): verify windows of 2 to 16 tokens whose routes collide on a few experts (and repeated experts within
// a token), Q8_0 / Q4_K / Q5_K / Q6_K experts, src1 per route and broadcast. The two arms run in child processes of this binary
// (the switch is read once per process); the parent compares the output bytes of every case. The CPU backend is a sanity
// reference (NMSE, not bytes). A negative control changes one src1 value: its output must differ. Prints the number of cases,
// the paired / single entries of each case (from the reference list builder, ggml-metal-mmid-pair.h) and the differing ones;
// exit 1 on any difference. SKIP (exit 0) without a Metal device or when MTL0 does not support the op.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../ggml/src/ggml-metal/ggml-metal-mmid-pair.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

struct pair_case {
    ggml_type type;
    int64_t   K, M, n_as, n_used, T;
    bool      bcast;     // src1 [K, 1, T] instead of [K, n_used, T]
    bool      dup;       // repeated experts within a token
    bool      control;   // case 0 with one src1 value changed
};

static std::vector<pair_case> make_cases() {
    std::vector<pair_case> cases;
    const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    struct cfg { int64_t n_used, n_as; };
    const cfg cfgs[] = { { 1, 4 }, { 2, 8 }, { 8, 128 }, { 10, 512 } };
    int i = 0;
    for (ggml_type t : types) {
        for (const cfg & c : cfgs) {
            for (int64_t T : { 2, 3, 5, 8, 16 }) {
                const int64_t M = (i % 2) ? 70 : 96;
                cases.push_back({ t, 512, M, c.n_as, c.n_used, T, c.n_used == 1 && T % 2 == 1, false, false });
                i++;
            }
        }
        cases.push_back({ t, 768, 64, 16, 4, 7, false, true, false });
    }
    pair_case ctl = cases[0];
    ctl.control = true;
    cases.push_back(ctl);
    return cases;
}

// ids of a window that collides: the experts of every token come from a pool a little larger than n_used
static uint32_t case_seed(const pair_case & c) {
    return 1234 + (uint32_t) (c.K + 31*c.M + 131*c.n_as + 977*c.n_used + 7919*c.T + 104729*(int) c.type + 3*c.dup + 5*c.bcast);
}

static std::vector<int32_t> make_ids(const pair_case & c) {
    std::mt19937 rng(case_seed(c) ^ 0x1D5u);
    std::vector<int32_t> ids(c.n_used*c.T);
    const int64_t pool = std::min<int64_t>(c.n_as, c.n_used + 2);
    std::vector<int32_t> base(pool);
    for (int64_t j = 0; j < pool; ++j) {
        base[j] = (int32_t) ((j*7919 + 13) % c.n_as);
    }
    for (int64_t t = 0; t < c.T; ++t) {
        std::vector<int32_t> p = base;
        std::shuffle(p.begin(), p.end(), rng);
        for (int64_t s = 0; s < c.n_used; ++s) {
            ids[t*c.n_used + s] = c.dup && s > 0 && rng() % 3 == 0 ? ids[t*c.n_used + s - 1] : p[s];
        }
    }
    return ids;
}

// output of one case on one backend; empty with *supported = false when the backend does not run the op
static std::vector<float> run_case(ggml_backend_t be, const pair_case & c, bool * supported) {
    std::mt19937 rng(case_seed(c));

    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const int64_t n_b = c.bcast ? 1 : c.n_used;

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, c.type, c.K, c.M, c.n_as);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, n_b, c.T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.T);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);

    *supported = ggml_backend_supports_op(be, out);
    if (!*supported) {
        ggml_free(ctx);
        return {};
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> wf(c.K*c.M*c.n_as);
    for (float & v : wf) {
        v = ud(rng);
    }
    std::vector<uint8_t> wq(ggml_nbytes(as));
    ggml_quantize_chunk(c.type, wf.data(), wq.data(), 0, c.M*c.n_as, c.K, nullptr);
    ggml_backend_tensor_set(as, wq.data(), 0, wq.size());

    std::vector<float> bf(c.K*n_b*c.T);
    for (float & v : bf) {
        v = ud(rng);
    }
    if (c.control) {
        bf[c.K + 3] += 0.5f;
    }
    ggml_backend_tensor_set(b, bf.data(), 0, bf.size()*sizeof(float));

    std::vector<int32_t> iv = make_ids(c);
    ggml_backend_tensor_set(ids, iv.data(), 0, iv.size()*sizeof(int32_t));

    std::vector<float> res;
    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return res;
}

static ggml_backend_t init_metal() {
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                strncmp(ggml_backend_dev_name(dev), "MTL", 3) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// child: every case on MTL0, written as (int32 supported, int64 n, n floats) records
static int child(const char * path) {
    ggml_backend_t be = init_metal();
    if (be == nullptr) {
        return 2;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return 3;
    }
    for (const pair_case & c : make_cases()) {
        bool sup = false;
        std::vector<float> r = run_case(be, c, &sup);
        const int32_t s = sup && !r.empty();
        const int64_t n = (int64_t) r.size();
        fwrite(&s, sizeof(s), 1, f);
        fwrite(&n, sizeof(n), 1, f);
        if (n > 0) {
            fwrite(r.data(), sizeof(float), n, f);
        }
    }
    fclose(f);
    ggml_backend_free(be);
    return 0;
}

struct rec {
    int32_t supported;
    std::vector<float> v;
};

static bool read_recs(const std::string & path, size_t n_cases, std::vector<rec> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    out.assign(n_cases, {});
    for (size_t i = 0; i < n_cases; ++i) {
        int64_t n = 0;
        if (fread(&out[i].supported, sizeof(int32_t), 1, f) != 1 || fread(&n, sizeof(n), 1, f) != 1) {
            fclose(f);
            return false;
        }
        out[i].v.resize(n);
        if (n > 0 && fread(out[i].v.data(), sizeof(float), n, f) != (size_t) n) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    if (argc == 3 && strcmp(argv[1], "--child") == 0) {
        return child(argv[2]);
    }

    ggml_backend_t mtl = init_metal();
    if (mtl == nullptr) {
        printf("test-metal-mmid-pair-exact: SKIP (no Metal device)\n");
        return 0;
    }
    ggml_backend_free(mtl);

    const std::vector<pair_case> cases = make_cases();

    const char * tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    std::string out_path[2];
    int n_log[2] = { 0, 0 };
    for (int arm = 0; arm < 2; ++arm) {
        const std::string stem = std::string(tmp) + "/test-metal-mmid-pair-" + std::to_string((long) getpid()) + "-" + std::to_string(arm);
        out_path[arm] = stem + ".bin";
        const std::string log_path = stem + ".log";
        // GGML_METAL_MMID_PAIR_DEBUG=1 in both arms: one log line per op that takes the paired path (0 in the off arm)
        const std::string cmd = std::string("GGML_METAL_MMID_PAIR=") + std::to_string(arm) + " GGML_METAL_MMID_PAIR_DEBUG=1 '" + argv[0] +
            "' --child '" + out_path[arm] + "' 2> '" + log_path + "'";
        const int rc = system(cmd.c_str());
        FILE * lf = fopen(log_path.c_str(), "r");
        if (lf != nullptr) {
            char line[1024];
            while (fgets(line, sizeof(line), lf)) {
                n_log[arm] += strstr(line, "mmid pair:") != nullptr;
            }
            fclose(lf);
            unlink(log_path.c_str());
        }
        if (rc != 0) {
            printf("test-metal-mmid-pair-exact: child (GGML_METAL_MMID_PAIR=%d) failed, rc %d\n", arm, rc);
            return 1;
        }
    }

    std::vector<rec> r[2];
    for (int arm = 0; arm < 2; ++arm) {
        if (!read_recs(out_path[arm], cases.size(), r[arm])) {
            printf("test-metal-mmid-pair-exact: cannot read %s\n", out_path[arm].c_str());
            return 1;
        }
        unlink(out_path[arm].c_str());
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    int n_run = 0, n_unsup = 0, n_diff = 0, n_bad_ref = 0, n_pairs_all = 0, n_singles_all = 0;
    bool ctl_ok = false;

    for (size_t i = 0; i < cases.size(); ++i) {
        const pair_case & c = cases[i];
        if (!r[0][i].supported || !r[1][i].supported) {
            n_unsup++;
            continue;
        }
        if (c.control) {
            // differs from case 0 (same inputs but one src1 value)
            ctl_ok = r[1][i].v.size() == r[1][0].v.size() && memcmp(r[1][i].v.data(), r[1][0].v.data(), r[1][0].v.size()*sizeof(float)) != 0;
            continue;
        }
        n_run++;

        // the entries the map kernel builds for these ids
        const std::vector<int32_t> iv = make_ids(c);
        std::vector<int32_t> list(ggml_metal_mmid_pair_list_len(c.n_used*c.T));
        const int32_t n_ent = ggml_metal_mmid_pair_build(iv.data(), c.n_used, (int32_t) c.n_used, (int32_t) c.T, list.data());
        int n_pairs = 0;
        for (int32_t e = 0; e < n_ent; ++e) {
            n_pairs += list[1 + 3*e + 2] >= 0;
        }
        n_pairs_all   += n_pairs;
        n_singles_all += n_ent - n_pairs;

        size_t nd = 0;
        if (r[0][i].v.size() != r[1][i].v.size()) {
            nd = r[0][i].v.size() + 1;
        } else {
            for (size_t j = 0; j < r[0][i].v.size(); ++j) {
                nd += memcmp(&r[0][i].v[j], &r[1][i].v[j], sizeof(float)) != 0;
            }
        }

        double nmse = 0.0;
        if (cpu != nullptr) {
            bool sup = false;
            std::vector<float> ref = run_case(cpu, c, &sup);
            if (sup && ref.size() == r[0][i].v.size()) {
                double num = 0.0, den = 0.0;
                for (size_t j = 0; j < ref.size(); ++j) {
                    num += (r[0][i].v[j] - ref[j])*(double) (r[0][i].v[j] - ref[j]);
                    den += ref[j]*(double) ref[j];
                }
                nmse = den > 0 ? num/den : num;
                n_bad_ref += !(nmse < 5e-4);
            }
        }

        printf("  %-5s K %4lld M %3lld n_as %3lld used %2lld T %2lld%s%s: entries %3d (pairs %3d, singles %3d), nmse vs CPU %.2e, differ %zu\n",
            ggml_type_name(c.type), (long long) c.K, (long long) c.M, (long long) c.n_as, (long long) c.n_used, (long long) c.T,
            c.bcast ? " bcast" : "", c.dup ? " dup" : "", n_ent, n_pairs, n_ent - n_pairs, nmse, nd);

        n_diff += nd != 0;
    }

    if (cpu != nullptr) {
        ggml_backend_free(cpu);
    }

    if (n_run == 0) {
        printf("test-metal-mmid-pair-exact: SKIP (MTL0 does not support MUL_MAT_ID: %d of %zu cases not supported)\n", n_unsup, cases.size());
        return 0;
    }

    // the paired path must have run in the on arm (one line per op) and never in the off arm
    const bool engaged = n_log[1] >= n_run && n_log[0] == 0;

    printf("test-metal-mmid-pair-exact: %d cases, %d differ, %d not supported, entries %d paired + %d single, CPU sanity bad %d, control %s, "
        "paired-path ops %d (off arm %d)%s\n",
        n_run, n_diff, n_unsup, n_pairs_all, n_singles_all, n_bad_ref, ctl_ok ? "ok" : "FAIL", n_log[1], n_log[0], engaged ? "" : " NOT ENGAGED");

    return n_diff == 0 && n_bad_ref == 0 && ctl_ok && engaged ? 0 : 1;
}
