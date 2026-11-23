// Bit identity of the token-tiled gated delta-net kernel (ggml-cuda, GGML_CUDA_GDN_TILED=1) and the AR kernel
// (GGML_CUDA_GDN_TILED=0): speculative verify with LLAMA_MTP_EXACT=1 relies on every kernel of the op giving the bytes of
// the AR kernel. The kill switch is read once per process, so each mode runs in a child process (this binary with
// --child MODE FILE) that dumps the whole op output (attention scores + state snapshots) of every shape; the parent
// compares the dumps byte for byte. GGML_CUDA_GDN_TILED_LOG=1 makes the backend log every tiled launch: the tiled child
// must log exactly one launch per shape and the AR child none, so a silent fallback fails the test.
// Two tiled children: the base form (GGML_CUDA_GDN_TILED_PF=0) and the prefetch form (=1), each against the AR child; the
// prefetch child must log the prefetch form and the base child must not.
// Gated shapes (one sequence) build beta = sigmoid(b) and g = softplus(a + bias) * A in the graph. Where the backend hands the raw
// gates to the op (ggml_cuda_try_gdn_gates: not where the chunked mma kernel may run, e.g. sm 120; GGML_CUDA_GDN_GATES_DEBUG lines
// counted) the base tiled form runs the AR kernel for them and the prefetch form applies them itself; elsewhere the gates are separate
// kernels and every child runs them as plain shapes (noted, not a failure).
// Shapes keep the q / k head count != 16, so the chunked MMA kernel declines them on every arch.
// The kernel is ggml-cuda's and wave32 only (gdn_use_tiled): without a CUDA / ROCm device, or on a wave64 ROCm device (gfx90a,
// gfx942, ...), it prints SKIP and exits 0. Metal / Vulkan / SYCL / OpenCL devices are not used.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct gdn_shape {
    int64_t head_count; // q / k heads
    int     v_repeat;   // v heads = head_count * v_repeat
    int64_t head_size;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t K;
    bool    gates = false; // raw gates in the graph (sigmoid / add-softplus-mul before the op)
};

static std::vector<gdn_shape> shapes() {
    std::vector<gdn_shape> s;
    // H = 16 / 32 / 48 value heads with q / k heads 8 / 32 / 24
    const int64_t hk[3] = {8, 32, 24};
    const int     vr[3] = {2, 1, 2};
    for (int h = 0; h < 3; ++h) {
        for (int64_t nt : {16, 64, 512, 2047}) {
            for (int64_t ns : {1, 2}) {
                for (int64_t K : {1, 4}) {
                    s.push_back({hk[h], vr[h], 128, nt, ns, K});
                }
            }
        }
    }
    // head size 64
    s.push_back({8, 2, 64, 512, 2, 1});
    s.push_back({8, 2, 64, 100, 1, 4});
    // 48 value heads (q / k 24 x 2) with raw gates, one sequence (the gates go to the op for one sequence only)
    for (int64_t nt : {512, 2047}) {
        for (int64_t K : {1, 4}) {
            s.push_back({24, 2, 128, nt, 1, K, true});
        }
    }
    return s;
}

static int n_tiled_logs = 0;
static int n_pf_logs    = 0;
static int n_gate_logs  = 0; // the g chain handed to the op (GGML_CUDA_GDN_GATES_DEBUG=1, "a kept 1")
static std::string init_log; // the backend's device listing (ggml-cuda logs each device's wave size on ROCm)

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    init_log += text;
    if (strstr(text, ": reader ") != nullptr && strstr(text, "a kept 1") != nullptr) {
        n_gate_logs++;
    }
    if (strstr(text, "token-tiled gated delta-net, prefetch form") != nullptr) {
        n_pf_logs++;
    }
    if (strstr(text, "token-tiled gated delta-net kernel") != nullptr) {
        n_tiled_logs++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void fill(std::vector<float> & v, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & x : v) {
        x = d(rng);
    }
}

static bool is_rocm(ggml_backend_dev_t d) {
    return strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "ROCm") == 0;
}

// the first CUDA / ROCm GPU: the token-tiled kernel exists in ggml-cuda only
static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_IGPU) { // IGPU: Strix Halo, unified memory
            continue;
        }
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        if (strcmp(reg, "CUDA") == 0 || strcmp(reg, "ROCm") == 0) {
            return d;
        }
    }
    return nullptr;
}

// the wave size ggml-cuda logged for a ROCm device ("  Device N: ..., Wave Size: W, ..."), 0 when not found
static int rocm_wave_size(ggml_backend_dev_t d) {
    const char * name = ggml_backend_dev_name(d); // "ROCm<N>"
    const std::string key = std::string("Device ") + (name + 4) + ":";
    for (size_t pos = init_log.find(key); pos != std::string::npos; pos = init_log.find(key, pos + 1)) {
        const size_t eol = init_log.find('\n', pos);
        const size_t ws  = init_log.find("Wave Size: ", pos);
        if (ws != std::string::npos && (eol == std::string::npos || ws < eol)) {
            return atoi(init_log.c_str() + ws + strlen("Wave Size: "));
        }
    }
    return 0;
}

static int run_child(const char * path) {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) {
        fprintf(stderr, "child: no GPU device\n");
        return 3;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return 3;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return 4;
    }
    int si = 0;
    for (const gdn_shape & sh : shapes()) {
        const int64_t S = sh.head_size, Hk = sh.head_count, H = sh.head_count * sh.v_repeat, T = sh.n_tokens, N = sh.n_seqs;

        ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, N);
        ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, N);
        ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, N);
        ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, N);
        ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, N);
        // gated shapes: g and beta are the graph's gate outputs of raw a / b (as the qwen35 / qwen4exp graphs build them)
        ggml_tensor * a_raw = sh.gates ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, T, N) : nullptr;
        ggml_tensor * b_raw = sh.gates ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, T, N) : nullptr;
        ggml_tensor * a_b   = sh.gates ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        ggml_tensor * a_s   = sh.gates ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        ggml_tensor * g_op    = g;
        ggml_tensor * beta_op = beta;
        if (sh.gates) {
            g_op    = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, a_raw, a_b)), a_s), 1, H, T, N);
            beta_op = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, b_raw), 1, H, T, N);
        }
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, N);
        ggml_tensor * qn    = ggml_l2_norm(ctx, q, 1e-6f);
        ggml_tensor * kn    = ggml_l2_norm(ctx, k, 1e-6f);
        ggml_tensor * out   = ggml_gated_delta_net(ctx, qn, kn, v, g_op, beta_op, state, sh.K);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            fclose(f);
            return 5;
        }

        std::mt19937 rng(1234 + si);
        auto set = [&](ggml_tensor * t, float lo, float hi) {
            std::vector<float> d(ggml_nelements(t));
            fill(d, rng, lo, hi);
            ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
        };
        set(q, -1.0f, 1.0f);
        set(k, -1.0f, 1.0f);
        set(v, -0.3f, 5.0f);
        set(g, -20.0f, -1e-4f);
        set(beta, 0.0f, 1.0f);
        set(state, -1.0f, 1.0f);
        if (sh.gates) {
            set(a_raw, -6.0f, 24.0f); // both softplus branches (x > 20 returns x)
            set(b_raw, -4.0f, 4.0f);
            set(a_b, -1.0f, 1.0f);
            set(a_s, -2.0f, -0.05f);
        }

        // the backend may leave snapshot slots it does not own unwritten: start from the same bytes in both children
        {
            std::vector<uint8_t> z(ggml_nbytes(out), 0);
            ggml_backend_tensor_set(out, z.data(), 0, z.size());
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fclose(f);
            return 6;
        }
        std::vector<uint8_t> o(ggml_nbytes(out));
        ggml_backend_tensor_get(out, o.data(), 0, o.size());
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), 1, o.size(), f);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        si++;
    }
    // the prefetch-form engagement and the gate hand-overs in the high digits
    const int n_logs_packed = n_tiled_logs + 1000 * n_gate_logs + 1000000 * n_pf_logs;
    fwrite(&n_logs_packed, sizeof(n_logs_packed), 1, f);
    fclose(f);
    ggml_backend_free(backend);
    return 0;
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static bool read_dump(const std::string & path, std::vector<std::vector<uint8_t>> & outs, int & n_logs, size_t n_shapes) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    for (size_t i = 0; i < n_shapes; ++i) {
        uint64_t n = 0;
        if (fread(&n, sizeof(n), 1, f) != 1) {
            fclose(f);
            return false;
        }
        outs.emplace_back(n);
        if (fread(outs.back().data(), 1, n, f) != n) {
            fclose(f);
            return false;
        }
    }
    const bool ok = fread(&n_logs, sizeof(n_logs), 1, f) == 1;
    fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }

    // the parent never initialises a GPU: only decide whether there is one (the registry enumerates without a context)
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) {
        printf("SKIP: no CUDA / ROCm device (the token-tiled gated delta-net kernel is ggml-cuda only)\n");
        return 0;
    }
    if (is_rocm(dev)) {
        const int ws = rocm_wave_size(dev);
        if (ws != 32) {
            printf("SKIP: %s has wave size %d (the token-tiled kernel runs on wave32 devices only)\n", ggml_backend_dev_name(dev), ws);
            return 0;
        }
    }

    const std::vector<gdn_shape> sh = shapes();
    // mode 0: AR, 1: tiled base form, 2: tiled prefetch form
    std::vector<std::vector<uint8_t>> outs[3];
    int n_logs[3] = {-1, -1, -1};
    int n_pf[3]   = {-1, -1, -1};
    int n_gates[3] = {-1, -1, -1};
    size_t n_gated = 0;
    for (const gdn_shape & s : sh) {
        n_gated += s.gates ? 1 : 0;
    }
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/gdn_tiled_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_GDN_TILED", mode ? "1" : "0");
        set_env("GGML_CUDA_GDN_TILED_PF", mode == 2 ? "1" : "0");
        set_env("GGML_CUDA_GDN_GATES_DEBUG", "1");
        set_env("GGML_CUDA_GDN_TILED_LOG", "1");
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
#ifdef _WIN32
        cmd = "\"" + cmd + "\""; // cmd.exe /c strips the outer pair of quotes of a line with more than two
#endif
        const int rc = std::system(cmd.c_str());
        if (rc != 0 || !read_dump(path, outs[mode], n_logs[mode], sh.size())) {
            fprintf(stderr, "FAIL: child mode %d rc %d\n", mode, rc);
            return 1;
        }
        std::remove(path.c_str());
        n_pf[mode]    = n_logs[mode] / 1000000;
        n_gates[mode] = n_logs[mode] / 1000 % 1000;
        n_logs[mode]  = n_logs[mode] % 1000;
    }

    int n_fail = 0;
    for (int mode = 1; mode < 3; ++mode)
    for (size_t i = 0; i < sh.size(); ++i) {
        const gdn_shape & s = sh[i];
        const auto & a = outs[0][i];
        const auto & b = outs[mode][i];
        const bool same = a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0;
        size_t first = 0;
        if (!same) {
            while (first < a.size() && first < b.size() && a[first] == b[first]) {
                first++;
            }
            n_fail++;
        }
        const int64_t attn_bytes = s.head_size * s.head_count * s.v_repeat * s.n_tokens * s.n_seqs * (int64_t) sizeof(float);
        printf("  GDN tiled %s vs AR: Hk=%3lld H=%3lld S_v=%3lld n_tokens=%5lld n_seqs=%lld K=%lld bytes=%zu: %s",
               mode == 2 ? "prefetch" : "base", (long long) s.head_count, (long long) (s.head_count * s.v_repeat), (long long) s.head_size,
               (long long) s.n_tokens, (long long) s.n_seqs, (long long) s.K, a.size(), same ? "memcmp == 0\n" : "DIFFER");
        if (!same) {
            printf(" (first differing byte %zu, in the %s)\n", first, (int64_t) first < attn_bytes ? "attention output" : "state");
        }
    }
    printf("tiled launches logged: AR child %d, tiled base child %d, tiled prefetch child %d (expected 0 / %zu / %zu); "
           "prefetch form logged: %d / %d / %d (expected 0 / 0 / at least 1); raw gates handed to the op: %d / %d / %d (expected at least %zu each)\n",
           n_logs[0], n_logs[1], n_logs[2], (size_t) (n_gates[0] > 0 ? sh.size() - n_gated : sh.size()), sh.size(), n_pf[0], n_pf[1], n_pf[2],
           n_gates[0], n_gates[1], n_gates[2], n_gated);
    // the gates go to the op in every child or in none (a property of the device, not of the mode)
    const bool handed = n_gates[0] > 0;
    if ((n_gates[0] > 0) != (n_gates[1] > 0) || (n_gates[0] > 0) != (n_gates[2] > 0) || (handed && (n_gates[0] < (int) n_gated ||
            n_gates[1] < (int) n_gated || n_gates[2] < (int) n_gated))) {
        printf("FAIL: the raw gates went to the op in some children only, or not for every gated shape\n");
        return 1;
    }
    if (!handed) {
        printf("NOTE: this device does not hand the raw gates to the op; the gated shapes ran as plain ones in every child\n");
    }
    const int n_base_tiled = (int) (handed ? sh.size() - n_gated : sh.size());
    if (n_logs[0] != 0 || n_logs[1] != n_base_tiled || n_logs[2] != (int) sh.size()) {
        printf("FAIL: the tiled kernel did not run exactly once per shape in the tiled children (or ran in the AR child)\n");
        return 1;
    }
    if (n_pf[0] != 0 || n_pf[1] != 0 || n_pf[2] < 1) {
        printf("FAIL: the prefetch form did not run in the prefetch child only\n");
        return 1;
    }
    if (n_fail) {
        printf("FAIL: %d / %zu shape comparisons differ\n", n_fail, 2 * sh.size());
        return 1;
    }
    printf("OK: %zu / %zu shapes bit-identical in both tiled forms\n", sh.size(), sh.size());
    return 0;
}
