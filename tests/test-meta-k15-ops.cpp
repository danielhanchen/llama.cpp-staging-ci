// the ops this tree adds to ggml (GGML_OP_KQ_MASK, GGML_OP_MOE_CACHE_LOOKUP, the GATED_DELTA_NET replay window) on the meta backend
// (-sm tensor) and over RPC: neither has a split state / remote check for them, so both must refuse them in supports_op and the scheduler
// must then compute them on the CPU and copy the result, as llama's KQ mask graph needs (llama builds KQ_MASK for ubatches of 64+ tokens).
// A meta device over the CPU (and, with the RPC backend loaded, an in-process RPC server over the CPU) gets a graph whose KQ_MASK feeds an
// ADD with a weight in the meta / RPC buffer; the result is checked against the causal mask. Before the fix the meta case aborted with
// "ggml op not implemented: KQ_MASK".
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// the RPC client aborts when it cannot connect: wait until the in-process server listens
static bool port_open(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in a = {};
    a.sin_family      = AF_INET;
    a.sin_port        = htons((uint16_t) port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok = connect(fd, (const sockaddr *) &a, sizeof(a)) == 0;
    close(fd);
    return ok;
}
#endif

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    s.n_segments = 1;
    s.nr[0]      = 1;
    return s;
}

static const int n_kv  = 96;
static const int n_tok = 64;

// a dummy node of the given op (supports_op of meta / RPC look at the op and its op params first)
static bool refuses(ggml_backend_dev_t dev, ggml_op op, int32_t p1) {
    ggml_tensor t = {};
    t.type = GGML_TYPE_F32;
    t.op   = op;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        t.ne[i] = 1;
    }
    ((int32_t *) t.op_params)[1] = p1;
    return !ggml_backend_dev_supports_op(dev, &t);
}

// KQ_MASK -> ADD(mask, w) with w in a buffer of `be`, scheduled over { be, cpu }; returns the number of failures
static int run_graph(const char * name, ggml_backend_t be, ggml_backend_t cpu) {
    int fails = 0;

    ggml_init_params wp = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context_ptr ctx_w(ggml_init(wp));
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w.get(), GGML_TYPE_F32, n_kv, n_tok);
    ggml_set_name(w, "w");
    ggml_backend_buffer_ptr wbuf(ggml_backend_alloc_ctx_tensors(ctx_w.get(), be));
    if (!wbuf) {
        printf("%s: weight alloc failed\n", name);
        return 1;
    }
    ggml_backend_buffer_set_usage(wbuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> wv((size_t) n_kv*n_tok, 1.0f);
    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));

    ggml_init_params gp = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(gp));
    ggml_tensor * cells = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, 5, n_kv, 1);
    ggml_tensor * toks  = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, 8, n_tok, 1);
    ggml_set_input(cells);
    ggml_set_input(toks);
    ggml_tensor * mask = ggml_kq_mask(ctx.get(), cells, toks, GGML_TYPE_F32);
    // as llama-graph: the op params keep the host inputs
    ggml_tensor * srcs[2] = { cells, toks };
    memcpy(mask->op_params, srcs, sizeof(srcs));
    ggml_tensor * out = ggml_add(ctx.get(), ggml_reshape_2d(ctx.get(), mask, n_kv, n_tok), w);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backends[2] = { be, cpu };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 64, false, true));
    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        printf("%s: sched alloc failed\n", name);
        return 1;
    }
    // cells: position j, sequence 0; tokens: positions n_kv - n_tok .. n_kv - 1 of sequence 0, causal
    std::vector<int32_t> cv(5*n_kv, 0), tv(8*n_tok, 0);
    for (int j = 0; j < n_kv; ++j) {
        cv[5*j + 0] = j;
        cv[5*j + 1] = 1;
    }
    for (int i = 0; i < n_tok; ++i) {
        tv[8*i + 0] = n_kv - n_tok + i;
        tv[8*i + 2] = 1;
    }
    ggml_backend_tensor_set(cells, cv.data(), 0, ggml_nbytes(cells));
    ggml_backend_tensor_set(toks,  tv.data(), 0, ggml_nbytes(toks));

    ggml_backend_t mask_be = ggml_backend_sched_get_tensor_backend(sched.get(), mask);
    ggml_backend_t out_be  = ggml_backend_sched_get_tensor_backend(sched.get(), out);
    printf("%s: KQ_MASK on %s, ADD on %s, %d splits\n", name, mask_be ? ggml_backend_name(mask_be) : "?",
        out_be ? ggml_backend_name(out_be) : "?", ggml_backend_sched_get_n_splits(sched.get()));
    if (mask_be != cpu) {
        printf("%s: FAIL: KQ_MASK not on the CPU\n", name);
        fails++;
    }
    if (out_be != be) {
        printf("%s: FAIL: the ADD is not on %s (the test does not cover the copy)\n", name, ggml_backend_name(be));
        fails++;
    }
    if (ggml_backend_sched_graph_compute(sched.get(), gf) != GGML_STATUS_SUCCESS) {
        printf("%s: FAIL: compute\n", name);
        return fails + 1;
    }
    std::vector<float> res((size_t) n_kv*n_tok);
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    int bad = 0;
    for (int i = 0; i < n_tok; ++i) {
        for (int j = 0; j < n_kv; ++j) {
            const float v   = res[(size_t) i*n_kv + j];
            const bool keep = j <= n_kv - n_tok + i;
            if (keep ? v != 1.0f : !(std::isinf(v) && v < 0)) {
                bad++;
            }
        }
    }
    printf("%s: %d of %d mask values wrong\n", name, bad, n_kv*n_tok);
    return fails + (bad != 0);
}

int main() {
    ggml_backend_load_all();
    int fails = 0;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        printf("no CPU device\n");
        return 1;
    }
    ggml_backend_ptr cpu(ggml_backend_dev_init(cpu_dev, nullptr));

    // meta (-sm tensor) over the CPU
    {
        ggml_backend_dev_t devs[1] = { cpu_dev };
        ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 1, mirrored, nullptr);
        const bool r_kq = refuses(meta, GGML_OP_KQ_MASK, 0), r_mc = refuses(meta, GGML_OP_MOE_CACHE_LOOKUP, 0);
        const bool r_gdn_replay = refuses(meta, GGML_OP_GATED_DELTA_NET, 1);
        printf("meta supports_op refuses: KQ_MASK %d, MOE_CACHE_LOOKUP %d, GATED_DELTA_NET replay %d\n", r_kq, r_mc, r_gdn_replay);
        fails += !r_kq + !r_mc + !r_gdn_replay;
        // with KQ_MASK accepted the scheduler puts the mask on the meta backend, which aborts ("ggml op not implemented: KQ_MASK")
        ggml_backend_ptr be(ggml_backend_dev_init(meta, nullptr));
        fails += run_graph("meta", be.get(), cpu.get());
    }

    // RPC: an in-process server over the CPU on a local port (not on Windows: the test waits for the port with BSD sockets)
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
#ifdef _WIN32
    rpc_reg = nullptr;
#endif
    if (rpc_reg == nullptr) {
        printf("RPC: backend not built (or Windows), skipped\n");
    } else {
#ifndef _WIN32
        typedef void (*start_server_t)(const char *, const char *, size_t, size_t, ggml_backend_dev_t *);
        typedef ggml_backend_reg_t (*add_server_t)(const char *);
        auto start_server = (start_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server");
        auto add_server   = (add_server_t)   ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
        if (start_server == nullptr || add_server == nullptr) {
            printf("RPC: FAIL: no server entry points\n");
            fails++;
        } else {
            int port = 20000 + (int) (getpid() % 30000);
            for (int i = 0; i < 20 && port_open(port); ++i) {
                port++;
            }
            const std::string ep = "127.0.0.1:" + std::to_string(port);
            // the server never returns; the process ends with it running (std::_Exit below)
            std::thread([start_server, ep, cpu_dev]() {
                ggml_backend_dev_t sdevs[1] = { cpu_dev };
                start_server(ep.c_str(), nullptr, 2, 1, sdevs);
            }).detach();
            bool up = false;
            for (int i = 0; i < 200 && !up; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                up = port_open(port);
            }
            ggml_backend_reg_t reg = up ? add_server(ep.c_str()) : nullptr;
            if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
                printf("RPC: FAIL: could not connect to %s\n", ep.c_str());
                fails++;
            } else {
                ggml_backend_dev_t rdev = ggml_backend_reg_dev_get(reg, 0);
                const bool r_kq = refuses(rdev, GGML_OP_KQ_MASK, 0), r_mc = refuses(rdev, GGML_OP_MOE_CACHE_LOOKUP, 0);
                const bool r_gdn_replay = refuses(rdev, GGML_OP_GATED_DELTA_NET, 1), r_gdn = refuses(rdev, GGML_OP_GATED_DELTA_NET, 0);
                printf("RPC supports_op refuses: KQ_MASK %d, MOE_CACHE_LOOKUP %d, GATED_DELTA_NET replay %d (plain %d)\n", r_kq, r_mc,
                        r_gdn_replay, r_gdn);
                fails += !r_kq + !r_mc + !r_gdn_replay + r_gdn;
                ggml_backend_ptr be(ggml_backend_dev_init(rdev, nullptr));
                fails += run_graph("RPC", be.get(), cpu.get());
            }
        }
#endif
    }

    printf("%s\n", fails == 0 ? "OK" : "FAIL");
    fflush(stdout);
    std::_Exit(fails == 0 ? 0 : 1);
}
