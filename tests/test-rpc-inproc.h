#pragma once

// In-process RPC servers for tests (not on Windows: BSD sockets). ggml_backend_rpc_start_server serves forever and has no stop call, so
// a test that starts one must end with std::_Exit once its checks are done: returning from main runs the static destructors of the
// backends while the server thread can still be computing a graph (a SIGSEGV after the test printed its result). The port is one the
// kernel picks (bind to port 0), not one derived from the pid: a fixed or pid-derived port can be held by another socket on a busy host,
// and the server's bind then fails ("Failed to create server socket"). A server whose bind still fails is started again on a new port.

#ifndef _WIN32

#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

static bool rpc_test_port_open(int port) {
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

// a loopback port no socket is bound to right now (0 on failure)
static int rpc_test_free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return 0;
    }
    sockaddr_in a = {};
    a.sin_family      = AF_INET;
    a.sin_port        = 0;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(a);
    int port = 0;
    if (bind(fd, (const sockaddr *) &a, sizeof(a)) == 0 && getsockname(fd, (sockaddr *) &a, &len) == 0) {
        port = ntohs(a.sin_port);
    }
    close(fd);
    return port;
}

typedef void (*rpc_test_start_server_t)(const char *, const char *, size_t, size_t, ggml_backend_dev_t *);

// starts a server over dev on a free loopback port and waits until it listens; endpoint "127.0.0.1:<port>" on success
static bool rpc_test_start_server(rpc_test_start_server_t start_server, ggml_backend_dev_t dev, std::string & endpoint) {
    for (int attempt = 0; attempt < 8; ++attempt) {
        const int port = rpc_test_free_port();
        if (port == 0) {
            continue;
        }
        const std::string ep = "127.0.0.1:" + std::to_string(port);
        // set when start_server returns, which it does only when it could not set up (bind / listen failed)
        auto ended = std::make_shared<std::atomic<bool>>(false);
        std::thread([start_server, ep, dev, ended]() {
            ggml_backend_dev_t sdevs[1] = { dev };
            start_server(ep.c_str(), nullptr, 2, 1, sdevs);
            ended->store(true);
        }).detach();
        for (int i = 0; i < 200 && !ended->load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (rpc_test_port_open(port) && !ended->load()) {
                endpoint = ep;
                return true;
            }
        }
    }
    return false;
}

#endif // _WIN32
