// round trip of the asynchronous host <-> device copies on the Metal devices: ggml_backend_tensor_set_async from host pointers at
// page and non-page offsets, sizes from 4 bytes to 8 MiB (the zero-copy upload of 64 KiB or more and the copied one below it),
// then ggml_backend_tensor_get_async into non-page-aligned host memory; every byte must come back. The source is overwritten only
// after the synchronize. SKIP without a Metal device; exit 1 on any difference.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main() {
    ggml_backend_load_all();

    const size_t n_bytes = 32u << 20;
    int n_cases = 0;
    int n_diff  = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (strncmp(ggml_backend_dev_name(dev), "MTL", 3) != 0) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) {
            continue;
        }
        ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) n_bytes);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        std::mt19937 rng(42);
        std::vector<uint8_t> host(n_bytes + 2*16384);
        std::vector<uint8_t> back(n_bytes + 2*16384);
        for (size_t size : { (size_t) 4, (size_t) 100, (size_t) 4096, (size_t) 65535, (size_t) 65536, (size_t) 65536 + 7,
                (size_t) 1 << 20, ((size_t) 8 << 20) + 12 }) {
            for (size_t src_off : { (size_t) 0, (size_t) 4, (size_t) 100, (size_t) 4097, (size_t) 16384 }) {
                for (size_t dst_off : { (size_t) 0, (size_t) 12, (size_t) 16384 + 4 }) {
                    if (dst_off + size > n_bytes) {
                        continue;
                    }
                    std::vector<uint8_t> keep(size);
                    for (size_t i = 0; i < size; ++i) {
                        keep[i] = host[src_off + i] = (uint8_t) rng();
                    }
                    ggml_backend_tensor_set_async(be, t, host.data() + src_off, dst_off, size);
                    ggml_backend_synchronize(be);
                    // the source may change after the synchronize
                    memset(host.data() + src_off, 0xA5, size);
                    const size_t back_off = 3 + (src_off % 7);
                    memset(back.data(), 0, size + back_off + 16);
                    ggml_backend_tensor_get_async(be, t, back.data() + back_off, dst_off, size);
                    ggml_backend_synchronize(be);
                    bool ok = memcmp(keep.data(), back.data() + back_off, size) == 0;
                    // the async get wrote nothing outside its range
                    ok = ok && back[back_off - 1] == 0 && back[back_off + size] == 0;
                    n_cases++;
                    if (!ok) {
                        n_diff++;
                        printf("  DIFFER size %zu src_off %zu dst_off %zu\n", size, src_off, dst_off);
                    }
                }
            }
        }
        // the uploaded bytes themselves: one more upload checked against a kept copy
        {
            const size_t size = ((size_t) 8 << 20) + 12;
            std::vector<uint8_t> keep(size);
            for (size_t i = 0; i < size; ++i) {
                keep[i] = host[100 + i] = (uint8_t) rng();
            }
            ggml_backend_tensor_set_async(be, t, host.data() + 100, 12, size);
            ggml_backend_synchronize(be);
            memset(host.data() + 100, 0, size);
            std::vector<uint8_t> got(size);
            ggml_backend_tensor_get(t, got.data(), 12, size);
            const bool ok = memcmp(keep.data(), got.data(), size) == 0;
            n_cases++;
            if (!ok) {
                n_diff++;
                printf("  DIFFER upload content\n");
            }
            for (size_t size2 : { (size_t) 100, (size_t) 65536 + 7, (size_t) 1 << 20 }) {
                std::vector<uint8_t> keep2(size2);
                for (size_t i = 0; i < size2; ++i) {
                    keep2[i] = host[4097 + i] = (uint8_t) rng();
                }
                ggml_backend_tensor_set_async(be, t, host.data() + 4097, 16384 + 4, size2);
                ggml_backend_synchronize(be);
                std::vector<uint8_t> got2(size2);
                ggml_backend_tensor_get(t, got2.data(), 16384 + 4, size2);
                n_cases++;
                if (memcmp(keep2.data(), got2.data(), size2) != 0) {
                    n_diff++;
                    printf("  DIFFER upload content size %zu\n", size2);
                }
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }
    if (n_cases == 0) {
        printf("test-metal-async-copy: SKIP (no Metal device)\n");
        return 0;
    }
    printf("test-metal-async-copy: %d cases, %d differ\n", n_cases, n_diff);
    printf("%s\n", n_diff == 0 ? "OK" : "FAIL");
    return n_diff == 0 ? 0 : 1;
}
