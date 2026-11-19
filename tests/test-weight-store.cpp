// llama_weight_store: extents, ring reads ahead (plan), demand reads, short reads / EIO / O_DIRECT EINVAL fallbacks, VERIFY
// usage: test-weight-store [scratch file path]

#include "../src/llama-weight-store.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
int main() {
    printf("test-weight-store: skipped on Windows\n");
    return 0;
}
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static int n_fail = 0;

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

struct fault {
    std::atomic<int> mode { 0 }; // 0 none, 1 short reads, 2 EIO on the first calls, 3 EINVAL on direct reads, 4 corrupt one byte, 5 EIO always
    std::atomic<int> calls { 0 };
};

static int64_t read_hook(int fd, void * buf, size_t n, uint64_t off, bool direct, void * ud) {
    fault * f = (fault *) ud;
    const int c = f->calls.fetch_add(1);
    switch (f->mode.load()) {
        case 1:
            // short: direct reads stop at a 4 KiB boundary, buffered ones after 1000 bytes
            if (direct) {
                n = std::min<size_t>(n, 8192);
            } else {
                n = std::min<size_t>(n, 1000);
            }
            break;
        case 2:
            if (c % 3 == 0) {
                return -EIO;
            }
            break;
        case 3:
            if (direct) {
                return -EINVAL;
            }
            break;
        case 5:
            return -EIO;
        default:
            break;
    }
    const ssize_t r = pread(fd, buf, n, (off_t) off);
    if (r < 0) {
        return -errno;
    }
    if (f->mode.load() == 4 && r > 0 && c == 1) {
        ((uint8_t *) buf)[r/2] ^= 0x5a;
    }
    return r;
}

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "test-weight-store.bin";
    const size_t file_size = (size_t) 48 << 20;

    // random file
    std::vector<uint8_t> data(file_size);
    std::mt19937_64 rng(1234);
    for (size_t i = 0; i < file_size; i += 8) {
        const uint64_t v = rng();
        memcpy(data.data() + i, &v, std::min<size_t>(8, file_size - i));
    }
    {
        FILE * f = fopen(path.c_str(), "wb");
        if (!f || fwrite(data.data(), 1, file_size, f) != file_size) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            return 1;
        }
        fclose(f);
    }
    const int fd = open(path.c_str(), O_RDONLY);
    void * map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t src_buf = ggml_backend_cpu_buffer_from_ptr(map, file_size);

    // source tensors at unaligned offsets, one larger than the ring
    struct tdesc { size_t off; size_t size; };
    const std::vector<tdesc> descs = {
        { 4096*3 + 96, (size_t) 3 << 20 },             // A
        { (size_t) (4 << 20) + 32, (size_t) 5 << 20 },  // B
        { (size_t) (10 << 20), (size_t) 1 << 20 },      // C
        { (size_t) (12 << 20) + 160, (size_t) 20 << 20 }, // D: larger than the ring
        { (size_t) (33 << 20), 4096 + 7 },              // E: small
    };
    ggml_init_params ip = { /*.mem_size =*/ 64*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx_src = ggml_init(ip);
    ggml_context * ctx_dst = ggml_init(ip);
    std::vector<ggml_tensor *> src, dst;
    for (const auto & d : descs) {
        ggml_tensor * s = ggml_new_tensor_1d(ctx_src, GGML_TYPE_I8, (int64_t) d.size);
        ggml_backend_tensor_alloc(src_buf, s, (uint8_t *) map + d.off);
        src.push_back(s);
        dst.push_back(ggml_new_tensor_1d(ctx_dst, GGML_TYPE_I8, (int64_t) d.size));
    }
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors(ctx_dst, backend);

    auto ext = std::make_shared<llama_weight_extents>();
    {
        llama_weight_file f;
        f.path     = path;
        f.map_addr = (const uint8_t *) map;
        f.map_size = file_size;
        f.size     = file_size;
        ext->files.push_back(f);
        for (size_t i = 0; i < descs.size(); ++i) {
            llama_weight_extent e;
            e.file = 0;
            e.offs = descs[i].off;
            e.size = descs[i].size;
            ext->tensors.emplace(src[i], e);
        }
    }

    // extents
    {
        uint16_t file = 9;
        uint64_t offs = 0;
        CHECK(ext->resolve(src[1]->data, 100, file, offs) && file == 0 && offs == descs[1].off);
        CHECK(!ext->resolve(data.data(), 100, file, offs));
        CHECK(ext->find(src[3], &offs) != nullptr && offs == descs[3].off);
        CHECK(ext->bytes_streamable() == ((3 + 5 + 1 + 20) << 20) + 4096 + 7);
    }

    auto clear_dst = [&]() { ggml_backend_buffer_clear(dst_buf, 0); };
    auto check_range = [&](int t, size_t off, size_t size, const char * what) {
        std::vector<uint8_t> got(size);
        ggml_backend_tensor_get(dst[t], got.data(), off, size);
        if (memcmp(got.data(), data.data() + descs[t].off + off, size) != 0) {
            fprintf(stderr, "%s: tensor %d [%zu, %zu) differs\n", what, t, off, off + size);
            n_fail++;
        }
    };

    for (int io : { LLAMA_WEIGHT_STREAM_IO_BUFFERED, LLAMA_WEIGHT_STREAM_IO_DIRECT, LLAMA_WEIGHT_STREAM_IO_AUTO }) {
        for (int fmode : { 0, 1, 2, 3, 4 }) {
            size_t n_dropped = 0; // pages the drop below really evicted (a kernel may keep a large folio that spans the range)
            if (io == LLAMA_WEIGHT_STREAM_IO_AUTO) {
                // auto copies the resident pages from the mapping and reads the others unbuffered: drop every other 1 MiB of the file
                fsync(fd);
                for (size_t o = 0; o < file_size; o += (size_t) 2 << 20) {
                    madvise((uint8_t *) map + o, (size_t) 1 << 20, MADV_DONTNEED); // mapped pages are not dropped from the page cache
#ifdef POSIX_FADV_DONTNEED
                    posix_fadvise(fd, (off_t) o, (off_t) 1 << 20, POSIX_FADV_DONTNEED);
#endif
                }
                const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
                std::vector<unsigned char> res((file_size + pg - 1) / pg);
#if defined(__APPLE__)
                if (mincore(map, file_size, (char *) res.data()) == 0) { // macOS: char * vector
#else
                if (mincore(map, file_size, res.data()) == 0) {
#endif
                    for (unsigned char r : res) {
                        n_dropped += (r & 1) == 0;
                    }
                }
            }
            fault flt;
            flt.mode = fmode;
            llama_weight_store_params p;
            p.io        = io;
            p.ring_size = (size_t) 8 << 20;   // 6 MiB ring + 2 MiB demand with 1 MiB reads
            p.n_threads = 3;
            p.chunk     = (size_t) 1 << 20;
            p.verify    = fmode == 4;
            p.ram_avail = (size_t) 1 << 30;
            llama_weight_store ws(ext, p);
            if (fmode != 0) {
                ws.set_read_fn(read_hook, &flt);
            }
            char what[64];
            snprintf(what, sizeof(what), "io %d fault %d", io, fmode);

            CHECK(ws.covers(src[0]));

            // planned: two graphs with the same plan (the second is read ahead), then another plan
            const ggml_tensor * plan[] = { src[0], src[1], src[2], src[3], src[4] };
            for (int g = 0; g < 3; ++g) {
                clear_dst();
                ws.plan(plan, 5);
                CHECK(ws.upload(backend, src[0], dst[0], 0, 0, ggml_nbytes(src[0])));
                // B in runs with gaps (like the routed experts), C skipped
                CHECK(ws.upload(backend, src[1], dst[1], 1000, 1000, 70000));
                CHECK(ws.upload(backend, src[1], dst[1], 2 << 20, 2 << 20, (size_t) (1 << 20) + 512));
                CHECK(ws.upload(backend, src[1], dst[1], (4 << 20) + 5, (4 << 20) + 5, (1 << 20) - 5));
                CHECK(ws.upload(backend, src[3], dst[3], 0, 0, ggml_nbytes(src[3])));
                CHECK(ws.upload(backend, src[4], dst[4], 0, 0, ggml_nbytes(src[4])));
                ggml_backend_synchronize(backend);
                check_range(0, 0, ggml_nbytes(src[0]), what);
                check_range(1, 1000, 70000, what);
                check_range(1, 2 << 20, (1 << 20) + 512, what);
                check_range(1, (4 << 20) + 5, (1 << 20) - 5, what);
                check_range(3, 0, ggml_nbytes(src[3]), what);
                check_range(4, 0, ggml_nbytes(src[4]), what);
            }
            // a src not in the plan: demand read
            {
                const ggml_tensor * plan2[] = { src[3] };
                ws.plan(plan2, 1);
                clear_dst();
                CHECK(ws.upload(backend, src[2], dst[2], 0, 0, ggml_nbytes(src[2])));
                ggml_backend_synchronize(backend);
                check_range(2, 0, ggml_nbytes(src[2]), what);
                ws.plan(nullptr, 0);
            }
            // demand batch over several tensors, one region larger than a half of the demand area
            {
                clear_dst();
                std::vector<llama_weight_store::region> r = {
                    { src[0], 7, dst[0], 7, 12345 },
                    { src[1], 3 << 20, dst[1], 3 << 20, (size_t) (2 << 20) - 100 },
                    { src[3], 1 << 20, dst[3], 1 << 20, (size_t) 5 << 20 },
                    { src[4], 0, dst[4], 0, ggml_nbytes(src[4]) },
                };
                CHECK(ws.upload_batch(backend, r.data(), r.size()));
                ggml_backend_synchronize(backend);
                check_range(0, 7, 12345, what);
                check_range(1, 3 << 20, (2 << 20) - 100, what);
                check_range(3, 1 << 20, 5 << 20, what);
                check_range(4, 0, ggml_nbytes(src[4]), what);
            }
            const auto st = ws.get_stats();
            CHECK(st.plan_reads > 0 && st.plan_uploaded > 0);
            CHECK(st.demand_bytes > 0);
            if (fmode == 1) {
                CHECK(st.short_reads > 0);
            }
            if (fmode == 2) {
                CHECK(st.io_errors > 0);
            }
            if (fmode == 3 && io == LLAMA_WEIGHT_STREAM_IO_DIRECT) {
                CHECK(st.direct_disabled == 1);
            }
            if (fmode == 4) {
                CHECK(st.verify_mismatch == 1);
                CHECK(st.verify_regions > 0);
            }
            if (fmode == 0 && io == LLAMA_WEIGHT_STREAM_IO_AUTO) {
                CHECK(st.cached_bytes > 0);
                if (n_dropped > 0) {
                    CHECK(st.direct_bytes > 0 || st.direct_disabled > 0);
                } else {
                    printf("%s: the page cache kept every page of the file (nothing to read unbuffered), direct check skipped\n", what);
                }
                printf("%s: cached %.1f MiB, direct %.1f MiB, buffered %.1f MiB\n", what, st.cached_bytes/1048576.0, st.direct_bytes/1048576.0, st.buffered_bytes/1048576.0);
            }
            if (fmode == 0 && io == LLAMA_WEIGHT_STREAM_IO_DIRECT) {
                // the test file system may not support O_DIRECT, then everything is buffered
                printf("%s: direct %.1f MiB, buffered %.1f MiB\n", what, st.direct_bytes/1048576.0, st.buffered_bytes/1048576.0);
            }
            printf("%s: ring %.1f MiB (uploaded %.1f, discarded %.1f), demand %.1f MiB, errors %llu, short %llu, fallback %.1f MiB\n", what,
                    st.plan_bytes/1048576.0, st.plan_uploaded/1048576.0, st.plan_discarded/1048576.0, st.demand_bytes/1048576.0,
                    (unsigned long long) st.io_errors, (unsigned long long) st.short_reads, st.fallback_bytes/1048576.0);
        }
    }

    // the ring is released after graphs without prefetched copies and allocated again for the next prompt; no reads ahead without wrap
    {
        llama_weight_store_params p;
        p.io        = LLAMA_WEIGHT_STREAM_IO_BUFFERED;
        p.ring_size = (size_t) 8 << 20;
        p.chunk     = (size_t) 1 << 20;
        llama_weight_store ws(ext, p);
        const ggml_tensor * plan[] = { src[0], src[1] };
        for (int round = 0; round < 3; ++round) {
            clear_dst();
            ws.set_wrap(round != 1);
            ws.plan(plan, 2);
            CHECK(ws.upload(backend, src[0], dst[0], 0, 0, ggml_nbytes(src[0])));
            CHECK(ws.upload(backend, src[1], dst[1], 0, 0, ggml_nbytes(src[1])));
            ggml_backend_synchronize(backend);
            check_range(0, 0, ggml_nbytes(src[0]), "ring release");
            check_range(1, 0, ggml_nbytes(src[1]), "ring release");
            // generation: the reads ahead finish or are dropped, then the ring goes
            for (int i = 0; i < 200 && ws.pinned_size() > ((size_t) 3 << 20); ++i) {
                ws.plan(nullptr, 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            CHECK(ws.pinned_size() < ((size_t) 3 << 20)); // only the demand area is left
        }
        CHECK(ws.ring_allocs() == 3);
        const auto st = ws.get_stats();
        printf("ring release: allocs %llu, ring %.1f MiB, discarded %.1f MiB\n", (unsigned long long) ws.ring_allocs(), st.plan_bytes/1048576.0, st.plan_discarded/1048576.0);
    }

    // reads that always fail: the bytes come from the mapping
    {
        fault flt;
        flt.mode = 5;
        llama_weight_store_params p;
        p.io        = LLAMA_WEIGHT_STREAM_IO_BUFFERED;
        p.ring_size = (size_t) 8 << 20;
        p.chunk     = (size_t) 1 << 20;
        p.wrap      = false; // no reads ahead of a next graph: every read is of the upload below
        llama_weight_store ws(ext, p);
        ws.set_read_fn(read_hook, &flt);
        clear_dst();
        const ggml_tensor * plan[] = { src[1] };
        ws.plan(plan, 1);
        CHECK(ws.upload(backend, src[1], dst[1], 0, 0, ggml_nbytes(src[1])));
        ggml_backend_synchronize(backend);
        check_range(1, 0, ggml_nbytes(src[1]), "EIO always");
        CHECK(ws.get_stats().fallback_bytes == ggml_nbytes(src[1]));
    }

    // a tensor that is not mapped is not covered
    {
        llama_weight_store_params p;
        llama_weight_store ws(ext, p);
        CHECK(!ws.covers(dst[0]));
        CHECK(!ws.upload(backend, dst[0], dst[1], 0, 0, 100));
    }

    // enabled(): modes and the kill switch
    unsetenv("LLAMA_WEIGHT_STREAM");
    CHECK(!llama_weight_store::enabled(LLAMA_WEIGHT_STREAM_OFF, 100, 10));
    CHECK(llama_weight_store::enabled(LLAMA_WEIGHT_STREAM_ON, 100, 1000));
    CHECK(llama_weight_store::enabled(LLAMA_WEIGHT_STREAM_AUTO, 100, 100));
    CHECK(!llama_weight_store::enabled(LLAMA_WEIGHT_STREAM_AUTO, 100, 1000));
    setenv("LLAMA_WEIGHT_STREAM", "0", 1);
    CHECK(!llama_weight_store::enabled(LLAMA_WEIGHT_STREAM_ON, 100, 10));
    setenv("LLAMA_WEIGHT_STREAM", "1", 1);
    CHECK(llama_weight_store::enabled(LLAMA_WEIGHT_STREAM_OFF, 100, 1000));
    unsetenv("LLAMA_WEIGHT_STREAM");

    ggml_backend_buffer_free(dst_buf);
    ggml_backend_buffer_free(src_buf);
    ggml_free(ctx_src);
    ggml_free(ctx_dst);
    ggml_backend_free(backend);
    munmap(map, file_size);
    close(fd);
    remove(path.c_str());

    printf("test-weight-store: %s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
#endif
