// llama_sysmem::available with synthetic /proc/meminfo, /proc/self/cgroup and cgroup files (src/llama-sysmem.h)
#include "../src/llama-sysmem.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static int n_fail = 0;

static void put(const fs::path & p, const std::string & s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << s;
}

static void check(const char * name, uint64_t got, uint64_t want) {
    const bool ok = got == want;
    printf("%-58s got %12llu MiB want %12llu MiB %s\n", name, (unsigned long long) (got >> 20), (unsigned long long) (want >> 20), ok ? "OK" : "FAIL");
    n_fail += !ok;
}

int main() {
    constexpr uint64_t MiB = 1024*1024;
    const fs::path root = fs::temp_directory_path() / ("test-sysmem-" + std::to_string(getpid()));
    fs::remove_all(root);
    const std::string mi = (root / "meminfo").string();
    const std::string cg = (root / "cgroup").string();
    const std::string sys = (root / "sys").string();

    // the T4 case: a 31 GiB cgroup limit, 30 GiB of it page cache, MemAvailable 30455 MiB
    put(mi, "MemTotal:       32864756 kB\nMemFree:          500000 kB\nMemAvailable:   31185920 kB\n");
    put(cg, "0::/kaggle\n");
    put(root / "sys/kaggle/memory.max", std::to_string(31*1024*MiB) + "\n");
    put(root / "sys/kaggle/memory.current", std::to_string(31*1024*MiB - 1101*MiB) + "\n");
    put(root / "sys/kaggle/memory.stat", "anon 1048576000\nfile 30000000000\nactive_file " + std::to_string(10240*MiB) + "\ninactive_file " +
        std::to_string(18000*MiB) + "\nslab_reclaimable " + std::to_string(500*MiB) + "\nshmem 0\n");
    // the root has no limit (memory.max is "max")
    put(root / "sys/memory.max", "max\n");
    put(root / "sys/memory.current", "999999999999\n");
    // used = (31744 - 1101) - (10240 + 18000 + 500) = 1903 MiB -> 29841 MiB left under the limit, below MemAvailable (30455 MiB)
    check("v2, page cache not counted as used", llama_sysmem::available(mi, cg, sys), (31744 - 1903)*MiB);
    check("v2, cgroup part alone", llama_sysmem::cgroup_available(cg, sys), (31744 - 1903)*MiB);

    // MemAvailable is lower than the cgroup room: it wins
    put(mi, "MemAvailable:   10240000 kB\n");
    check("v2, MemAvailable below the cgroup room", llama_sysmem::available(mi, cg, sys), 10240000ull*1024);

    // anonymous memory and shmem are used: a cgroup full of anon memory has nothing left
    put(mi, "MemAvailable:   31185920 kB\n");
    put(root / "sys/kaggle/memory.current", std::to_string(31*1024*MiB) + "\n");
    put(root / "sys/kaggle/memory.stat", "anon 33285996544\nactive_file 0\ninactive_file 0\nshmem 0\n");
    check("v2, all anon: 1 byte (0 means unknown)", llama_sysmem::available(mi, cg, sys), 1);

    // nested v2: the parent's limit is tighter than the child's
    put(cg, "0::/a/b\n");
    put(root / "sys/a/b/memory.max", std::to_string(64*1024*MiB) + "\n");
    put(root / "sys/a/b/memory.current", std::to_string(8*1024*MiB) + "\n");
    put(root / "sys/a/b/memory.stat", "active_file " + std::to_string(4*1024*MiB) + "\ninactive_file 0\n");
    put(root / "sys/a/memory.max", std::to_string(16*1024*MiB) + "\n");
    put(root / "sys/a/memory.current", std::to_string(12*1024*MiB) + "\n");
    put(root / "sys/a/memory.stat", "active_file " + std::to_string(2*1024*MiB) + "\ninactive_file " + std::to_string(2*1024*MiB) + "\n");
    // child: 64 - (8 - 4) = 60 GiB, parent: 16 - (12 - 4) = 8 GiB
    check("v2 nested, the parent's limit applies", llama_sysmem::cgroup_available(cg, sys), 8*1024*MiB);

    // v1: hierarchical total_* counters win over the local ones
    put(cg, "12:memory:/docker/x\n3:cpu,cpuacct:/docker/x\n");
    put(root / "sys/memory/docker/x/memory.limit_in_bytes", std::to_string(20*1024*MiB) + "\n");
    put(root / "sys/memory/docker/x/memory.usage_in_bytes", std::to_string(19*1024*MiB) + "\n");
    put(root / "sys/memory/docker/x/memory.stat", "active_file 1\ninactive_file 1\ntotal_active_file " + std::to_string(9*1024*MiB) +
        "\ntotal_inactive_file " + std::to_string(6*1024*MiB) + "\n");
    put(root / "sys/memory/memory.limit_in_bytes", "9223372036854771712\n");
    put(root / "sys/memory/memory.usage_in_bytes", "1\n");
    check("v1, total_*_file not counted as used", llama_sysmem::cgroup_available(cg, sys), (20 - 4)*1024*MiB);

    // v1 without total_* (older kernels): the local counters
    put(root / "sys/memory/docker/x/memory.stat", "active_file " + std::to_string(3*1024*MiB) + "\ninactive_file " + std::to_string(1*1024*MiB) + "\n");
    check("v1, local *_file counters", llama_sysmem::cgroup_available(cg, sys), (20 - 15)*1024*MiB);

    // a cgroup without a memory.stat: the whole usage counts (the old rule)
    fs::remove(root / "sys/memory/docker/x/memory.stat");
    check("v1, no memory.stat: usage counts", llama_sysmem::cgroup_available(cg, sys), 1*1024*MiB);

    // no cgroup file, no limit
    check("no /proc/self/cgroup: no limit", llama_sysmem::cgroup_available((root / "none").string(), sys), UINT64_MAX);
    check("no meminfo: unknown", llama_sysmem::available((root / "none").string(), cg, sys), 0);

    // one cgroup v2 directory (the weight store's namespace-root read); the numbers of the 8 GiB scope probes
    // (tests/k19_ram_avail/probe.cpp, MiB): mlock 4 GiB of the model's mapping + 1 GiB anon, memfd (shmem) 3 GiB + 1 GiB anon, mmap 4 GiB clean
    {
        const fs::path d = root / "ns";
        put(d / "memory.max", std::to_string(8192*MiB) + "\n");
        put(d / "memory.current", std::to_string(5123*MiB) + "\n");
        put(d / "memory.stat", "anon " + std::to_string(1024*MiB) + "\nfile " + std::to_string(4096*MiB) +
            "\nactive_file 0\ninactive_file 0\nfile_mapped " + std::to_string(4096*MiB) + "\nunevictable " + std::to_string(4096*MiB) +
            "\nslab_reclaimable 0\nshmem 0\n");
        check("v2 dir, mlocked file pages are used", llama_sysmem::cgroup_v2_dir_available(d.string(), false), (8192 - 5123)*MiB);
        check("v2 dir, LLAMA_SYSMEM_CREDIT_FILE=1 (old)", llama_sysmem::cgroup_v2_dir_available(d.string(), true), (8192 - 1027)*MiB);
        put(d / "memory.current", std::to_string(4106*MiB) + "\n");
        put(d / "memory.stat", "anon " + std::to_string(1024*MiB) + "\nfile " + std::to_string(3072*MiB) +
            "\nactive_file 0\ninactive_file 0\nshmem " + std::to_string(3072*MiB) + "\nslab_reclaimable " + std::to_string(7*MiB) + "\n");
        check("v2 dir, shmem (memfd / tmpfs) is used", llama_sysmem::cgroup_v2_dir_available(d.string(), false), (8192 - 4106 + 7)*MiB);
        // clean page cache, mapped (active) or not: reclaimable, credited exactly as before (unchanged reading)
        put(d / "memory.current", std::to_string(5123*MiB) + "\n");
        put(d / "memory.stat", "anon " + std::to_string(1024*MiB) + "\nfile " + std::to_string(4096*MiB) + "\nactive_file " +
            std::to_string(4096*MiB) + "\ninactive_file 0\nfile_mapped " + std::to_string(4096*MiB) + "\nslab_reclaimable 0\nshmem 0\n");
        check("v2 dir, clean mapped page cache is reclaimable", llama_sysmem::cgroup_v2_dir_available(d.string(), false), (8192 - 1027)*MiB);
        check("v2 dir, same as the old reading there", llama_sysmem::cgroup_v2_dir_available(d.string(), true), (8192 - 1027)*MiB);
        put(d / "memory.max", "max\n");
        check("v2 dir, memory.max \"max\": no limit", llama_sysmem::cgroup_v2_dir_available(d.string(), false), UINT64_MAX);
        check("v2 dir, missing: no limit", llama_sysmem::cgroup_v2_dir_available((root / "nodir").string(), false), UINT64_MAX);
    }

    // the live system gives a value
    printf("live: %llu MiB\n", (unsigned long long) (llama_sysmem::available() >> 20));

    // the reclaimable figure (MemAvailable minus the registered model pages; 1 when nothing is left, 0 when unknown)
    {
        constexpr uint64_t GiB = 1024*MiB;
        check("reclaimable: nothing pinned", llama_sysmem::reclaimable(79*GiB, 0), 79*GiB);
        check("reclaimable: A100 fn4nd core (48.5 GiB registered)", llama_sysmem::reclaimable(79*GiB, (int64_t) (48.5*GiB)), (uint64_t) (30.5*GiB));
        check("reclaimable: + the 26.8 GiB pin", llama_sysmem::reclaimable(79*GiB, (int64_t) (75.3*GiB)), (uint64_t) (79*GiB - (uint64_t) (75.3*GiB)));
        check("reclaimable: more pinned than available -> 1", llama_sysmem::reclaimable(10*GiB, (int64_t) (20*GiB)), 1);
        check("reclaimable: unknown stays unknown", llama_sysmem::reclaimable(0, (int64_t) (20*GiB)), 0);
        const int64_t p0 = llama_sysmem::pinned_file_bytes().load();
        llama_sysmem::pinned_file_add((int64_t) (3*GiB));
        check("pinned_file_add +3 GiB", (uint64_t) (llama_sysmem::pinned_file_bytes().load() - p0), 3*GiB);
        llama_sysmem::pinned_file_add(-(int64_t) (3*GiB));
        check("pinned_file_add -3 GiB", (uint64_t) (llama_sysmem::pinned_file_bytes().load() - p0), 0);
    }

    fs::remove_all(root);
    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
