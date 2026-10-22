#pragma once

// system memory that can be used without swapping (header only, also used by tests/test-sysmem.cpp with synthetic files)
//
// Linux: MemAvailable of /proc/meminfo, capped by the memory left under the cgroup limits of this process and of its parent
// cgroups (v2 and v1); a cgroup's usage includes its page cache, which the kernel reclaims before it fails an allocation, so the
// reclaimable file pages (active_file + inactive_file, and slab_reclaimable on v2) are not counted as used, like MemAvailable does
// for the host (a container with a warm page cache read 1101 MiB available here while MemAvailable was 30455 MiB)

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif

namespace llama_sysmem {

// first number in a file, false if it is missing or not a number (e.g. "max")
inline bool read_u64(const std::string & path, uint64_t & res) {
    std::ifstream f(path);
    return (bool) (f >> res);
}

// sum of the values of the given keys in a "key value" file (memory.stat), the first key list that is present wins
inline uint64_t stat_sum(const std::string & path, const char * const * keys_a, const char * const * keys_b) {
    std::ifstream f(path);
    std::string key;
    uint64_t value = 0;
    uint64_t sum_a = 0;
    uint64_t sum_b = 0;
    bool     have_a = false;
    while (f >> key >> value) {
        for (const char * const * k = keys_a; *k != nullptr; ++k) {
            if (key == *k) {
                sum_a += value;
                have_a = true;
            }
        }
        for (const char * const * k = keys_b; keys_b != nullptr && *k != nullptr; ++k) {
            if (key == *k) {
                sum_b += value;
            }
        }
    }
    return have_a ? sum_a : sum_b;
}

// memory left under the cgroup limits of the cgroups listed in proc_cgroup (the /proc/self/cgroup format) and of their parents,
// with the cgroup file systems under sys_root ("/sys/fs/cgroup"), UINT64_MAX if there is no limit
inline uint64_t cgroup_available(const std::string & proc_cgroup, const std::string & sys_root) {
    uint64_t res = UINT64_MAX;
    std::ifstream f(proc_cgroup);
    std::string line;
    while (std::getline(f, line)) {
        // "0::<path>" (v2) or "<id>:<controllers>:<path>" (v1)
        const size_t a = line.find(':');
        const size_t b = a == std::string::npos ? a : line.find(':', a + 1);
        if (b == std::string::npos) {
            continue;
        }
        const std::string ctrl = line.substr(a + 1, b - a - 1);
        std::string path = line.substr(b + 1);
        std::string root;
        std::string f_max;
        std::string f_cur;
        bool v2 = false;
        if (line.compare(0, a, "0") == 0 && ctrl.empty()) {
            root = sys_root; f_max = "/memory.max"; f_cur = "/memory.current"; v2 = true;
        } else if (("," + ctrl + ",").find(",memory,") != std::string::npos) {
            root = sys_root + "/memory"; f_max = "/memory.limit_in_bytes"; f_cur = "/memory.usage_in_bytes";
        } else {
            continue;
        }
        // the limit of every ancestor applies too
        while (true) {
            const std::string dir = root + (path == "/" ? "" : path);
            uint64_t lim = 0;
            uint64_t cur = 0;
            if (read_u64(dir + f_max, lim) && lim < (UINT64_C(1) << 60) && read_u64(dir + f_cur, cur)) {
                static const char * const keys_v2[]       = { "active_file", "inactive_file", "slab_reclaimable", nullptr };
                static const char * const keys_v1_total[] = { "total_active_file", "total_inactive_file", nullptr };
                static const char * const keys_v1[]       = { "active_file", "inactive_file", nullptr };
                const uint64_t reclaimable = v2 ? stat_sum(dir + "/memory.stat", keys_v2, nullptr)
                                                : stat_sum(dir + "/memory.stat", keys_v1_total, keys_v1);
                const uint64_t used = cur - std::min(cur, reclaimable);
                res = std::min(res, lim > used ? lim - used : 0);
            }
            if (path.empty() || path == "/") {
                break;
            }
            const size_t p = path.find_last_of('/');
            path = p == 0 || p == std::string::npos ? "/" : path.substr(0, p);
        }
    }
    return res;
}

// MemAvailable of a /proc/meminfo file in bytes, 0 if it is missing
inline uint64_t meminfo_available(const std::string & meminfo) {
    std::ifstream f(meminfo);
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (f >> key >> value) {
        std::getline(f, unit);
        if (key == "MemAvailable:") {
            return value*1024;
        }
    }
    return 0;
}

// MemTotal of a /proc/meminfo file in bytes, 0 if it is missing
inline uint64_t meminfo_total(const std::string & meminfo) {
    std::ifstream f(meminfo);
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (f >> key >> value) {
        std::getline(f, unit);
        if (key == "MemTotal:") {
            return value*1024;
        }
    }
    return 0;
}

// MemTotal of the host (Linux), 0 if unknown
inline uint64_t total() {
#if defined(__linux__)
    return meminfo_total("/proc/meminfo");
#else
    return 0;
#endif
}

// see the top of the file; 0 if unknown
inline uint64_t available(const std::string & meminfo, const std::string & proc_cgroup, const std::string & sys_root) {
    const uint64_t m = meminfo_available(meminfo);
    if (m == 0) {
        return 0;
    }
    return std::max<uint64_t>(1, std::min(m, cgroup_available(proc_cgroup, sys_root))); // 0 means unknown
}

inline uint64_t available() {
#if defined(__linux__)
    return available("/proc/meminfo", "/proc/self/cgroup", "/sys/fs/cgroup");
#elif defined(__APPLE__) && defined(__MACH__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) != KERN_SUCCESS) {
        return 0;
    }
    return ((uint64_t) vm.free_count + vm.inactive_count + vm.purgeable_count)*(uint64_t) vm_page_size;
#else
    return 0;
#endif
}

// CPUs a cgroup quota allows (cpu.max "quota period" on v2, cpu.cfs_quota_us / cpu.cfs_period_us on v1, the tightest of the cgroup and
// its parents, rounded up), 0 if there is no quota
inline uint32_t cgroup_cpus(const std::string & proc_cgroup, const std::string & sys_root) {
    uint32_t res = 0;
    std::ifstream f(proc_cgroup);
    std::string line;
    while (std::getline(f, line)) {
        const size_t a = line.find(':');
        const size_t b = a == std::string::npos ? a : line.find(':', a + 1);
        if (b == std::string::npos) {
            continue;
        }
        const std::string ctrl = line.substr(a + 1, b - a - 1);
        std::string path = line.substr(b + 1);
        const bool v2 = line.compare(0, a, "0") == 0 && ctrl.empty();
        if (!v2 && ("," + ctrl + ",").find(",cpu,") == std::string::npos) {
            continue;
        }
        const std::string root = v2 ? sys_root : sys_root + "/cpu";
        while (true) {
            const std::string dir = root + (path == "/" ? "" : path);
            uint64_t quota = 0;
            uint64_t period = 0;
            bool ok = false;
            if (v2) {
                std::ifstream fm(dir + "/cpu.max");
                std::string q;
                if (fm >> q >> period && q != "max") {
                    try { quota = std::stoull(q); ok = true; } catch (...) { ok = false; }
                }
            } else {
                long long q = -1;
                std::ifstream fq(dir + "/cpu.cfs_quota_us");
                ok = (bool) (fq >> q) && q > 0 && read_u64(dir + "/cpu.cfs_period_us", period);
                quota = ok ? (uint64_t) q : 0;
            }
            if (ok && period > 0 && quota > 0) {
                const uint32_t n = (uint32_t) std::max<uint64_t>(1, (quota + period - 1)/period);
                res = res == 0 ? n : std::min(res, n);
            }
            if (path.empty() || path == "/") {
                break;
            }
            const size_t p = path.find_last_of('/');
            path = p == 0 || p == std::string::npos ? "/" : path.substr(0, p);
        }
    }
    return res;
}

// one logical CPU per physical core this process may run on (the lowest allowed SMT sibling of each core, Linux sysfs topology), at most
// the cgroup CPU quota and n_max of them; empty where the topology or the affinity mask cannot be read
inline std::vector<int> physical_cpus(int n_max) {
    std::vector<int> res;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return res;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        std::string list;
        if (!std::getline(f, list) || list.empty()) {
            return {};
        }
        // "0,24" or "0-1": the first allowed sibling of the core keeps it
        int first = -1;
        size_t p = 0;
        while (p < list.size() && first < 0) {
            const size_t e = list.find_first_of(",", p);
            const std::string tok = list.substr(p, e == std::string::npos ? std::string::npos : e - p);
            const size_t d = tok.find('-');
            int a = -1;
            int b = -1;
            try {
                a = std::stoi(tok.substr(0, d));
                b = d == std::string::npos ? a : std::stoi(tok.substr(d + 1));
            } catch (...) {
                return {};
            }
            for (int c = a; c <= b && first < 0; ++c) {
                if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &set)) {
                    first = c;
                }
            }
            p = e == std::string::npos ? list.size() : e + 1;
        }
        if (first == cpu) {
            res.push_back(cpu);
        }
    }
    const uint32_t q = cgroup_cpus("/proc/self/cgroup", "/sys/fs/cgroup");
    size_t n = res.size();
    if (q > 0) {
        n = std::min<size_t>(n, q);
    }
    n = std::min<size_t>(n, (size_t) std::max(0, n_max));
    res.resize(n);
#else
    (void) n_max;
#endif
    return res;
}

// logical CPUs this process may run on: the affinity mask (Linux), capped by a cgroup CPU quota
inline uint32_t usable_cpus() {
    uint32_t n = std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        n = std::max(1, CPU_COUNT(&set));
    }
    const uint32_t q = cgroup_cpus("/proc/self/cgroup", "/sys/fs/cgroup");
    if (q > 0) {
        n = std::min(n, q);
    }
#endif
    return n;
}

} // namespace llama_sysmem
