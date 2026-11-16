// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"

#include "../src/llama-ext.h"

#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(__linux__)
#include <sys/vfs.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    int32_t smt_factor = 1;
    int phy_cpus = common_cpu_get_num_physical_cores();
    int logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (phy_cpus > 0 && logical_cpus > phy_cpus) {
        smt_factor = logical_cpus / phy_cpus;
    }
    return phy_cpus * std::min(smt_factor, 2);
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        COM_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    num_digits = std::min<size_t>(num_digits, 128);

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            COM_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    const int verbosity = common_log_get_verbosity_thold();
    COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, verbosity);

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices && verbosity >= LOG_LEVEL_TRACE) {
        COM_TRC("%s", "device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    COM_TRC("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
    }
    if (*pattern == '[') {
        const char * class_end = pattern + 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (*class_end == ']' || *class_end == '-') {
            class_end++;
        }
        while (*class_end != '\0' && *class_end != ']') {
            class_end++;
        }
        if (*class_end == ']') {
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}

std::string wstring_to_utf8(const std::wstring & str) {
    if (str.empty()) {
        return std::string();
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0, NULL, NULL);

    if (size <= 0) {
        return std::string();
    }

    std::string utf8(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), &utf8[0], size, NULL, NULL);

    return utf8;
}
#endif

// returns the path as a UTF-8 string, preserving its separators
std::string fs_path_to_utf8(const std::filesystem::path & path) {
    const auto value = path.u8string();
    return std::string(value.begin(), value.end());
}

void fs_write_atomic(const std::filesystem::path & path, const std::string & data) {
    std::error_code ec;
    std::filesystem::path path_tmp = path;
    path_tmp += ".tmp";

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    std::ofstream file(path_tmp, std::ios::binary);
    file << data;
    file.close();

    if (!file.fail()) {
        std::filesystem::rename(path_tmp, path, ec);
    }

    if (file.fail() || ec) {
        std::filesystem::remove(path_tmp, ec);
        throw std::runtime_error("failed to write file: " + fs_path_to_utf8(path));
    }
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string common_get_env(const std::string & name) {
    const char * value = std::getenv(name.c_str());
    return value == nullptr ? "" : value;
}

void common_set_env(const std::string & name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (value.empty()) {
        unsetenv(name.c_str());
    } else {
        setenv(name.c_str(), value.c_str(), 1);
    }
#endif
}

// k15-fn: a draft model (no MoE cache of its own) is measured with the weight copies of one prefetch split counted
// (LLAMA_FIT_STAGING_SPLIT, see sched_reserve): with its experts on the host (--spec-draft-cpu-moe) the arena does not fit beside a
// full MoE cache, the copies then land in its graph allocation (Qwen3.8-Flash-Next MTP Q8_0 at 32 GB: measured 199 MiB of compute,
// 982 MiB once loaded, out of memory at the first draft). LLAMA_FIT_STAGING_SPLIT=0: the old measure
struct common_draft_measure_env {
    bool set = false;
    explicit common_draft_measure_env(bool on = true) {
        if (on && getenv("LLAMA_FIT_STAGING_SPLIT") == nullptr) {
            common_set_env("LLAMA_FIT_STAGING_SPLIT", "1");
            set = true;
        }
    }
    ~common_draft_measure_env() {
        if (set) {
            common_set_env("LLAMA_FIT_STAGING_SPLIT", "");
        }
    }
};


std::filesystem::path common_get_path_from_env(const std::string & name) {
#if defined(_WIN32)
    const std::wstring wname = utf8_to_wstring(name);
    const wchar_t * wvalue = _wgetenv(wname.c_str());
    return wvalue ? std::filesystem::path(wvalue) : std::filesystem::path();
#else
    const char * value = std::getenv(name.c_str());
    return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

#if !defined(_WIN32)
static std::filesystem::path get_home_directory() {
    std::filesystem::path home = common_get_path_from_env("HOME");
    if (!home.empty()) {
        return home;
    }
    const struct passwd * pw = getpwuid(getuid());
    if (!pw || !pw->pw_dir || !*pw->pw_dir) {
        throw std::runtime_error("Failed to find $HOME directory");
    }
    return pw->pw_dir;
}
#endif

std::filesystem::path fs_get_cache_directory() {
    std::filesystem::path cache_directory = common_get_path_from_env("LLAMA_CACHE");
    if (!cache_directory.empty()) {
        return cache_directory;
    }
#if defined(_WIN32)
    cache_directory = common_get_path_from_env("LOCALAPPDATA");
    if (cache_directory.empty()) {
        throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
    }
#elif defined(__APPLE__)
    cache_directory = get_home_directory() / "Library/Caches";
#else
    cache_directory = common_get_path_from_env("XDG_CACHE_HOME");
    if (cache_directory.empty()) {
        cache_directory = get_home_directory() / ".cache";
    }
#endif
    return cache_directory / "llama.cpp";
}

std::filesystem::path fs_get_config_directory() {
    std::filesystem::path config_directory;
#if defined(_WIN32)
    config_directory = common_get_path_from_env("APPDATA");
    if (config_directory.empty()) {
        throw std::runtime_error("Failed to find %APPDATA% directory");
    }
#else
    config_directory = common_get_path_from_env("XDG_CONFIG_HOME");
    if (config_directory.empty()) {
        config_directory = get_home_directory() / ".config";
    }
#endif
    return config_directory / "llama.cpp";
}

std::filesystem::path fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    const std::filesystem::path cache_directory = fs_get_cache_directory();
    std::error_code ec;
    common_create_directories(cache_directory, ec);
    if (ec) {
        throw std::runtime_error("failed to create cache directory: " + fs_path_to_utf8(cache_directory));
    }
    return cache_directory / std::filesystem::u8path(filename);
}

//
// TTY utils
//

bool common_is_tty(FILE * file) {
#if defined(_WIN32)
    return _isatty(_fileno(file));
#else
    return isatty(fileno(file));
#endif
}

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    return common_is_tty(stdout) || common_is_tty(stderr);
}

bool tty_enable_ansi() {
#if defined(_WIN32)
    // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
    for (DWORD id : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
        HANDLE h    = GetStdHandle(id);
        DWORD  mode = 0;
        if (GetConsoleMode(h, &mode) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            return false;
        }
    }
#endif
    return true;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    common_threadpools threadpools;

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

static const std::map<common_decision_type, std::string> COMMON_DECISION_TYPE_NAMES = {
    { COMMON_DECISION_TYPE_OPENJEV,        "openjev"       },
    { COMMON_DECISION_TYPE_LEV,            "lev"           },
    { COMMON_DECISION_TYPE_KEV,            "kev"           },
    { COMMON_DECISION_TYPE_NIMBLE,         "nimble"        },
    { COMMON_DECISION_TYPE_LAYA,           "laya"          },
    { COMMON_DECISION_TYPE_CLEF,           "clef"          },
    { COMMON_DECISION_TYPE_PPLX_DECIDER,   "pplx-decider"  },
    { COMMON_DECISION_TYPE_VLLM_DECISION2, "vllm-decision2" },
    { COMMON_DECISION_TYPE_VLLM_D3,        "vllm-d3"        },
    { COMMON_DECISION_TYPE_LFM2_D1,        "lfm2-d1"       },
    { COMMON_DECISION_TYPE_LFM2_D1_OMNI,   "lfm2-d1-omni"  },
    { COMMON_DECISION_TYPE_GRANITE_SWITCH, "granite-switch"},
    // for future models, ALWAYS add org name prefix to avoid name collisions, e.g. "orgname-modelname"
};

static common_decision_type common_decision_type_from_string(const std::string & str) {
    for (const auto & pair : COMMON_DECISION_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return COMMON_DECISION_TYPE_UNKNOWN;
}

common_decision_type common_get_decision_type(const struct llama_model * model) {
    char buf[64];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    const std::string key = std::string(buf) + ".decision.type";
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    return common_decision_type_from_string(buf);
}

common_gguf_info common_get_gguf_info(const std::string & fname) {
    common_gguf_info info;

    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(fname.c_str(), gguf_params));
    if (!gguf_ctx) {
        return info; // missing or unreadable file
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return info; // no architecture in the metadata
    }
    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch.empty()) {
        return info;
    }

    const int64_t type_id = gguf_find_key(gguf_ctx.get(), (arch + ".decision.type").c_str());
    if (type_id < 0) {
        info.decision_type = COMMON_DECISION_TYPE_NONE;
    } else if (gguf_get_kv_type(gguf_ctx.get(), type_id) == GGUF_TYPE_STRING) {
        info.decision_type = common_decision_type_from_string(gguf_get_val_str(gguf_ctx.get(), type_id));
    }

    // same key and type as the model loader
    const int64_t ctx_id = gguf_find_key(gguf_ctx.get(), (arch + ".context_length").c_str());
    if (ctx_id >= 0 && gguf_get_kv_type(gguf_ctx.get(), ctx_id) == GGUF_TYPE_UINT32) {
        info.n_ctx_train = gguf_get_val_u32(gguf_ctx.get(), ctx_id);
    }

    return info;
}

static struct llama_model_params common_model_params_to_llama_impl(common_params & params, bool fit_deferred);
static enum llama_load_mode common_moe_cache_load_mode(const common_params & params, enum llama_load_mode load_mode, int32_t n_gpu_layers);
static bool common_moe_lowram_maps(const common_params & params, int32_t n_gpu_layers, uint64_t * expert_bytes_out = nullptr, uint64_t * avail_out = nullptr);
static bool common_moe_lowram_short(const common_params & params, uint64_t * expert_bytes_out = nullptr, uint64_t * avail_out = nullptr);
static bool common_experts_on_offloaded_host(const common_params & params, const llama_model_tensor_buft_override * overrides, int32_t n_gpu_layers);
static bool common_moe_hybrid_gate_on(const common_params & params, const llama_model_tensor_buft_override * overrides, int32_t n_gpu_layers);
static double common_moe_host_expert_bytes(const std::string & path, const llama_model_tensor_buft_override * overrides, int32_t n_gpu_layers);
static double common_moe_hybrid_max_frac();
static int64_t common_model_n_layer(const std::string & path, std::vector<uint8_t> & experts);

// bytes of one expert in the MoE cache: the sum over the expert tensors of the layer (the size of a slot) and the largest single tensor (the
// size of a slot in its bank), with the number of experts, for the layer with the largest experts; false if the GGUF cannot be read
// lend_*: the group of layers with the same expert sizes that holds the most expert bytes (the MoE cache splits its budget between such
// groups by their host bytes and lends a graph allocation from a bank tail of its largest group, see llama_moe_cache lend): its slot,
// its largest tensor per expert and its share of the expert bytes of the model
// k19-df2cost: the DFlash2 cached window band's upper cached share (common/speculative.cpp) and the device memory it keeps for its second width
#ifndef K19_DF2COST_FRAC_MAX
#define K19_DF2COST_FRAC_MAX 0.5
#endif
#ifndef K19_DF2COST_RESERVE_MIB
#define K19_DF2COST_RESERVE_MIB 512
#endif
struct common_moe_slot_bytes { int64_t n_expert = 0; int64_t sum = 0; int64_t max = 0; int64_t total = 0; int64_t n_layer = 0; int64_t n_used = 0;
    int64_t lend_sum = 0; int64_t lend_max = 0; double lend_share = 1.0; };
static bool common_moe_slot_bytes_of(const std::string & path, common_moe_slot_bytes & out) {
    std::string prefix;
    int n_split = 1;
    static const std::regex re_split("-([0-9]{5})-of-([0-9]{5})\\.gguf$");
    std::smatch m;
    if (std::regex_search(path, m, re_split)) {
        prefix  = path.substr(0, m.position(0));
        n_split = std::stoi(m[2].str());
    }
    std::map<long, common_moe_slot_bytes> per_layer;
    int64_t total = 0;
    for (int i = 0; i < n_split; ++i) {
        std::string fname = path;
        if (n_split > 1) {
            std::vector<char> split_path(prefix.size() + 32);
            llama_split_path(split_path.data(), split_path.size(), prefix.c_str(), i, n_split);
            fname = split_path.data();
        }
        ggml_context * meta = nullptr;
        struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ &meta };
        gguf_context_ptr ctx(gguf_init_from_file(fname.c_str(), gp));
        if (!ctx) {
            return false;
        }
        ggml_context_ptr meta_ptr(meta);
        if (i == 0) { // experts per token (<arch>.expert_used_count), 0 if absent
            const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
            if (arch_id >= 0 && gguf_get_kv_type(ctx.get(), arch_id) == GGUF_TYPE_STRING) {
                const std::string key = std::string(gguf_get_val_str(ctx.get(), arch_id)) + ".expert_used_count";
                const int64_t kid = gguf_find_key(ctx.get(), key.c_str());
                if (kid >= 0 && gguf_get_kv_type(ctx.get(), kid) == GGUF_TYPE_UINT32) {
                    out.n_used = gguf_get_val_u32(ctx.get(), kid);
                } else if (kid >= 0 && gguf_get_kv_type(ctx.get(), kid) == GGUF_TYPE_INT32) {
                    out.n_used = gguf_get_val_i32(ctx.get(), kid);
                }
            }
        }
        for (int64_t t = 0; t < gguf_get_n_tensors(ctx.get()); ++t) {
            const char * name = gguf_get_tensor_name(ctx.get(), t);
            if (strncmp(name, "blk.", 4) == 0 && strstr(name, "_exps") != nullptr) {
                if (const ggml_tensor * w = ggml_get_tensor(meta, name)) {
                    total += (int64_t) ggml_nbytes(w); // every expert tensor (weights, biases, scales) of every layer
                }
            }
            if (strncmp(name, "blk.", 4) != 0 || strstr(name, "_exps.weight") == nullptr) {
                continue;
            }
            char * end = nullptr;
            const long il = strtol(name + 4, &end, 10);
            const ggml_tensor * w = ggml_get_tensor(meta, name);
            if (end == name + 4 || w == nullptr || w->ne[2] <= 1) {
                continue;
            }
            const int64_t b = (int64_t) (ggml_nbytes(w) / w->ne[2]);
            common_moe_slot_bytes & l = per_layer[il];
            l.n_expert = std::max<int64_t>(l.n_expert, w->ne[2]);
            l.sum += b;
            l.max  = std::max(l.max, b);
        }
    }
    const int64_t n_used = out.n_used;
    std::map<std::pair<int64_t, int64_t>, int64_t> group_bytes; // (slot, largest tensor) -> expert bytes of its layers
    int64_t layer_bytes = 0;
    for (const auto & it : per_layer) {
        if (it.second.sum > out.sum) {
            out = it.second;
        }
        group_bytes[{ it.second.sum, it.second.max }] += it.second.n_expert*it.second.sum;
        layer_bytes += it.second.n_expert*it.second.sum;
    }
    int64_t lend_bytes = 0;
    for (const auto & it : group_bytes) {
        if (it.second > lend_bytes) {
            lend_bytes   = it.second;
            out.lend_sum = it.first.first;
            out.lend_max = it.first.second;
        }
    }
    out.lend_share = layer_bytes > 0 ? (double) lend_bytes/(double) layer_bytes : 1.0;
    out.n_used  = n_used;
    out.total   = total;
    out.n_layer = (int64_t) per_layer.size();
    return out.sum > 0;
}

uint64_t common_moe_expert_bytes(const std::string & path) {
    common_moe_slot_bytes sb;
    return common_moe_slot_bytes_of(path, sb) ? (uint64_t) sb.total : 0;
}

double common_moe_cache_tokens(const std::string & path, uint64_t cache_bytes) {
    common_moe_slot_bytes sb;
    if (!common_moe_slot_bytes_of(path, sb) || sb.n_used <= 0 || sb.n_layer <= 0 || sb.sum <= 0) {
        return -1.0;
    }
    return (double) cache_bytes/((double) sb.n_layer*(double) sb.n_used*(double) sb.sum);
}

// n_ubatch of the draft context of speculative decoding (an MTP context or a draft model) for a target context at n_ubatch u: the MoE
// cache rules grow the target's n_ubatch so that a prompt uploads the experts it misses fewer times (common_moe_cache_auto_ubatch).
// The draft gains nothing from that (a draft model's weights are on the device or never uploaded per ubatch, an MTP context processes
// the prompt in chunks of its own n_ubatch over one layer), but its compute buffers grow with it and are counted against the cache: at
// 32 GB with Qwen3.8-Flash-Next + MTP the spill rule stopped at 4096 instead of 8192 because of them, so an 8k prompt was uploaded twice.
// Without -ub the draft keeps LLAMA_MOE_DRAFT_UB_MAX (2048; 0 = the target's n_ubatch, as before)
uint32_t common_draft_n_ubatch(const common_params & params, uint32_t u) {
    if (params.n_ubatch_set || params.moe_cache_size == 0) {
        return u;
    }
    static const int64_t cap = getenv("LLAMA_MOE_DRAFT_UB_MAX") != nullptr ? atoll(getenv("LLAMA_MOE_DRAFT_UB_MAX")) : 2048;
    return cap > 0 ? std::min<uint32_t>(u, (uint32_t) cap) : u;
}

// MoE cache mode without -ub: every expert a prompt ubatch does not find in the cache is uploaded once per ubatch, so fewer, larger
// ubatches upload less. n_ubatch is 2048 instead of the default (512); the growth of the compute buffers and of the KV cache is taken
// from the cache, so the device memory stays what the default ubatch would use. A larger n_ubatch (4096/8192, LLAMA_MOE_PREFILL_UB_MAX)
// only when it costs the decode nothing:
// - the KV cache does not grow with n_ubatch (the SWA cache holds the window plus one ubatch, every later decode step attends over it)
// - the cache still holds every expert of the model after giving up the growth (so the decode reads the same experts from it)
// LLAMA_MOE_PREFILL_UB_AUTO=1 (opt-in): the rule measured before (largest n_ubatch whose compute buffer growth over 2048 is at most
// LLAMA_MOE_PREFILL_UB_FRAC (0.15) of the cache, 4096 at most with a growing KV cache): faster prompts, but -1..-17% decode where the
// cache does not hold every expert. LLAMA_MOE_CACHE_LEND=2: the compute buffers are lent from the cache, it gives up only the KV growth.
// LLAMA_MOE_PREFILL_UB=0 disables (n_ubatch stays the default), =N forces N.
// speculative decoding with a model (a -md draft model, or a draft type that runs one: the built-in MTP heads, Eagle3, DFlash, DSpark)
static bool common_spec_draft_model(const common_params & params) {
    return params.speculative.has_dft() || std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP ||
                            t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK; });
}

static void common_moe_cache_auto_ubatch_rules(common_params & params, llama_model_params & mparams, llama_context_params & cparams);
static uint64_t common_model_file_bytes(const std::string & path);
static bool     common_model_file_in_ram(const std::string & path);
static uint64_t common_available_memory();

// <arch>.<key> of a GGUF as an integer (u32 / i32 / u64), -1 if absent or unreadable
static int64_t common_gguf_u32(const std::string & path, const char * key) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), gp));
    if (!ctx) {
        return -1;
    }
    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return -1;
    }
    const int64_t kid = gguf_find_key(ctx.get(), (std::string(gguf_get_val_str(ctx.get(), arch_id)) + "." + key).c_str());
    if (kid < 0) {
        return -1;
    }
    switch (gguf_get_kv_type(ctx.get(), kid)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx.get(), kid);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx.get(), kid);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx.get(), kid);
        default:               return -1;
    }
}

// k15-up: the layer-major batch (tokens per merged group). Default 8192; 16384 (one group for a 16k prompt, one pass over the host
// experts instead of two) where the draft model was left out by --fit (fit_off): no drafter or MTP heads keep n_batch-sized rows
// beside it (an MTP drafter on the device grew the peak RSS by 570 MiB at 16384, a dropped one by 50-96 MiB). A group whose lent
// allocation does not fit is refused and split (ggml sched lend guard), so a cache that cannot lend for 8 members runs 4 + 4 as at
// 8192. G4 Flash-Next UD-IQ4_XS Studio argv, 16k: 24 GB 3.661 -> 3.216 s, 16 GB (own-buffer lend) 3.820 -> 3.184 s, md5 unchanged.
// LLAMA_MOE_LAYER_MAJOR_BATCH=N sets N; LLAMA_MOE_LAYER_MAJOR_BATCH_AUTO=0 keeps 8192
// k15-up (2): built-in MTP heads (no -md model) keep one next-n row per batch token (llama_context::output_reserve, embd_nextn unmasked:
// n_embd floats x n_batch), which is all the drafter's prompt pass needs after the batch. 16384 adds n_embd x 4 B x 8192 to the host
// output buffer; it is taken where that is at most LLAMA_MOE_LAYER_MAJOR_BATCH_RSS_MIB (100): Qwen3.6-35B-A3B (n_embd 2048) +64 MiB.
// The rest of the measured peak RSS growth (~170 MiB, G4 q4x 12 GB +237) is the 8-member graph's own host buffers, the same without a
// draft (Flash-Next 16 GB, draft dropped: +170)
// k17-small: what a launch without the drafter passes where the fit dropped an MTP drafter (LLAMA_FIT_MTP_DROP, the 2-thread MTP-cache
// rule): no draft types (no next-n rows, one output per sequence, no rollback states), no draft context flag (the layer-major refill
// and the state capture of a drafting server), the default capture count. Before this the server ran in the "draft left out" mode:
// a 16384-token layer-major batch, the capture and refill of a drafting context, and its prompts and checkpoints differed from a
// plain launch (2k / 8k text not equal to it). The trial fit runs with these context parameters, so the placement is that launch's
// k19-rollback: where --fit leaves the draft model out (no room on the device beside the main model and its MoE cache: G4 capped at 8 GB,
// Qwen3.8-Flash-Next UD-IQ4_XS + its Q8_0 MTP drafter), the server drafts n-gram suffix windows instead (cost-priced, as the server's own
// n-gram drafting). Their rollback keeps no per-token recurrent rows on the device: the dropped draft's context already runs with n_rs_seq 0,
// so a partly accepted window of a recurrent model restores the server's checkpoint of the state (host memory) and decodes the accepted
// tokens again. Where nothing repeats nothing is drafted, and the launch is the plain one (no rows, no checkpoint).
// LLAMA_SPEC_DROP_NGRAM: 0 off (the server runs without drafts, as before), 1 on; LLAMA_SPEC_DROP_NGRAM_SPILL: the drafting mode under a
// MoE cache with host experts (server_spec_spill_mode; default 1: drafts verified as ordinary batches with the host path, other text)
static int common_fit_drop_ngram_env() {
    const char * e = getenv("LLAMA_SPEC_DROP_NGRAM");
    return e != nullptr ? atoi(e) : 0;
}

static bool common_fit_drop_ngram(common_params & params, llama_context_params & cp) {
    auto & sp = params.speculative;
    if (common_fit_drop_ngram_env() == 0 || params.embedding || params.n_parallel < 1 || sp.has_synth() ||
            !sp.ngram_cache.lookup_cache_static.empty() || !sp.ngram_cache.lookup_cache_dynamic.empty()) {
        return false;
    }
    sp.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX };
    sp.ngram_suffix.auto_mode     = true;
    sp.ngram_suffix.drop_fallback = true;
    const char * es = getenv("LLAMA_SPEC_DROP_NGRAM_SPILL");
    sp.ngram_suffix.drop_spill = es != nullptr ? std::max(0, std::min(3, atoi(es))) : 1;
    // the verify windows need their outputs (the server sized them for the dropped drafter's n_max)
    const common_speculative_output_limits lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&sp));
    params.n_outputs_max         = std::max(params.n_outputs_max, lim.total);
    params.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, lim.per_seq);
    cp.n_outputs_max             = std::max(params.n_outputs_max, 0);
    cp.n_outputs_max_per_seq     = std::max(params.n_outputs_max_per_seq, 0);
    cp.n_rs_seq                  = 0;
    COM_INF("%s: --fit: the draft model does not fit: n-gram suffix drafts instead, %d tokens per window at most, rolled back from a host "
        "checkpoint (no recurrent rows on the device; LLAMA_SPEC_DROP_NGRAM=0: no drafts)\n", __func__, common_speculative_n_max(&sp));
    return true;
}

static void common_fit_plain_launch(common_params & params, llama_context_params & cp) {
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
    const common_speculative_output_limits lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel, 0);
    params.n_outputs_max         = lim.total;
    params.n_outputs_max_per_seq = lim.per_seq;
    cp.n_rs_seq              = params.speculative.need_n_rs_seq();
    cp.n_outputs_max         = std::max(params.n_outputs_max, 0);
    cp.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 0);
    cp.spec_draft            = false;
    cp.n_state_capture       = llama_context_default_params().n_state_capture;
}

static int64_t common_lm_batch_default(const common_params & params, bool * nodraft = nullptr) {
    if (const char * e = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH")) {
        return atoll(e);
    }
    // (was opt-in: G4 k15g4y1 Flash-Next 16 GB VmHWM +170 (fn4) / +188 (fn1) MiB, transient during the group, mostly its host KQ masks)
    // k15-pp: on by default again (LLAMA_MOE_LAYER_MAJOR_BATCH_AUTO=0: 8192) now that the 16384 group's own host memory is gone:
    // device-built KQ masks (no host mask per member), no CUDA graph records for the merged graph, next-n rows on the device beside
    // MTP heads. G4 fn4 16 GB (k15g4n1): VmHWM 75300 vs 75347 MiB for the 8192 default with host masks, 16k TTFT 3.24 vs 4.07 s
    static const bool auto_env = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_AUTO") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_AUTO")) != 0;
    if (!auto_env) {
        return 8192;
    }
    if (params.speculative.draft.fit_off && params.speculative.has_dft()) {   // a dropped MTP drafter: as a plain launch (k17-small)
        return 16384;
    }
    const bool mtp_heads = !params.speculative.has_dft() && std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP; });
    // k17: no draft model and no MTP heads (no speculation, or n-gram drafts only) keep no next-n rows either, as a dropped draft does:
    // 16384 there too (one pass over the host experts for a 16k prompt and no group boundary). G4 k17fn24 Flash-Next UD-IQ4_XS 24 GB no
    // draft 16k / 30k TTFT 3.34 / 6.41 -> 3.03 / 5.81 s at the previous depth. LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT=0 keeps 8192
    static const bool nodraft_env = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT")) != 0;
    if (nodraft_env && !params.speculative.has_dft() && !mtp_heads) {
        if (nodraft) {
            *nodraft = true;   // the caller keeps 8192 where the cache has no room for the 16384 graphs (common_lm_nodraft_room)
        }
        return 16384;
    }
    // k19-np12c, opt-in (LLAMA_MOE_LAYER_MAJOR_BATCH_LEND=1): beside a separate MTP drafter (-md, draft-mtp) whose device weights the
    // target's prompt groups borrow (kdp, LLAMA_MTP_DRAFT_LEND) 16384. Off by default: the server's drafter pass covers the whole last
    // batch, so a 30k prompt passed 14332 drafter rows instead of 6140 (G4 Flash-Next UD-IQ4_XS 12 GB, Studio's -np 4 with the q3_k drafter:
    // TTFT 6.89 vs 6.66 s, e2e 9.837 vs 9.572 s), and the choice is made before --fit knows whether anything is lent (16 GB, nothing lent:
    // MoE cache 4488 vs 4576 MiB, 30k e2e 8.616 vs 8.486 s)
    static const bool lend_batch_env = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_LEND") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_LEND")) != 0;
    const char * e_lend = getenv("LLAMA_MTP_DRAFT_LEND");
    const bool   lend_on = e_lend == nullptr || atoi(e_lend) != 0;
    const bool   dft_mtp = params.speculative.has_dft() && std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP; });
    if (lend_batch_env && lend_on && dft_mtp) {
        return 16384;
    }
    // beside built-in MTP heads 16384 (LLAMA_MOE_LAYER_MAJOR_BATCH_MTP=0: 8192). It was opt-in: G4 k15g4q1 q4x 12 GB 16k TTFT -13% but
    // decode 8k / 16k -11 / -10%, RssAnon +95 MiB. k17 (on the scheduler trim, the input arena fit and the MTP state release): G4 k17lmm3 q4x
    // + MTP 12 GB, 3 launches: 16k / 30k TTFT 1.46 -> 1.34 / 2.74 -> 2.36 s, decode equal, md5 sets equal, end RssAnon 2446-2459 against
    // 2497-2509 MiB for the previous 8192 default; KLD inside the model's own band. Only beside built-in heads: a -md draft keeps 8192
    static const bool mtp_env = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_MTP") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_MTP")) != 0;
    if (mtp_heads && mtp_env) {
        static const double rss_mib = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_RSS_MIB") ? atof(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_RSS_MIB")) : 100.0;
        const int64_t n_embd = common_gguf_u32(params.model.path, "embedding_length");
        if (n_embd > 0 && (double) n_embd*4.0*8192.0/1048576.0 <= rss_mib) {
            return 16384;
        }
    }
    return 8192;
}

// layer-major prompts (llama_context::decode_layer_major): the ubatches of one batch upload each host expert once, so n_ubatch stays the
// default (no compute buffer growth taken from the cache) and the batch holds up to LLAMA_MOE_LAYER_MAJOR_BATCH (8192) tokens, at most the
// context and lm_ub_max ubatches. LLAMA_MOE_LAYER_MAJOR=1 forces it for spilled experts, =0 turns it off; unset (auto): on for spilled
// experts with speculative decoding by a model (the built-in MTP heads, a -md draft), where the rules below keep the default n_ubatch
// (no lending beside a draft: G4, Qwen3.6-35B-A3B UD-Q3_K_M + MTP at 8 GB, 2k / 8k prompt 2288 -> 3709 / 2430 -> 6609 tok/s, same text);
// where they grow n_ubatch (a lent graph) the prompt stays ubatch by ubatch
// the share of the host experts below which a prompt's layer-major groups run beside the hybrid gate (llama_context lm_hybrid_frac_ok,
// LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC, default 0.25). k18 (LLAMA_MOE_LM_GATE_BATCH, default on; =0 the k16 line, 0.05): the batch of an
// explicit -ub grows to hold groups wherever the context runs them; with 0.05 a cache of 5-10% kept one ubatch per batch under the gate,
// so no group formed (G4, Qwen3.8-Flash-Next UD-IQ4_XS 12 GB, 6.3% cached, Studio -ub 2048: 8k prompt 1945 tok/s against 4463 at
// 16 GB with groups; 16 GB beside a Q8_0 MTP drafter, 8.0%: 1845 tok/s, TTFT 1.84 -> 4.44 s)
// k17-fnbatch: the 16384 batch without a draft (common_lm_batch_default) only where the cache has room for its merged graphs: the member
// estimate n_ubatch x 4 B x (4 n_embd + 4 n_expert_used n_ff_exp), (k - 1) of them, at most 1/LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM (2)
// of the cache; else 8192. G4 / T4 k17nbg / k17nbo: Flash-Next UD-IQ4_XS 12 GB (2321 MiB cache) 16k / 30k text changed beside the
// hybrid gate for -0.7 / -1.1% TTFT; gpt-oss-20b 8 GB on T4 (3026 MiB cache) still split 4 + 4, its input arena 91 -> 182 MiB and 16k
// TTFT +1.0%; 24 GB (15068 MiB) one group, 30k TTFT -2.3%, text unchanged
static bool common_lm_nodraft_room(const common_params & params, int64_t n_ubatch, int64_t want, int64_t * est_out = nullptr) {
    static const double room = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM") ? std::max(0.0, atof(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM"))) : 2.0;   // 0: always 16384 (the callers skip the check)
    const int64_t n_embd = common_gguf_u32(params.model.path, "embedding_length");
    const int64_t n_ff   = common_gguf_u32(params.model.path, "expert_feed_forward_length");
    const int64_t n_used = common_gguf_u32(params.model.path, "expert_used_count");
    const int64_t k      = n_ubatch > 0 ? want/n_ubatch : 0;
    if (n_embd <= 0 || n_ff <= 0 || n_used <= 0 || k < 2) {
        return false;
    }
    const int64_t est = n_ubatch*4*(4*n_embd + 4*n_used*n_ff);
    if (est_out) {
        *est_out = est;
    }
    return (double) params.moe_cache_size >= room*(double) ((k - 1)*est);
}

static double common_lm_hybrid_frac() {
    if (const char * e = getenv("LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC")) {
        return std::max(0.0, atof(e));
    }
    const char * g = getenv("LLAMA_MOE_LM_GATE_BATCH");
    return g != nullptr && atoi(g) == 0 ? 0.05 : 0.25;
}

static void common_moe_cache_auto_ubatch(common_params & params, llama_model_params & mparams, llama_context_params & cparams) {
    if (params.moe_cache_size == 0 || params.embedding || cparams.embeddings) {
        return;
    }
    const char * env_lm = getenv("LLAMA_MOE_LAYER_MAJOR");
    const int    lm     = env_lm == nullptr || strcmp(env_lm, "auto") == 0 ? -1 : (atoi(env_lm) != 0 ? 1 : 0);
    // k15-small4: layer-major prompts without a draft (LLAMA_MOE_LAYER_MAJOR_NODRAFT=0 keeps the prompt ubatch by ubatch)
    static const bool nd_on = getenv("LLAMA_MOE_LAYER_MAJOR_NODRAFT") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_NODRAFT")) != 0;
    if (params.n_ubatch_set) {
        // k9-prompt-pass: an explicit -ub without -b (Unsloth Studio passes --ubatch-size 2048 for a vision projector) kept n_batch at its
        // default 2048, so a batch held one ubatch, layer-major never engaged and a long prompt copied every spilled expert once per
        // ubatch (G4, Flash-Next UD-IQ4_XS 32 GB, 8k prompt: 4 passes, 147.7 GB). The ubatch stays as given (none of the rules below runs),
        // only the batch grows to hold several ubatches, under the same conditions as layer-major auto (or LLAMA_MOE_LAYER_MAJOR=1).
        // LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB=0 keeps n_batch. Not on unified memory under Windows (params.moe_uma_keep_batch) unless
        // LLAMA_MOE_LAYER_MAJOR=1: groups of large ubatches were slower there than one ubatch per graph (see common_init_result)
        static const bool env_x = getenv("LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB")) != 0;
        if (!env_x || lm == 0 || params.n_batch_set || (params.moe_uma_keep_batch && lm != 1) || !(lm == 1 || common_spec_draft_model(params) || nd_on) ||
                !common_experts_on_offloaded_host(params, mparams.tensor_buft_overrides, mparams.n_gpu_layers)) {
            return;
        }
        // k10: not where the MoE cache's default hybrid gate will run (it holds at most LLAMA_MOE_CACHE_HYBRID_MAX_FRAC of the host experts):
        // that path refuses every layer-major group, so the raise only changed the batches (local B200, Qwen3.6 UD-Q4_K_XL + MTP, 1500 MiB
        // cache, -ub 2048: 8k text differs from k9 with no group run)
        // k16: except below LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC (0.05) of the host experts, where llama_context runs the groups beside the
        // gate (k13-small, lm_hybrid_frac_ok) but this early return kept n_batch at one ubatch, so no group formed: G4, Qwen3.8-Flash-Next
        // UD-IQ4_XS 12 GB with Studio's -ub 2048 (cache 2642 of 56762 MiB, 4.7%), 8k / 16k / 30k prompt 1917 / 1934 / 1896 tok/s against
        // 4396 / 4460 at 16 GB (groups) and Strata 3345 / 4261 / 4129. LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB_HYBRID=0 restores the early return
        static const bool xh_on = getenv("LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB_HYBRID") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB_HYBRID")) != 0;
        static const double lm_hyb_frac = common_lm_hybrid_frac();
        const double host_x = xh_on ? common_moe_host_expert_bytes(params.model.path, mparams.tensor_buft_overrides, mparams.n_gpu_layers) : 0.0;
        const bool lm_beside_gate = xh_on && host_x > 0.0 && (double) params.moe_cache_size < lm_hyb_frac*host_x;
        if (lm != 1 && !lm_beside_gate && common_moe_hybrid_gate_on(params, mparams.tensor_buft_overrides, mparams.n_gpu_layers)) {
            COM_INF("MoE cache: n_batch stays %u at the given n_ubatch %u (the cache holds a small share of the host experts: the hybrid gate, "
                "no layer-major groups)\n", cparams.n_batch, cparams.n_ubatch);
            return;
        }
        bool nd16 = false;
        int64_t want = common_lm_batch_default(params, &nd16);
        const int64_t ub_max = getenv("LLAMA_MOE_LAYER_MAJOR_UB_MAX") != nullptr ? std::max(1, atoi(getenv("LLAMA_MOE_LAYER_MAJOR_UB_MAX"))) : 16;
        want = std::min<int64_t>(want, ub_max*(int64_t) cparams.n_ubatch);
        if (params.n_ctx > 0) {
            want = std::min<int64_t>(want, params.n_ctx);
        }
        // the cache given up for the merged graphs of k ubatches (k17-fnbatch: a function of k, to compare the 16384 and 8192 batches)
        auto reserve_for = [&](const int64_t k) -> int64_t {
            int64_t reserve = -1;
            if (const char * e = getenv("LLAMA_MOE_LAYER_MAJOR_RESERVE_MIB")) {
                reserve = std::max<int64_t>(0, atoll(e)) << 20;
            } else {
                const bool ckpt_batch = getenv("LLAMA_MOE_LAYER_MAJOR_CKPT") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_CKPT")) != 0;
                double frac = ckpt_batch ? 0.025 : 0.0;
                // k18 (LLAMA_MOE_LM_GATE_RESERVE_FRAC, default 0.10; 0 = as before): beside the hybrid gate the fit leaves ~550 MiB
                // free while a 4-member group at -ub 2048 grows the compute buffers ~850 MiB, and the cache cannot lend the whole merged
                // allocation, so every 8k prompt ran as 2 + 2 and read the host experts twice (G4 Flash-Next UD-IQ4_XS: 12 GB no draft
                // and 16 GB beside the Q8_0 MTP drafter, "split at 2", 8k TTFT 2.62 / 2.88 s against 1.83 at 16 GB with one graph)
                // Only the shortfall: (k - 1) member estimates (+10%) less LLAMA_MOE_LM_GATE_RESERVE_FREE_MIB (default 512, the free memory
                // found beside the gate at group time: 545 / 576 MiB on G4), at most that share of the cache; a group that fits takes nothing
                double gate_frac = 0.0;
                if (lm_beside_gate && lm != 1 && common_moe_hybrid_gate_on(params, mparams.tensor_buft_overrides, mparams.n_gpu_layers)) {
                    gate_frac = 0.10;
                    if (const char * eg = getenv("LLAMA_MOE_LM_GATE_RESERVE_FRAC")) {
                        gate_frac = std::max(0.0, atof(eg));
                    }
                }
                int64_t gate_free = 512ll << 20;
                if (const char * eg = getenv("LLAMA_MOE_LM_GATE_RESERVE_FREE_MIB")) {
                    gate_free = std::max<int64_t>(0, atoll(eg)) << 20;
                }
                if (const char * ef = getenv("LLAMA_MOE_LAYER_MAJOR_RESERVE_FRAC")) {
                    frac = std::max(0.0, atof(ef));
                }
                const int64_t n_embd = common_gguf_u32(params.model.path, "embedding_length");
                const int64_t n_ff   = common_gguf_u32(params.model.path, "expert_feed_forward_length");
                const int64_t n_used = common_gguf_u32(params.model.path, "expert_used_count");
                if (n_embd > 0 && n_ff > 0 && n_used > 0 && k > 1) {
                    const int64_t est = (int64_t) cparams.n_ubatch*4*(4*n_embd + 4*n_used*n_ff);
                    reserve = std::min<int64_t>((int64_t) (frac*(double) params.moe_cache_size), (k - 1)*est);
                    if (gate_frac > 0.0) {
                        const int64_t need = std::max<int64_t>(0, (int64_t) (1.1*(double) ((k - 1)*est)) - gate_free);
                        reserve = std::max<int64_t>(reserve, std::min<int64_t>((int64_t) (gate_frac*(double) params.moe_cache_size), need));
                    }
                }
            }
            return reserve;
        };
        // k17-fnbatch: the 16384 batch without a draft only where it gives up no more of the cache than 8192 and the cache has room
        // for its graphs (common_lm_nodraft_room); else 8192. G4 k17nbg Flash-Next UD-IQ4_XS 16 GB beside the hybrid gate: 540.51 instead
        // of 412.00 MiB kept free (cache 4859 vs 4992 MiB), 8k / 30k TTFT +2.0 / +1.0%. LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM=0: always
        const int64_t ub_n = (int64_t) cparams.n_ubatch;
        static const bool nd_room_on = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM") == nullptr || atof(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM")) > 0.0;
        if (nd16 && nd_room_on && want > 8192) {
            const int64_t want_lo = 8192;
            int64_t est = 0;
            const bool room = common_lm_nodraft_room(params, ub_n, want, &est);
            const int64_t r_hi = want    >= 2*ub_n ? reserve_for(want/ub_n)    : 0;
            const int64_t r_lo = want_lo >= 2*ub_n ? reserve_for(want_lo/ub_n) : 0;
            if (!room || r_hi > r_lo) {
                COM_INF("MoE cache: layer-major batch without a draft stays %lld: %s (cache %.2f MiB, %lld graph members of %.2f MiB; "
                    "kept free %.2f vs %.2f MiB; LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM=0 for 16384)\n", (long long) want_lo,
                    !room ? "no room in the cache for the 16384 graphs" : "16384 would take more of the cache", params.moe_cache_size/1048576.0,
                    (long long) (want/ub_n - 1), est/1048576.0, r_hi/1048576.0, r_lo/1048576.0);
                want = want_lo;
            }
        }
        if (want >= 2*(int64_t) cparams.n_ubatch && want > (int64_t) cparams.n_batch) {
            // the server clamps its checkpoint offsets to the old n_batch: the same batches and ubatches as without the raise
            params.n_batch_ckpt = (int32_t) cparams.n_batch;
            params.n_batch  = (int32_t) want;
            cparams.n_batch = (uint32_t) want;
            // k15-small4: without a draft the context turns layer-major on only with this mark (as the n_ubatch-default branch below sets it;
            // an explicit -ub, e.g. Studio's --ubatch-size 2048, never reached that branch)
            if (lm != 1 && !common_spec_draft_model(params)) {
                common_set_env("LLAMA_MOE_LAYER_MAJOR_AUTO_NODRAFT", "1");
            }
            // a merged graph of k ubatches keeps what crosses its cuts for every member; at a large ubatch that is more than the fit leaves
            // free (G4, Flash-Next UD-IQ4_XS 32 GB + Q8_0 MTP drafter, -ub 2048: 4 ubatches needed 755 MiB, 744 MiB were free, so the group
            // ran as 2 + 2 and read the experts twice). With LLAMA_MOE_LAYER_MAJOR_RESERVE_FRAC (default 0.025 when the server checkpoint is
            // one batch back, LLAMA_MOE_LAYER_MAJOR_CKPT=1, else 0) the cache gives up min(frac of itself, (k - 1) x the estimate of one
            // member: n_ubatch x 4 B x (4 n_embd + 4 n_expert_used n_ff_exp)); LLAMA_MOE_LAYER_MAJOR_RESERVE_MIB=N takes N MiB.
            // Not by default: with the checkpoint one ubatch back an 8k prompt is a group of 3 (fits) + the checkpoint ubatch, and the 481 MiB
            // the cache gave up there cost the decode after it 1.9% (hit rate 86.84 -> 86.62%) for no prompt gain (same G4 VM, 8k cell)
            int64_t reserve = reserve_for((int64_t) cparams.n_batch/(int64_t) cparams.n_ubatch);
            if (reserve > 0 && (int64_t) params.moe_cache_size > 2*reserve) {
                params.moe_cache_size  -= (size_t) reserve;
                cparams.moe_cache_size  = params.moe_cache_size;
                // k11: the memory given up here is for the merged graphs only. The check whether a prompt's host seed fits
                // (llama_context::sched_reserve_seed) must not count it as free: with it counted, a 128 MiB reserve made the host seed fit
                // and replaced the device seed of Flash-Next UD-IQ4_XS 32 GB (Studio argv), whose 2k prompt then waited on the routing of
                // each layer (G4 k11pgAA: 2k -8.5%, 20004 instead of 5731 uploads per ubatch)
                common_set_env("LLAMA_MOE_LAYER_MAJOR_RESERVED_BYTES", std::to_string(reserve));
            } else {
                reserve = 0;
            }
            COM_INF("MoE cache: layer-major prompts at the given n_ubatch %u: n_batch %u, %.2f MiB of the cache kept free for their graphs "
                "(LLAMA_MOE_LAYER_MAJOR_EXPLICIT_UB=0 keeps n_batch, LLAMA_MOE_LAYER_MAJOR_RESERVE_MIB=0 the cache)\n",
                cparams.n_ubatch, cparams.n_batch, reserve/1048576.0);
        }
        return;
    }
    const bool   spill  = lm != 0 && common_experts_on_offloaded_host(params, mparams.tensor_buft_overrides, mparams.n_gpu_layers);
    const auto   set_batch = [&]() {
        if (!params.n_batch_set) {
            bool nd16 = false;
            int64_t want = common_lm_batch_default(params, &nd16);
            const int64_t ub_max = getenv("LLAMA_MOE_LAYER_MAJOR_UB_MAX") != nullptr ? std::max(1, atoi(getenv("LLAMA_MOE_LAYER_MAJOR_UB_MAX"))) : 16;
            want = std::min<int64_t>(want, ub_max*(int64_t) cparams.n_ubatch);
            if (params.n_ctx > 0) {
                want = std::min<int64_t>(want, params.n_ctx);
            }
            // k17-fnbatch: as for an explicit -ub, 16384 without a draft only where the cache has room for its graphs
            static const bool nd_room_on = getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM") == nullptr || atof(getenv("LLAMA_MOE_LAYER_MAJOR_BATCH_NODRAFT_ROOM")) > 0.0;
            if (nd16 && nd_room_on && want > 8192 && !common_lm_nodraft_room(params, (int64_t) cparams.n_ubatch, want)) {
                want = 8192;
            }
            if (want > (int64_t) cparams.n_batch) {
                params.n_batch  = (int32_t) want;
                cparams.n_batch = (uint32_t) want;
            }
        }
    };
    if (lm == 1 && spill) {
        set_batch();
        COM_INF("MoE cache: layer-major prompts: n_ubatch stays %u, n_batch %u (LLAMA_MOE_LAYER_MAJOR=1)\n", cparams.n_ubatch, cparams.n_batch);
        return;
    }
    const uint32_t ub0 = cparams.n_ubatch;
    common_moe_cache_auto_ubatch_rules(params, mparams, cparams);
    if (lm == -1 && spill && common_spec_draft_model(params) && cparams.n_ubatch == ub0) {
        set_batch();
        COM_INF("MoE cache: layer-major prompts beside speculative decoding: n_ubatch stays %u, n_batch %u (LLAMA_MOE_LAYER_MAJOR=0 disables)\n",
            cparams.n_ubatch, cparams.n_batch);
    } else if (lm == -1 && spill && !common_spec_draft_model(params) && cparams.n_ubatch == ub0) {
        // k15-small4: without a draft too, wherever the rules above keep the default n_ubatch with spilled experts (the small-device
        // placement keeps it for good): every ubatch otherwise uploads the experts the cache misses again. T4 4 GB, Qwen3.6-35B-A3B
        // UD-Q4_K_XL, no MTP: 2k / 8k prompt 163 / 167 tok/s ubatch by ubatch against 590 / 911 in layer-major groups (the same ubatches,
        // layer by layer). LLAMA_MOE_LAYER_MAJOR_NODRAFT=0 keeps the prompt ubatch by ubatch
        if (nd_on) {
            set_batch();
            common_set_env("LLAMA_MOE_LAYER_MAJOR_AUTO_NODRAFT", "1");
            COM_INF("MoE cache: layer-major prompts without a draft: n_ubatch stays %u, n_batch %u (LLAMA_MOE_LAYER_MAJOR_NODRAFT=0 disables)\n",
                cparams.n_ubatch, cparams.n_batch);
        }
    }
}

static void common_moe_cache_auto_ubatch_rules(common_params & params, llama_model_params & mparams, llama_context_params & cparams) {
    const char * env = getenv("LLAMA_MOE_PREFILL_UB");
    const int    force = env != nullptr ? atoi(env) : -1;
    if (force == 0) {
        return;
    }
    // k15-pp: LLAMA_MTP_EXACT=1 promises the drafted text byte equal to the text without a draft, but these rules raise n_ubatch
    // without a model drafter and keep the default beside one, and the prompt's numerics follow n_ubatch (Qwen3.6-35B-A3B UD-Q4_K_XL +
    // MTP heads, --cpu-moe --moe-cache-mib 4000, strict: no draft at n_ubatch 4096 differed from the drafter at 512 from the second
    // token on; the no-draft arm at 512 matched it and the all-device runs). In the exact mode n_ubatch stays the default with and
    // without a draft (the layer-major groups below then read the prompt, as beside a draft). LLAMA_MOE_PREFILL_UB=N still forces N,
    // LLAMA_MOE_PREFILL_UB_EXACT=1 keeps the rules
    if (force < 0) {
        static const bool exact_keep = [] {
            const char * x = getenv("LLAMA_MTP_EXACT");
            const char * k = getenv("LLAMA_MOE_PREFILL_UB_EXACT");
            return !(x != nullptr && atoi(x) != 0) || (k != nullptr && atoi(k) != 0);
        }();
        if (!exact_keep) {
            COM_INF("MoE cache: n_ubatch stays %u with LLAMA_MTP_EXACT=1, with or without a draft (LLAMA_MOE_PREFILL_UB_EXACT=1 keeps "
                "the rules, LLAMA_MOE_PREFILL_UB=N forces N)\n", cparams.n_ubatch);
            return;
        }
    }
    // k13-small: no larger n_ubatch beside the small-device placement (common_fit_small_device): its 256 MiB margin has no room for the
    // run-time buffers of a larger ubatch. G4 4 GB q3m without a draft: "every expert cached" took n_ubatch 2048 (compute 804 MiB) and the
    // first 2k prompt ran out of device memory (CUDA error). LLAMA_MOE_PREFILL_UB=N still forces N
    // k19-l4poly: not on a host-bound small device (common_fit_hostbound): its larger ubatches run with the cache banks released (the
    // tiny-cache rule below, k19-tiny-ub, or the k19-place8 context), so they are sized within the cache, which keeps its size
    if (force < 0 && common_fit_small_device() && !common_fit_hostbound()) {
        COM_INF("MoE cache: small device: n_ubatch stays %u (LLAMA_MOE_PREFILL_UB=N forces N)\n", cparams.n_ubatch);
        return;
    }
    if (!common_experts_on_offloaded_host(params, mparams.tensor_buft_overrides, mparams.n_gpu_layers)) {
        return;
    }
    const char * env_a = getenv("LLAMA_MOE_PREFILL_UB_AUTO");
    const bool   rule_frac = env_a != nullptr && atoi(env_a) != 0;
    double frac = 0.15;
    if (const char * env_f = getenv("LLAMA_MOE_PREFILL_UB_FRAC")) {
        frac = std::max(0.0, atof(env_f));
    }
    // LLAMA_MOE_PREFILL_UB_AUTO=1 or LLAMA_MOE_CACHE_LEND=2 with a KV cache sized by n_ubatch: at most 4096 (measured -14% decode after an
    // 8k prompt at n_ubatch 8192 with every expert cached, gemma-4)
    uint32_t ub_max_kv = 4096;
    if (const char * env_s = getenv("LLAMA_MOE_PREFILL_UB_SWA_MAX")) {
        ub_max_kv = (uint32_t) std::max(0, atoi(env_s));
    }

    const uint32_t ub_def = std::min(cparams.n_batch, cparams.n_ubatch == 0 ? cparams.n_batch : cparams.n_ubatch);
    // with causal attention the batch is limited by the context size (0 = the training context, not known here)
    const uint32_t ub_lim = params.n_batch_set ? cparams.n_batch : (params.n_ctx > 0 ? (uint32_t) params.n_ctx : UINT32_MAX);
    // at most 8192 (LLAMA_MOE_PREFILL_UB_MAX): 12288 measured no faster than 8192 on 8k/16k prompts and 2-6% slower decode
    uint32_t ub_max = 8192;
    if (const char * env_m = getenv("LLAMA_MOE_PREFILL_UB_MAX")) {
        ub_max = (uint32_t) std::max(0, atoi(env_m));
    }
    std::vector<uint32_t> cands = { 2048, 4096, 8192, 16384 };
    cands.erase(std::remove_if(cands.begin() + 1, cands.end(), [&](uint32_t u) { return u > ub_max; }), cands.end());
    if (force > 0) {
        cands = { (uint32_t) force };
    }
    cands.erase(std::remove_if(cands.begin(), cands.end(), [&](uint32_t u) { return u <= ub_def || u > ub_lim; }), cands.end());
    if (cands.empty()) {
        return;
    }

    // device memory of a context at ubatch u, summed over the devices: {compute buffers, compute buffers + KV}; < 0 if the measure failed
    // the measure counts the compute buffers of the loaded context (LLAMA_MOE_MEASURE_LENT while it runs, see sched_reserve): without the
    // staging copies, which the cache lends (and which do not change with n_ubatch), with the seed graph of the MoE cache
    struct mem { int64_t compute; int64_t total; int n_dev = 0; int64_t dft = 0; }; // dft: the draft context (in total)
    // (not with LLAMA_MOE_CACHE_LEND=2, whose rule counts the staging copies lent together with the graph allocation)
    const bool lend2_env    = getenv("LLAMA_MOE_CACHE_LEND") != nullptr && atoi(getenv("LLAMA_MOE_CACHE_LEND")) == 2;
    const bool had_lent_env = getenv("LLAMA_MOE_MEASURE_LENT") != nullptr || lend2_env;
    if (!had_lent_env) {
        common_set_env("LLAMA_MOE_MEASURE_LENT", "1");
    }
    // speculative decoding: the draft context (an MTP context on the weights of the main model, or a draft model) is created from the
    // same params, so it runs at the same n_ubatch: its compute buffers and KV cache are counted too, as for --fit (a failed measure of it
    // keeps the default n_ubatch)
    // (a draft model that --fit left out, fit_off, runs no context)
    const bool has_draft = params.speculative.has_dft() && !params.speculative.draft.fit_off;
    const bool spec_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    // k9-draft: n-gram drafts (no draft context, verify batches of a few tokens) leave the prompt rules as without speculative decoding:
    // the k6 spec gate was measured with model drafters (the decode after a lent prompt with an MTP head lost 5.5-6%), and with it
    // ngram-mod's prompts lost 68-82% (LFM2.5 at 8 GB, plans/k9_vs_competitors.md); LLAMA_MOE_SPILL_LEND_NGRAM=0 restores the gate
    static const bool lend_ngram = getenv("LLAMA_MOE_SPILL_LEND_NGRAM") == nullptr || atoi(getenv("LLAMA_MOE_SPILL_LEND_NGRAM")) != 0;
    const bool has_spec  = has_draft || std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t != COMMON_SPECULATIVE_TYPE_NONE && (!lend_ngram || t == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE ||
            t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
            t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK); });
    common_params        params_dft;
    llama_model_params   mparams_dft  = mparams;
    llama_context_params cparams_dft  = cparams;
    if (has_draft || spec_mtp) {
        params_dft  = common_base_params_to_speculative(params);
        mparams_dft = has_draft ? common_model_params_to_llama(params_dft) : mparams;
        cparams_dft = common_context_params_to_llama(params_dft);
        if (spec_mtp) {
            cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        }
        cparams_dft.n_rs_seq = 0;
    }
    const auto measure_dft = [&](uint32_t u, int64_t & ctx_compute) -> bool {
        ctx_compute = 0;
        if (!(has_draft || spec_mtp)) {
            return true;
        }
        llama_context_params cp = cparams_dft;
        cp.n_ctx    = cparams.n_ctx;
        cp.n_ubatch = common_draft_n_ubatch(params, u);
        cp.n_batch  = std::max(cp.n_batch, u);
        // a separate MTP drafter reads the prompt in ubatches of at most common_draft_ubatch_max() (see common_base_params_to_speculative)
        if (has_draft && spec_mtp && common_draft_ubatch_max() > 0) {
            cp.n_ubatch = std::min(cp.n_ubatch, common_draft_ubatch_max());
        }
        // an explicit draft ubatch wins, as in common_base_params_to_speculative
        if (params.speculative.draft.n_ubatch > 0) {
            cp.n_ubatch = (uint32_t) params.speculative.draft.n_ubatch;
        }
        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;
        try {
            common_draft_measure_env split_env(has_draft); // an MTP context on the main model's weights: unchanged
            const common_device_memory_data_vec dmd = common_get_device_memory_data(has_draft ? params_dft.model.path.c_str() : params.model.path.c_str(),
                &mparams_dft, &cp, devs, hp_ngl, hp_nct, hp_nex, params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
            for (size_t i = 0; i + 1 < dmd.size(); ++i) {
                ctx_compute += (int64_t) (dmd[i].context + dmd[i].compute);
            }
            return true;
        } catch (const std::exception & e) {
            COM_WRN("memory measure of the draft context at n_ubatch = %u failed: %s\n", u, e.what());
            return false;
        }
    };
    // the main context at ubatch u with LLAMA_MOE_MEASURE_LENT = mode ("1" lent, "" with the staging copies, "auto:<bytes>" see below)
    const auto measure_main = [&](uint32_t u, const std::string & mode) -> mem {
        if (!had_lent_env) {
            common_set_env("LLAMA_MOE_MEASURE_LENT", mode);
        }
        llama_context_params cp = cparams;
        cp.n_ubatch = u;
        cp.n_batch  = std::max(cp.n_batch, u);
        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;
        try {
            const common_device_memory_data_vec dmd = common_get_device_memory_data(params.model.path.c_str(), &mparams, &cp, devs, hp_ngl, hp_nct, hp_nex,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
            if (dmd.size() < 2) {
                return { -1, -1, 0 }; // no device, nothing is cached
            }
            mem m = { 0, 0, (int) dmd.size() - 1 };
            for (size_t i = 0; i + 1 < dmd.size(); ++i) {
                m.compute += (int64_t) dmd[i].compute;
                m.total   += (int64_t) (dmd[i].context + dmd[i].compute);
            }
            return m;
        } catch (const std::exception & e) {
            COM_WRN("memory measure at n_ubatch = %u failed: %s\n", u, e.what());
            return { -1, -1, 0 };
        }
    };
    const char * env_lend = getenv("LLAMA_MOE_CACHE_LEND");
    double lend_min_frac = 0.08;
    if (const char * e = getenv("LLAMA_MOE_CACHE_LEND_MIN_FRAC")) {
        lend_min_frac = std::max(0.0, atof(e));
    }
    // the staging copies are left out only if the cache will lend them: LLAMA_MOE_CACHE_LEND=1/2, or unset and the copies are at least
    // LLAMA_MOE_CACHE_LEND_MIN_FRAC of the cache, decided in the measured context from its own staging copies (auto:<bytes>). A decision
    // from the difference of a measure with and one without the copies failed: only the lent one counts the seed graph, so the copies
    // looked like 0 MiB (gemma-4 26B-A4B, 8 GB, n_ubatch 512: 187 MiB with the copies, 520 lent) and a cache lending 22% of itself was
    // measured as one that does not, 280 MiB more than the device needs were taken from it (decode -7%)
    const std::string lent_mode = env_lend != nullptr ? (atoi(env_lend) != 0 ? "1" : "") :
        "auto:" + std::to_string((uint64_t) (lend_min_frac*(double) params.moe_cache_size));
    const auto measure = [&](uint32_t u) -> mem {
        mem m = measure_main(u, lent_mode);
        if (m.total < 0) {
            return m;
        }
        // the draft context, with its staging copies (an MTP context lends little: its own cache holds the experts of one layer): counted
        // in the total (the growth taken from the cache), not in the compute of the main context
        if (!had_lent_env) {
            common_set_env("LLAMA_MOE_MEASURE_LENT", "");
        }
        int64_t dft = 0;
        if (!measure_dft(u, dft)) {
            return { -1, -1, 0 };
        }
        m.total += dft;
        m.dft    = dft;
        return m;
    };
    struct lent_env_reset {
        bool had;
        ~lent_env_reset() {
            if (!had) {
                common_set_env("LLAMA_MOE_MEASURE_LENT", "");
            }
        }
    } lent_env_guard = { had_lent_env };

    const int64_t t0    = ggml_time_us();
    const mem m_def = measure(ub_def);
    const mem m_ref = measure(cands[0]);
    if (m_def.total < 0 || m_ref.total < 0) {
        return;
    }
    const int64_t cache     = (int64_t) params.moe_cache_size;
    const int64_t cache_ref = cache - std::max<int64_t>(0, m_ref.total - m_def.total);

    // tiny cache: the cache holds the experts of fewer than LLAMA_MOE_PREFILL_UB_TINY_TOKENS (4, the minimum of the RAM tier,
    // LLAMA_MOE_HOST_MIN_TOKENS) tokens of a model whose experts spill. A prompt ubatch then uploads every expert of every layer from the
    // host (none from the cache) and decode misses most experts anyway, so the cache buys little, while every ubatch reads all the experts
    // again: from disk when they do not fit in RAM. Qwen3.8-Flash-Next UD-IQ1_S on a T4 at 8 GB (31 GB RAM, the experts of 1.6 tokens on
    // the device): an 8k prompt at n_ubatch 512 read 594 GB in 16 ubatches and ran 21% slower than the base, which keeps 6 whole layers
    // on the device. Here n_ubatch grows (1024 ... LLAMA_MOE_PREFILL_UB_MAX) while the cache keeps at least
    // LLAMA_MOE_PREFILL_UB_TINY_KEEP (0.25) of itself and the KV cache does not grow. LLAMA_MOE_PREFILL_UB_TINY=0 off, =N forces N here
    {
        common_set_env("LLAMA_MOE_TINY_RELEASE", ""); // set below for the context this rule sizes (k19-tiny-ub)
        common_moe_slot_bytes sbt;
        const char * env_t   = getenv("LLAMA_MOE_PREFILL_UB_TINY");
        const int    force_t = env_t != nullptr ? atoi(env_t) : -1;
        double tiny_tokens = 4.0;
        if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_TINY_TOKENS")) {
            tiny_tokens = std::max(0.0, atof(e));
        }
        double keep = 0.25;
        if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_TINY_KEEP")) {
            keep = std::min(1.0, std::max(0.0, atof(e)));
        }
        // k13-small: not beside the small-device placement (common_fit_small_device), whose 256 MiB margin has no room for the larger
        // ubatch's run-time buffers: G4 4 GB q3m without a draft grew to n_ubatch 2048 (compute 804 MiB) and ran out of device memory
        // in a request (CUDA error); LLAMA_MOE_PREFILL_UB_TINY=N still forces N
        const bool small_dev = common_fit_small_device() && !common_fit_hostbound() && force_t <= 0;
        if (force_t != 0 && !small_dev && force <= 0 && !rule_frac && common_moe_slot_bytes_of(params.model.path, sbt) && sbt.n_used > 0 && sbt.n_layer > 0) {
            const double  token_bytes = (double) sbt.n_layer*(double) sbt.n_used*(double) sbt.sum;
            const double  tokens      = (double) cache/token_bytes;
            const int64_t all_exp     = (int64_t) (1.02*(double) sbt.total) + sbt.n_layer*sbt.sum;
            if (tokens < tiny_tokens && cache < all_exp) {
                std::vector<uint32_t> ct = { 1024, 2048, 4096, 8192 };
                if (force_t > 0) {
                    ct = { (uint32_t) force_t };
                }
                ct.erase(std::remove_if(ct.begin(), ct.end(), [&](uint32_t u) { return u <= ub_def || u > ub_lim || (force_t <= 0 && u > ub_max); }), ct.end());
                uint32_t pick_t = ub_def;
                mem      m_t    = m_def;
                const auto kv_t = [](const mem & x) { return x.total - x.compute - x.dft; };
                for (uint32_t u : ct) {
                    const mem m = u == cands[0] ? m_ref : measure(u);
                    if (m.total < 0 || kv_t(m) > kv_t(m_def) + (1 << 20) || (double) (cache - (m.total - m_def.total)) < keep*(double) cache) {
                        break;
                    }
                    pick_t = u;
                    m_t    = m;
                }
                if (pick_t != ub_def) {
                    const int64_t taken_t = std::max<int64_t>(0, m_t.total - m_def.total);
                    // k19-tiny-ub (LLAMA_MOE_PREFILL_UB_TINY_RELEASE=0: the growth is taken from the cache for good, as before): the larger
                    // n_ubatch is for prompts only, so the cache keeps its size (less any KV growth, none by the rule above) and frees its
                    // banks while ubatches of more than the default n_ubatch run; the next ubatch of at most that many tokens frees the
                    // compute buffers (also an MTP drafter's) and allocates and refills the banks again with the experts they held
                    // (llama_context::tiny_rel_enter / tiny_rel_leave). Decode then reads the cache of the default n_ubatch with the prompt
                    // of the larger one. T4 6 GB, Qwen3.6-35B-A3B + MTP: the rule took 576 / 974 MiB of 1141 / 1398 MiB (q4x / q3m) and kept
                    // them through decode (chat decode 33.5 / 38.6 tok/s, released: 38.6 / 50.2). Single device, no separate draft model
                    // (its compute buffers are not freed)
                    // Only where the whole cache holds the experts of at least LLAMA_MOE_PREFILL_UB_TINY_RELEASE_MIN_TOKENS (1.5) tokens: below
                    // that the decode misses most experts of a token with or without the growth, so it gains little, while every prompt
                    // refills the whole cache after it and stages from the host the experts the smaller cache would have copied on the
                    // device (L4 LFM2.5-24B-A2B 4 GB, 1.2 tokens: e2e chat / 2k / 8k 1.04 / 1.03 / 0.96 of the rule before; T4 Qwen3.6-35B-A3B
                    // + MTP 6 GB, 1.6 / 2.3 tokens (q4x / q3m): 1.14 / 1.15 / 1.08 and 1.27 / 1.20 / 1.11)
                    static const bool tiny_rel_env = getenv("LLAMA_MOE_PREFILL_UB_TINY_RELEASE") == nullptr || atoi(getenv("LLAMA_MOE_PREFILL_UB_TINY_RELEASE")) != 0;
                    double tiny_rel_min_tokens = 1.5;
                    if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_TINY_RELEASE_MIN_TOKENS")) {
                        tiny_rel_min_tokens = std::max(0.0, atof(e));
                    }
                    // and only while the host keeps the experts the refill reads (as the spill lending: the host experts at most
                    // LLAMA_MOE_SPILL_LEND_MAX_RAM_FRAC (0.5) of the RAM, Linux; not with --load-mode none, which streams them from the files)
                    bool tiny_rel_ram = params.load_mode != LLAMA_LOAD_MODE_NONE;
#ifdef __linux__
                    if (tiny_rel_ram) {
                        double ram_frac = 0.5;
                        if (const char * e = getenv("LLAMA_MOE_SPILL_LEND_MAX_RAM_FRAC")) {
                            ram_frac = std::max(0.0, atof(e));
                        }
                        if (FILE * fm = fopen("/proc/meminfo", "r")) {
                            char line[256];
                            long long total_kb = 0;
                            while (fgets(line, sizeof(line), fm)) {
                                if (sscanf(line, "MemTotal: %lld kB", &total_kb) == 1) {
                                    break;
                                }
                            }
                            fclose(fm);
                            tiny_rel_ram = total_kb <= 0 || std::max(0.0, (double) sbt.total - (double) cache) <= ram_frac*(double) total_kb*1024.0;
                            // k19-smallm (LLAMA_MOE_TINY_RELEASE_RAM_FIT=0: the RAM share alone, as before): also where the whole model
                            // file fits in the available memory with LLAMA_MOE_TINY_RELEASE_RAM_RESERVE (0.10) of the RAM to spare. The
                            // refill reads only the cache's experts, from a page cache that then keeps the whole file. Kaggle T4 (4 vCPU,
                            // 31.3 GB RAM, 29.8 GB available), Qwen3.6-35B-A3B UD-Q4_K_XL + MTP at 6 GB: 19.2 GB of host experts are 61%
                            // of the RAM, so the share kept the old rule (cache 1139 -> 549 MiB, decode hit rate 25.7%) though the 21.8 GB
                            // file stayed resident (0 MB read from the disk during the requests)
                            static const bool ram_fit_env = getenv("LLAMA_MOE_TINY_RELEASE_RAM_FIT") == nullptr ||
                                atoi(getenv("LLAMA_MOE_TINY_RELEASE_RAM_FIT")) != 0;
                            if (!tiny_rel_ram && ram_fit_env && total_kb > 0) {
                                double reserve = 0.10;
                                if (const char * e = getenv("LLAMA_MOE_TINY_RELEASE_RAM_RESERVE")) {
                                    reserve = std::max(0.0, atof(e));
                                }
                                const uint64_t file  = common_model_file_bytes(params.model.path);
                                const uint64_t avail = common_available_memory();
                                const bool tiny_rel_fit = file > 0 && avail > 0 && !common_model_file_in_ram(params.model.path) &&
                                    (double) file + reserve*(double) total_kb*1024.0 <= (double) avail;
                                tiny_rel_ram = tiny_rel_fit;
                                if (tiny_rel_fit) {
                                    COM_INF("MoE cache: tiny-cache release: the host experts (%.0f MiB) are more than %.0f%% of the RAM, but the model "
                                        "file (%.0f MiB) fits in the available memory (%.0f MiB) with %.0f%% of the RAM to spare "
                                        "(LLAMA_MOE_TINY_RELEASE_RAM_FIT=0: the RAM share only)\n", std::max(0.0, (double) sbt.total - (double) cache)/1048576.0,
                                        100.0*ram_frac, file/1048576.0, avail/1048576.0, 100.0*reserve);
                                }
                            }
                        }
                    }
#endif
                    const int64_t kv_grow  = std::max<int64_t>(0, kv_t(m_t) - kv_t(m_def));
                    // k19-l4poly: a host-bound small device always releases (its 384 MiB margin has no room for the growth beside the
                    // cache; with k19-place8 the context releases the banks for every larger ubatch anyway)
                    const bool    tiny_rel = (common_fit_hostbound() && m_t.n_dev == 1 && m_def.n_dev == 1 && !has_draft) ||
                        (tiny_rel_env && m_t.n_dev == 1 && m_def.n_dev == 1 && !has_draft && tokens >= tiny_rel_min_tokens && tiny_rel_ram);
                    const int64_t cache_t  = cache - (tiny_rel ? kv_grow : taken_t);
                    if (tiny_rel) {
                        COM_INF("MoE cache: n_ubatch = %u (default %u, rule: tiny cache, the experts of %.1f tokens < %.1f, experts spill): +%.2f MiB of compute "
                            "and KV memory while a ubatch of more than %u tokens runs, the cache banks are released for it and refilled after it, the cache keeps "
                            "%.2f of %.2f MiB (LLAMA_MOE_PREFILL_UB_TINY_RELEASE=0: taken from the cache for good; LLAMA_MOE_PREFILL_UB_TINY=0 to disable)\n",
                            pick_t, ub_def, tokens, tiny_tokens, taken_t/1048576.0, ub_def, cache_t/1048576.0, cache/1048576.0);
                        common_set_env("LLAMA_MOE_TINY_RELEASE", std::to_string(ub_def) + ":" + std::to_string(pick_t) + ":" + std::to_string((uint64_t) cache_t));
                    } else {
                        COM_INF("MoE cache: n_ubatch = %u (default %u, rule: tiny cache, the experts of %.1f tokens < %.1f, experts spill): +%.2f MiB of compute "
                            "and KV memory taken from the cache, %.2f -> %.2f MiB (the experts of %.1f tokens; LLAMA_MOE_PREFILL_UB_TINY=0 to disable)\n",
                            pick_t, ub_def, tokens, tiny_tokens, taken_t/1048576.0, cache/1048576.0, (cache - taken_t)/1048576.0, (cache - taken_t)/token_bytes);
                    }
                    params.n_ubatch  = (int32_t) pick_t;
                    cparams.n_ubatch = pick_t;
                    if (!params.n_batch_set) {
                        params.n_batch  = std::max<int32_t>(params.n_batch, (int32_t) pick_t);
                        cparams.n_batch = std::max<uint32_t>(cparams.n_batch, pick_t);
                    }
                    params.moe_cache_size  = (size_t) cache_t;
                    cparams.moe_cache_size = (size_t) cache_t;
                    return;
                }
                COM_INF("MoE cache: tiny cache (the experts of %.1f tokens < %.1f) but no larger n_ubatch keeps %.0f%% of it, n_ubatch stays %u\n",
                    tokens, tiny_tokens, 100.0*keep, ub_def);
            }
        }
    }
    if (cache_ref < cache/2) {
        COM_INF("MoE cache: n_ubatch stays %u, n_ubatch %u would take %.2f of %.2f MiB from the cache\n", ub_def, cands[0],
            (m_ref.total - m_def.total)/1048576.0, cache/1048576.0);
        return;
    }
    const auto kv = [](const mem & m) { return m.total - m.compute - m.dft; };

    // LLAMA_MOE_CACHE_LEND=2: the graph allocation of a prompt ubatch is lent from the tail of one bank of the cache and given back for
    // decode, so the cache gives up only the KV growth; a ubatch is taken if its compute buffers fit in one bank tail (every slot but one
    // layer's experts, see llama_moe_cache lend) with LLAMA_MOE_PREFILL_UB_LEND_FRAC (0.85) of it, a larger one would not be lent and
    // would need its own buffer next to the cache (single device only: the cache does not lend with several devices)
    const char * env_l = getenv("LLAMA_MOE_CACHE_LEND");
    common_moe_slot_bytes sb;
    const bool have_sb = common_moe_slot_bytes_of(params.model.path, sb);
    // the cache holds every expert: all the expert tensors of the model, a slot per layer of slack (the extra slot of each bank, the
    // device lookup state) and 2%
    const int64_t all_experts = have_sb ? (int64_t) (1.02*(double) sb.total) + sb.n_layer*sb.sum : INT64_MAX;
    // spilled experts, LLAMA_MOE_CACHE_LEND unset: the graph allocation is lent as with LLAMA_MOE_CACHE_LEND=2 (LLAMA_MOE_SPILL_LEND=0
    // disables). A larger n_ubatch then costs the decode no cache (its compute buffers are taken from the cache only while a prompt
    // ubatch runs), unlike LLAMA_MOE_PREFILL_UB_SPILL=1, which shrinks the cache for good. G4 (RTX PRO 6000) at 32 GB, Qwen3.8-Flash-Next
    // UD-IQ4_XS, 2 reps: 8k / 16k prompt 1933 / 1931 -> 3583 / 4247 tok/s (3927 / 4252 with 3 splits of prefetch), decode 81.7 / 81.1 -> 84.2 / 82.3
    // (not with the opt-in LLAMA_MOE_PREFILL_UB_SPILL=1 rule). Only while the cache holds at most LLAMA_MOE_SPILL_LEND_MAX_FRAC (0.5) of
    // the experts: a cache holding most of them already makes a small share of each prompt pass, and its decode hit rate is sensitive
    // to the cache state a prompt leaves; there lending cost the decode after a single-ubatch prompt (Qwen3.6-35B-A3B UD-Q4_K_XL at 16 GB,
    // 66% cached, G4: decode after a 2k prompt 214.3 -> 202.3 tok/s, hit rate 96.9 -> 96.1%), while at 12 GB (44%) it was within 0.3%
    double spill_lend_max_frac = 0.5;
    if (const char * e = getenv("LLAMA_MOE_SPILL_LEND_MAX_FRAC")) {
        spill_lend_max_frac = std::max(0.0, atof(e));
    }
    // Not with speculative decoding unless LLAMA_MOE_SPILL_LEND=2: with an MTP draft (Flash-Next + MTP Q8_0 at 32 GB, G4) the prompt
    // gained +13% / +94% / +107% (2k / 8k / 16k) but the decode after it lost 4.5% / 9.3% / 0% (the cache is 18% smaller there and the
    // graph lends 29% of it during a prompt)
    const int  spill_lend_mode = getenv("LLAMA_MOE_SPILL_LEND") != nullptr ? atoi(getenv("LLAMA_MOE_SPILL_LEND")) : 1;
    // and only while the experts left on the host are at most LLAMA_MOE_SPILL_LEND_MAX_RAM_FRAC (0.5) of the RAM (Linux): the slots lent to a
    // prompt graph are refilled from the host experts after it, which on a host whose page cache cannot keep them reads them from the
    // disk again (L4 24 GB, 52 GB RAM, Flash-Next UD-IQ4_XS from the VM disk: decode after an 8k prompt 12.5 -> 7.6 tok/s)
    bool spill_lend_ram = true;
#ifdef __linux__
    {
        double ram_frac = 0.5;
        if (const char * e = getenv("LLAMA_MOE_SPILL_LEND_MAX_RAM_FRAC")) {
            ram_frac = std::max(0.0, atof(e));
        }
        if (FILE * fm = fopen("/proc/meminfo", "r")) {
            char line[256];
            long long total_kb = 0;
            while (fgets(line, sizeof(line), fm)) {
                if (sscanf(line, "MemTotal: %lld kB", &total_kb) == 1) {
                    break;
                }
            }
            fclose(fm);
            const double host_experts = std::max(0.0, (double) sb.total - (double) cache_ref);
            spill_lend_ram = total_kb <= 0 || host_experts <= ram_frac*(double) total_kb*1024.0;
        }
    }
#endif
    // strata-gap: with speculative decoding too (LLAMA_MOE_SPILL_LEND_SPEC=0 restores the k6 gate). B200 ballasted to 32 GB, Flash-Next
    // UD-IQ4_XS + Q4_K_M MTP drafter: the spec gate left the prompt at n_ubatch 2048, four expert passes per 8k prompt (169 GB H2D vs
    // Strata's 41 GB, 8k prompt 1630 vs 4566 tok/s); see plans/strata_gap_profile.md
    // LLAMA_MOE_SPILL_LEND_SPEC unset: with a separate drafter (-md) only; 0: never (the k6 gate); 1: with any draft. A built-in MTP head
    // (Qwen3.6-35B-A3B UD-Q4_K_XL on G4 at 12 / 16 GB, where k6 keeps n_ubatch 512) lost 5.5-6% of the decode after a lent 2k prompt (on a
    // different text: n_ubatch rounding; no seeding variant recovered it) for a 2.8-2.9x 2k prompt, so it stays off there by default;
    // Flash-Next + its Q4_K_M drafter at 32 GB (k6 at 2048): 2k decode 84.7 -> 87.0 tok/s with the auto pinned experts, same text
    static const int spill_lend_spec_mode = getenv("LLAMA_MOE_SPILL_LEND_SPEC") != nullptr ? atoi(getenv("LLAMA_MOE_SPILL_LEND_SPEC")) : -1;
    const bool spill_lend_spec = spill_lend_spec_mode > 0 || (spill_lend_spec_mode < 0 && has_draft);
    const bool spill_lend = env_l == nullptr && !rule_frac && have_sb && cache_ref < all_experts &&
        (double) cache_ref <= spill_lend_max_frac*(double) sb.total &&
        (spill_lend_mode >= 2 || (spill_lend_mode == 1 && (!has_spec || spill_lend_spec) && spill_lend_ram)) &&
        !(getenv("LLAMA_MOE_PREFILL_UB_SPILL") != nullptr && atoi(getenv("LLAMA_MOE_PREFILL_UB_SPILL")) != 0);
    bool lend_graph = ((env_l != nullptr && atoi(env_l) == 2) || spill_lend) && force <= 0 && m_def.n_dev == 1 && have_sb;
    double lend_frac = 0.85;
    if (const char * env_lf = getenv("LLAMA_MOE_PREFILL_UB_LEND_FRAC")) {
        lend_frac = std::max(0.0, atof(env_lf));
    }
    const auto lend_fits = [&](const mem & m) {
        const int64_t c       = cache - std::max<int64_t>(0, kv(m) - kv(m_def) + (m.dft - m_def.dft)); // the cache without the KV (and draft context) growth
        // in the bank tail of the largest group of layers with the same expert sizes, from its share of the cache (a model with
        // mixed expert types, e.g. Qwen3.8-Flash-Next UD-IQ4_XS, lends from a group with smaller slots than its largest layer: an
        // estimate from the largest layer took an n_ubatch 8192 graph of all-token outputs (llama-perplexity, 8770 MiB) that the
        // cache could not lend, and the context failed to allocate it; LLAMA_MOE_LEND_FIT_LARGEST=1 restores that estimate)
        static const bool largest = getenv("LLAMA_MOE_LEND_FIT_LARGEST") != nullptr && atoi(getenv("LLAMA_MOE_LEND_FIT_LARGEST")) != 0;
        const bool    grp     = !largest && sb.lend_sum > 0 && sb.lend_max > 0;
        const int64_t c_g     = grp ? (int64_t) ((double) c*sb.lend_share) : c;
        const int64_t n_slots = c_g / (grp ? sb.lend_sum : sb.sum);
        const int64_t n_keep  = std::min<int64_t>(n_slots / 2, sb.n_expert + 1);
        const int64_t tail    = (n_slots - n_keep) * (grp ? sb.lend_max : sb.max);
        return (double) m.compute <= lend_frac * (double) tail;
    };
    if (lend_graph) {
        COM_INF("MoE cache: graph lending group: slot %.2f MiB, largest tensor %.2f MiB, %.1f%% of the expert bytes (largest layer: slot %.2f MiB, "
            "largest tensor %.2f MiB)\n", sb.lend_sum/1048576.0, sb.lend_max/1048576.0, 100.0*sb.lend_share, sb.sum/1048576.0, sb.max/1048576.0);
    }
    if (lend_graph && !lend_fits(m_ref)) {
        lend_graph = false;
    }
    // where the rule below (fix 2) would keep the default n_ubatch, because the growth of the reference n_ubatch is more than
    // LLAMA_MOE_PREFILL_UB_SHRINK_MAX_FRAC of a spilling cache that holds the experts of at least LLAMA_MOE_PREFILL_UB_TINY_TOKENS tokens,
    // the compute buffers of the larger n_ubatch are lent from the cache instead (the LLAMA_MOE_CACHE_LEND=2 behaviour): the cache gives
    // up only the KV growth, so decode keeps the cache fix 2 kept, and prompts keep the larger ubatch the k5 rule gave them. Measured with
    // LLAMA_MOE_CACHE_LEND=2 forced (decode / 2k prompt, k6 -> lent): gpt-oss-20B Q2_K at 8 GB, G4 183.3 / 5262 -> 172.8 / 11846, L4
    // 58.3 / 1197 -> 55.1 / 2856, T4 38.0 / 891 -> 35.9 / 1211; Qwen3.6-35B-A3B UD-Q4_K_XL at 8 GB, T4 33.8 / 361 -> 33.5 / 753.
    // Only with the conditions of the spill lending (single device, the graph fits a bank tail, no speculative decoding, the host experts
    // at most LLAMA_MOE_SPILL_LEND_MAX_RAM_FRAC of the RAM); a host whose RAM fails that gate takes the growth from the cache as the k5 rule
    // does (the slots lent to a prompt would be refilled from the disk). LLAMA_MOE_PREFILL_UB_SHRINK_LEND=0: fix 2 as before (n_ubatch
    // stays the default); =2: lend also past the RAM gate and with speculative decoding. LLAMA_MOE_PREFILL_UB_SHRINK=1: the k5 rule.
    bool fix2_lend = false, fix2_take = false, fix2_cond = false;
    const int f2_mode = getenv("LLAMA_MOE_PREFILL_UB_SHRINK_LEND") != nullptr ? atoi(getenv("LLAMA_MOE_PREFILL_UB_SHRINK_LEND")) : 1;
    {
        static const bool shrink_ok = getenv("LLAMA_MOE_PREFILL_UB_SHRINK") != nullptr && atoi(getenv("LLAMA_MOE_PREFILL_UB_SHRINK")) != 0;
        double shrink_max = 0.05;
        if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_SHRINK_MAX_FRAC")) {
            shrink_max = std::max(0.0, atof(e));
        }
        double tiny_tokens = 4.0;
        if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_TINY_TOKENS")) {
            tiny_tokens = std::max(0.0, atof(e));
        }
        const bool rule_spill_env = getenv("LLAMA_MOE_PREFILL_UB_SPILL") != nullptr && atoi(getenv("LLAMA_MOE_PREFILL_UB_SPILL")) != 0;
        if (f2_mode != 0 && !shrink_ok && !rule_frac && !rule_spill_env && force <= 0 && env_l == nullptr && have_sb &&
            sb.n_layer > 0 && sb.n_used > 0) {
            const double  tokens = (double) cache/((double) sb.n_layer*(double) sb.n_used*(double) sb.sum);
            const int64_t growth = std::max<int64_t>(0, m_ref.total - m_def.total);
            fix2_cond = (double) growth > shrink_max*(double) cache && cache - growth < all_experts && tokens >= tiny_tokens;
            // with the spill lending on, the pick below decides (it keeps lending at the reference n_ubatch, see there)
            if (fix2_cond && !lend_graph) {
                const bool gate_ok = f2_mode >= 2 || ((!has_spec || spill_lend_spec) && spill_lend_ram); // strata-gap: with a draft too (LLAMA_MOE_SPILL_LEND_SPEC=0: as k6)
                if (gate_ok && m_def.n_dev == 1 && lend_fits(m_ref)) {
                    fix2_lend  = true;
                    lend_graph = true;
                } else if (f2_mode < 2 && !has_spec && !spill_lend_ram) {
                    fix2_take = true;
                }
                COM_INF("MoE cache: n_ubatch %u would take %.2f MiB (%.1f%%) from a cache of %.2f MiB (the experts of %.1f tokens) that does not "
                    "hold every expert: %s (LLAMA_MOE_PREFILL_UB_SHRINK_LEND=0 keeps n_ubatch %u)\n", cands[0], growth/1048576.0,
                    100.0*growth/(double) cache, cache/1048576.0, tokens, fix2_lend ? "its compute buffers are lent from the cache" :
                    fix2_take ? "taken from the cache (the host experts are more than LLAMA_MOE_SPILL_LEND_MAX_RAM_FRAC of the RAM)" :
                    "n_ubatch stays the default", ub_def);
            }
        }
    }
    // spilled experts: the cache cannot hold every expert of the model even at the reference n_ubatch. A prompt ubatch of at least
    // GGML_SCHED_PREFETCH_MIN_BATCH tokens stages every expert of every cached layer (the copies run ahead of the routing), so each ubatch
    // uploads all the experts the cache misses: a prompt of P tokens uploads them ceil(P/n_ubatch) times. A larger n_ubatch uploads them once
    // per n_ubatch tokens (an 8k prompt at 8192 instead of 4 times at 2048), so n_ubatch grows up to LLAMA_MOE_PREFILL_UB_MAX while the memory
    // it takes from the cache is at most LLAMA_MOE_PREFILL_UB_SPILL_FRAC (0.15) of the cache and the KV cache does not grow (decode then reads a
    // smaller cache). A n_ubatch >= LLAMA_MOE_CACHE_SEED_AUTO (4096) also skips the own-ids seed of each prompt (see llama_context).
    // OPT-IN, LLAMA_MOE_PREFILL_UB_SPILL=1 (G4 RTX PRO 6000, 3 reps + A/A): Qwen3.8-Flash-Next UD-IQ4_XS at 32 GB 8k prompt +78% over pinned
    // experts alone (2610 -> 4641 tok/s), decode -2%; but Qwen3.6-35B-A3B UD-Q4_K_XL at 12 / 16 GB decode -5..-7% after a 2k prompt (the
    // smaller cache and the skipped seed), so it is not the default. Unset or 0: the rule before (n_ubatch grows only while every expert
    // stays cached)
    bool rule_spill = !rule_frac && !lend_graph && force <= 0 && have_sb && cache_ref < all_experts;
    double spill_frac = 0.15;
    {
        const char * env_sp = getenv("LLAMA_MOE_PREFILL_UB_SPILL");
        rule_spill = rule_spill && env_sp != nullptr && atoi(env_sp) != 0;
    }
    if (const char * env_sf = getenv("LLAMA_MOE_PREFILL_UB_SPILL_FRAC")) {
        spill_frac = std::max(0.0, atof(env_sf));
    }
    const char * rule = force > 0 ? "forced" : lend_graph ? (spill_lend ? "experts spill, compute buffers lent from the cache (LLAMA_MOE_SPILL_LEND=0 to disable)" :
        fix2_lend ? "the growth would shrink a spilling cache, compute buffers lent from the cache (LLAMA_MOE_PREFILL_UB_SHRINK_LEND=0 to disable)" :
        "LLAMA_MOE_CACHE_LEND=2") : rule_frac ? "LLAMA_MOE_PREFILL_UB_AUTO=1" : fix2_take ? "the growth taken from the cache (k5 rule, RAM gate)" :
        rule_spill ? "experts spill, at most LLAMA_MOE_PREFILL_UB_SPILL_FRAC of the cache, no KV growth" : "every expert cached, no KV growth";

    uint32_t pick   = cands[0];
    mem      m_pick = m_ref;
    for (size_t i = 1; i < cands.size(); ++i) {
        const mem m = measure(cands[i]);
        if (m.total < 0) {
            break;
        }
        const bool kv_grows = kv(m) > kv(m_ref) + (1 << 20);
        if (lend_graph || rule_frac) {
            if (lend_graph ? (!lend_fits(m) || cache - (kv(m) - kv(m_def)) < cache/2)
                           : ((double) (m.compute - m_ref.compute) > frac*(double) cache_ref || cache - (m.total - m_def.total) < cache/2)) {
                break;
            }
            if (force <= 0 && kv_grows && cands[i] > ub_max_kv) {
                break;
            }
        } else if (rule_spill) {
            if (kv_grows || (double) (m.total - m_def.total) > spill_frac*(double) cache || cache - (m.total - m_def.total) < cache/2) {
                break;
            }
        } else if (kv_grows || cache - (m.total - m_def.total) < all_experts) {
            break;
        }
        pick   = cands[i];
        m_pick = m;
    }
    // the spill lending only pays with a larger n_ubatch than the reference one: else the rule before (the growth taken from the cache)
    if (spill_lend && lend_graph && pick == cands[0]) {
        lend_graph = false;
        rule = "every expert cached, no KV growth";
    }
    // the fix 2 case: lending pays only with a larger n_ubatch than the reference one. At the reference n_ubatch the growth is taken from
    // the cache as the k5 rule does (T4 at 8 GB, Qwen3.6-35B-A3B UD-Q4_K_XL, n_ubatch 2048, decode / 2k prompt: lent 32.55 / 722 and
    // 33.5 / 753, taken 32.5 / 815 and k5 32.85 / 786; fix 2 itself 32.5 / 358)
    if (fix2_cond && pick == cands[0] && (fix2_lend || (spill_lend && !lend_graph))) {
        fix2_lend  = false;
        lend_graph = false;
        fix2_take  = true;
        rule = "the growth taken from the cache (k5 rule: lending gives no larger n_ubatch)";
        COM_INF("MoE cache: lending gives no n_ubatch above %u: its growth is taken from the cache (LLAMA_MOE_PREFILL_UB_SHRINK_LEND=0 keeps "
            "n_ubatch %u)\n", cands[0], ub_def);
    }
    // the reference n_ubatch takes its compute (and KV) growth from the cache for good. Where the cache then no longer holds every
    // expert, every decode step reads the smaller cache: gpt-oss-20B Q2_K at an 8 GB cap (G4), 5347 -> 4984 MiB (6.8%): device hit rate
    // 87.8% and decode 142 tok/s (k5, n_ubatch 2048) vs 90.3% and 189 tok/s (512), for 2k prompts at 5258 instead of 9900 tok/s. So n_ubatch
    // stays the default when the growth is more than LLAMA_MOE_PREFILL_UB_SHRINK_MAX_FRAC (0.05) of the cache, unless the cache is tiny (the
    // experts of fewer than LLAMA_MOE_PREFILL_UB_TINY_TOKENS = 4 tokens): there the prompt ubatches of 512 tokens admit more experts into the
    // cache than the decode keeps, and both lose (gpt-oss-20B at 6 GB, the experts of 2.7 tokens: decode 84.5 -> 66.2 tok/s, hit rate
    // 68.6 -> 60.8%, 2k prompt 7700 -> 3594; lending the growth instead: 65.9 tok/s, the refilled slots miss). LLAMA_MOE_PREFILL_UB_SHRINK=1
    // takes the growth from the cache in every case
    if (!lend_graph && !fix2_take && !rule_frac && !rule_spill && force <= 0 && pick == cands[0] && have_sb && sb.n_layer > 0 && sb.n_used > 0) {
        static const bool shrink_ok = getenv("LLAMA_MOE_PREFILL_UB_SHRINK") != nullptr && atoi(getenv("LLAMA_MOE_PREFILL_UB_SHRINK")) != 0;
        double shrink_max = 0.05;
        if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_SHRINK_MAX_FRAC")) {
            shrink_max = std::max(0.0, atof(e));
        }
        double tiny_tokens = 4.0;
        if (const char * e = getenv("LLAMA_MOE_PREFILL_UB_TINY_TOKENS")) {
            tiny_tokens = std::max(0.0, atof(e));
        }
        const double  tokens = (double) cache/((double) sb.n_layer*(double) sb.n_used*(double) sb.sum);
        const int64_t growth = std::max<int64_t>(0, m_pick.total - m_def.total);
        if (!shrink_ok && (double) growth > shrink_max*(double) cache && cache - growth < all_experts && tokens >= tiny_tokens) {
            COM_INF("MoE cache: n_ubatch stays %u: n_ubatch %u would take %.2f MiB (%.1f%%) from a cache of %.2f MiB (the experts of %.1f tokens) "
                "that does not hold every expert (%.2f MiB), the decode would read the smaller cache (LLAMA_MOE_PREFILL_UB_SHRINK=1 or -ub to override)\n",
                ub_def, cands[0], growth/1048576.0, 100.0*growth/(double) cache, cache/1048576.0, tokens, all_experts/1048576.0);
            return;
        }
    }
    // strata-gap: a lent prompt graph with a draft keeps the draft context at the n_ubatch of the default (its compute buffer is not lent:
    // the growth would come out of the cache). G4, Qwen3.6-35B-A3B UD-Q4_K_XL + its MTP at 12 / 16 GB: the main context lent at n_ubatch 8192
    // grew the MTP context to 2048 (+293 MiB of compute), the cache shrank 7812 -> 7525 MiB, hit rate 87.5 -> 86.6% and chat decode -2.3% /
    // -1.7% (same tokens and acceptance). LLAMA_MOE_LEND_DRAFT_UB_KEEP=0: the draft grows with the main n_ubatch as before
    static const bool dft_keep_env = getenv("LLAMA_MOE_LEND_DRAFT_UB_KEEP") == nullptr || atoi(getenv("LLAMA_MOE_LEND_DRAFT_UB_KEEP")) != 0;
    const bool dft_keep = dft_keep_env && has_spec && spill_lend_spec && lend_graph && ((spill_lend && pick != cands[0]) || fix2_lend) &&
        params.speculative.draft.n_ubatch <= 0 && m_pick.dft > m_def.dft;
    const int64_t taken     = std::max<int64_t>(0, lend_graph ? kv(m_pick) - kv(m_def) + (dft_keep ? 0 : m_pick.dft - m_def.dft) : m_pick.total - m_def.total);
    const size_t  cache_new = (size_t) (cache - taken);
    if (lend_graph && ((spill_lend && pick != cands[0]) || fix2_lend)) {
        // the cache and the scheduler read the lending mode from the environment
        common_set_env("LLAMA_MOE_CACHE_LEND", "2");
        // strata-gap: with a draft the lent prompt does not seed the cache (n_ubatch >= LLAMA_MOE_CACHE_SEED_AUTO) and the verify batches lost
        // 9.8% decode after a 2k prompt (G4, Flash-Next + Q4_K_M drafter: miss fetch 2.8 -> 6.1 ms/token): the slots that come back after the
        // prompt are refilled with the experts its last ubatch routed most (LLAMA_MOE_CACHE_REFILL_ROUTED, set here unless given)
        if (dft_keep) {
            params.speculative.draft.n_ubatch = (int32_t) common_draft_n_ubatch(params, ub_def);
            COM_INF("MoE cache: lent prompt graphs with a draft: the draft context keeps n_ubatch %d (%.2f MiB of compute instead of %.2f MiB at "
                "n_ubatch %u stay in the cache; LLAMA_MOE_LEND_DRAFT_UB_KEEP=0 to disable)\n", params.speculative.draft.n_ubatch,
                m_def.dft/1048576.0, m_pick.dft/1048576.0, pick);
        }
        if (has_spec && spill_lend_spec && getenv("LLAMA_MOE_CACHE_REFILL_ROUTED") == nullptr) {
            common_set_env("LLAMA_MOE_CACHE_REFILL_ROUTED", "1");
            COM_INF("MoE cache: lent prompt graphs with a draft: the lent slots are refilled with the prompt's most routed experts (%s=0 to disable)\n", "LLAMA_MOE_CACHE_REFILL_ROUTED");
        }
    }
    if (lend_graph) {
        COM_INF("MoE cache: LLAMA_MOE_CACHE_LEND=2: n_ubatch %u, compute buffers %.2f MiB lent from one bank tail (slot %.2f MiB, largest tensor %.2f MiB, "
            "%" PRId64 " experts; at most %.0f%% of the tail, LLAMA_MOE_PREFILL_UB_LEND_FRAC), the cache gives up only the KV growth\n",
            pick, m_pick.compute/1048576.0, (sb.lend_sum > 0 ? sb.lend_sum : sb.sum)/1048576.0, (sb.lend_max > 0 ? sb.lend_max : sb.max)/1048576.0,
            sb.n_expert, 100.0*lend_frac);
    }

    COM_INF("MoE cache: n_ubatch = %u (default %u, rule: %s): +%.2f MiB of compute and KV memory taken from the cache, %.2f -> %.2f MiB "
        "(measured at n_ubatch %u / %u / %u: compute %.2f / %.2f / %.2f MiB, KV %.2f / %.2f / %.2f MiB; experts %.2f MiB; "
        "-ub or LLAMA_MOE_PREFILL_UB=0 to disable, LLAMA_MOE_PREFILL_UB_AUTO=1 for larger ubatches, %.0f ms)\n",
        pick, ub_def, rule, taken/1048576.0, cache/1048576.0, cache_new/1048576.0,
        ub_def, cands[0], pick, m_def.compute/1048576.0, m_ref.compute/1048576.0, m_pick.compute/1048576.0,
        kv(m_def)/1048576.0, kv(m_ref)/1048576.0, kv(m_pick)/1048576.0, have_sb ? sb.total/1048576.0 : -1.0, (ggml_time_us() - t0)/1000.0);

    if (has_spec) {
        COM_INF("MoE cache: speculative decoding: the draft context at n_ubatch %u / %u takes %.2f / %.2f MiB (counted above)\n",
            ub_def, pick, m_def.dft/1048576.0, m_pick.dft/1048576.0);
    }

    params.n_ubatch       = (int32_t) pick;
    cparams.n_ubatch      = pick;
    if (!params.n_batch_set) {
        params.n_batch  = std::max<int32_t>(params.n_batch, (int32_t) pick);
        cparams.n_batch = std::max<uint32_t>(cparams.n_batch, pick);
    }
    params.moe_cache_size  = cache_new;
    cparams.moe_cache_size = cache_new;
}

// the device memory of a DFlash / DSpark draft that cannot be created without its target: returns its K/V cache bytes and sets compute to
// the larger of its feature graph and its noise-block graph (see llama_context::sched_reserve), plus the growth of the target's compute
// buffer once the draft asks it for the inputs of its n_extract layers (path_tgt, n_ubatch_tgt): each is an output of the target's graph,
// n_embd floats per token, held twice (qwen4exp builds a stream mean; G4 Flash-Next at -ub 2048: 989.5 -> 1189.5 MiB, 2 x 100 MiB);
// 0 and compute unchanged for other drafts
static size_t common_fit_dflash_estimate(const std::string & path_dft, const llama_context_params & cp, size_t & compute,
        const std::string & path_tgt = std::string(), uint32_t n_ubatch_tgt = 0) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context_ptr ctx(gguf_init_from_file(path_dft.c_str(), gp));
    if (!ctx) {
        return 0;
    }
    ggml_context_ptr meta_ptr(meta);
    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING || std::string(gguf_get_val_str(ctx.get(), arch_id)) != "dflash") {
        return 0;
    }
    auto u32 = [&](const char * key, int64_t def) -> int64_t {
        const int64_t id = gguf_find_key(ctx.get(), (std::string("dflash.") + key).c_str());
        return id >= 0 && gguf_get_kv_type(ctx.get(), id) == GGUF_TYPE_UINT32 ? (int64_t) gguf_get_val_u32(ctx.get(), id) : def;
    };
    const int64_t n_layer   = u32("block_count", 0);
    const int64_t n_embd    = u32("embedding_length", 0);
    const int64_t n_head_kv = u32("attention.head_count_kv", 0);
    const int64_t n_head    = u32("attention.head_count", 0);
    const int64_t d_k       = u32("attention.key_length", n_head > 0 ? n_embd / n_head : 0);
    const int64_t d_v       = u32("attention.value_length", d_k);
    const int64_t n_block   = u32("block_size", 16);
    const int64_t n_swa     = u32("attention.sliding_window", 0);
    // every layer slides when the pattern says so for each (the only DFlash layout with a window so far), else the cache holds n_ctx
    bool all_swa = n_swa > 0;
    const int64_t pat_id = gguf_find_key(ctx.get(), "dflash.attention.sliding_window_pattern");
    if (all_swa && pat_id >= 0 && gguf_get_kv_type(ctx.get(), pat_id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx.get(), pat_id) == GGUF_TYPE_BOOL) {
        const bool * pat = (const bool *) gguf_get_arr_data(ctx.get(), pat_id);
        for (size_t i = 0; i < gguf_get_arr_n(ctx.get(), pat_id); ++i) {
            all_swa = all_swa && pat[i];
        }
    }
    // the encoder input: the fc matrix reads n_extract target layers
    const ggml_tensor * fc = ggml_get_tensor(meta, "fc.weight");
    const int64_t n_embd_enc = fc != nullptr ? fc->ne[0] : 0;
    // the vocabulary of the lm head (the draft's tokenizer, which matches the target's)
    int64_t n_vocab = 0;
    const int64_t tok_id = gguf_find_key(ctx.get(), "tokenizer.ggml.tokens");
    if (tok_id >= 0 && gguf_get_kv_type(ctx.get(), tok_id) == GGUF_TYPE_ARRAY) {
        n_vocab = (int64_t) gguf_get_arr_n(ctx.get(), tok_id);
    }
    if (n_layer <= 0 || n_embd <= 0 || n_head_kv <= 0 || d_k <= 0 || n_embd_enc <= 0 || n_vocab <= 0) {
        return 0;
    }
    const int64_t n_seq = std::max<int64_t>(1, cp.n_seq_max);
    const int64_t n_ub  = std::max<int64_t>(1, std::min<int64_t>(cp.n_ubatch, cp.n_ctx));
    int64_t cells = cp.n_ctx;
    if (all_swa) {
        cells = std::min<int64_t>(cp.n_ctx, GGML_PAD(n_swa*n_seq + n_ub, 256));
    }
    const double kv_row = (double) n_head_kv*((double) d_k*ggml_type_size(cp.type_k)/ggml_blck_size(cp.type_k) +
                                              (double) d_v*ggml_type_size(cp.type_v)/ggml_blck_size(cp.type_v));
    const size_t kv = (size_t) ((double) n_layer*(double) cells*kv_row);
    const double feat = (double) n_ub*(double) (n_embd_enc + 2*n_embd + 2*n_head_kv*d_k)*sizeof(float);
    const double tok  = (double) n_seq*(double) n_block*((double) n_vocab*2.0 + 8.0*(double) n_embd)*sizeof(float);
    compute = (size_t) (std::max(feat, tok) + 32.0*1024*1024);
    // the target's layer inputs
    const int64_t tl_id = gguf_find_key(ctx.get(), "dflash.target_layers");
    const int64_t n_extract = tl_id >= 0 && gguf_get_kv_type(ctx.get(), tl_id) == GGUF_TYPE_ARRAY ? (int64_t) gguf_get_arr_n(ctx.get(), tl_id) : 0;
    if (n_extract > 0 && n_ubatch_tgt > 0 && !path_tgt.empty()) {
        gguf_init_params gpt = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context_ptr ctx_t(gguf_init_from_file(path_tgt.c_str(), gpt));
        int64_t n_embd_tgt = 0;
        if (ctx_t) {
            const int64_t aid = gguf_find_key(ctx_t.get(), "general.architecture");
            if (aid >= 0 && gguf_get_kv_type(ctx_t.get(), aid) == GGUF_TYPE_STRING) {
                const int64_t eid = gguf_find_key(ctx_t.get(), (std::string(gguf_get_val_str(ctx_t.get(), aid)) + ".embedding_length").c_str());
                if (eid >= 0 && gguf_get_kv_type(ctx_t.get(), eid) == GGUF_TYPE_UINT32) {
                    n_embd_tgt = gguf_get_val_u32(ctx_t.get(), eid);
                }
            }
        }
        compute += (size_t) (2.0*(double) n_extract*(double) n_embd_tgt*(double) std::min<uint32_t>(n_ubatch_tgt, cp.n_ctx)*sizeof(float));
    }
    // k19-draftcost: a DFlash2 draft's restricted head (llama_model::draft_head_from_target), rows of the target's lm head in its type:
    // the LLAMA_DFLASH_DRAFT_VOCAB tokens + the LLAMA_MTP_DRAFT_VOCAB_CTX request rows (4096) + the control tokens (at most 1024
    // counted). Only when set: by default the head is made only for a target without a MoE cache, which this estimate does not size
    const char * env_dv = getenv("LLAMA_DFLASH_DRAFT_VOCAB");
    const int64_t n_dv = env_dv != nullptr ? atoi(env_dv) : 0;
    if (ggml_get_tensor(meta, "selector_hidden.weight") != nullptr && n_dv > 0 && !path_tgt.empty()) {
        const char * env_ctx = getenv("LLAMA_MTP_DRAFT_VOCAB_CTX");
        const int64_t n_rows = n_dv + std::max(0, env_ctx != nullptr ? atoi(env_ctx) : 4096) + 1024;
        std::string prefix;
        int n_split = 1;
        static const std::regex re_split("-([0-9]{5})-of-([0-9]{5})\\.gguf$");
        std::smatch m;
        if (std::regex_search(path_tgt, m, re_split)) {
            prefix  = path_tgt.substr(0, m.position(0));
            n_split = std::stoi(m[2].str());
        }
        for (int i = 0; i < n_split; ++i) {
            std::string fname = path_tgt;
            if (n_split > 1) {
                std::vector<char> split_path(prefix.size() + 32);
                llama_split_path(split_path.data(), split_path.size(), prefix.c_str(), i, n_split);
                fname = split_path.data();
            }
            ggml_context * meta_t = nullptr;
            gguf_init_params gpt = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_t };
            gguf_context_ptr ctx_t(gguf_init_from_file(fname.c_str(), gpt));
            if (!ctx_t) {
                break;
            }
            ggml_context_ptr meta_t_ptr(meta_t);
            if (const ggml_tensor * out = ggml_get_tensor(meta_t, "output.weight")) {
                if (n_rows*10 <= out->ne[1]*9) {
                    compute += (size_t) n_rows*ggml_row_size(out->type, out->ne[0]);
                }
                break;
            }
        }
    }
    return kv;
}

bool common_fit_dflash_draft_bytes(const char * path_dft, const llama_context_params & cp_dft, const char * path_tgt, uint32_t n_ubatch_tgt,
        size_t & weights, size_t & kv, size_t & compute) {
    weights = kv = compute = 0;
    kv = common_fit_dflash_estimate(path_dft, cp_dft, compute, path_tgt != nullptr ? path_tgt : "", n_ubatch_tgt);
    if (kv == 0) {
        return false;
    }
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr gctx(gguf_init_from_file(path_dft, gp));
    for (int64_t i = 0; gctx && i < gguf_get_n_tensors(gctx.get()); i++) {
        weights += gguf_get_tensor_size(gctx.get(), i);
    }
    return weights > 0;
}

// --fit with a separate draft model and a MoE cache: true if the draft model should run on the CPU (see common_init_result)
// the main model is fitted alone with an unbounded cache (on copies of the parameters), which gives the memory left after its layers;
// the draft model's device memory at that context size must fit in what the requested cache leaves, plus at most max_frac of the cache
// k19-mtpkeep: an MTP draft model refused at its device size is weighed again with its expert matrices in host memory (computed on the
// CPU for the drafter's few rows per step), a small n_ubatch (its prompt rows only write K / V, see LLAMA_MTP_PROMPT_KV_ONLY) and the
// target's K / V types. G4 10 / 12 GB, Qwen3.8-Flash-Next UD-IQ4_XS ctx 131072 / 262144 + its Q8_0 MTP GGUF: 3519 / 4100 MiB of
// device memory at its file's placement (2550 MiB of it the experts of its MoE layer, 2495 MiB of compute at n_ubatch 2048 on the
// whole card), 2019 / 2600 MiB requantized, against a cache of 2524 / 2272 MiB: refused, so Studio's MTP never drafted there.
// The caller applies the placement (the draft's tensor overrides, n_ubatch, cache types) to the draft params
struct common_fit_draft_hx {
    bool      on       = false;
    uint32_t  n_ubatch = 0;
    ggml_type type_k   = GGML_TYPE_F16;
    ggml_type type_v   = GGML_TYPE_F16;
};

static bool common_fit_draft_on_cpu(
        const common_params & params, const llama_model_params & mparams, const llama_context_params & cparams,
        const llama_model_params & mparams_dft, llama_context_params cparams_dft, const std::string & path_dft, ggml_log_level log_level,
        bool * needs_target = nullptr, common_fit_draft_hx * hx = nullptr) {
    constexpr double MiB = 1024.0*1024.0;
    // default 0.25 (k15-pp: 0.60 admits more drafts, +13..55% decode, but the drafter's own prompt pass, about 29 us per prompt token
    // plus about 50 ms per request, then sits on every prompt: G4 TTFT +8..18%, +42% at Flash-Next UD-IQ1_S 16 GB 16k, where the smaller
    // cache also lends too few slots for a full layer-major group (k15g4d2, plans/k15_pp.md section 20). LLAMA_FIT_DRAFT_MAX_CACHE_FRAC=0.60
    // opts in until the drafter's prompt pass is off the time to first token
    double max_frac = 0.25;
    if (const char * env = getenv("LLAMA_FIT_DRAFT_MAX_CACHE_FRAC")) {
        max_frac = std::max(0.0, atof(env));
    }
    // k15-pp: the cache the draft leaves must still hold more than this share of the host experts (LLAMA_FIT_DRAFT_MIN_CACHE_FRAC,
    // default LLAMA_MOE_CACHE_HYBRID_MAX_FRAC = 0.10; 0 off), so a draft never shrinks the cache into the hybrid gate's range. G4 16 GB:
    // Flash-Next IQ1_S + Q4_K_M leaves 11.9% (admitted, the win above); + Q8_0 (1 GiB more) would leave about 9%, refused
    double min_frac = common_moe_hybrid_max_frac();
    if (const char * env = getenv("LLAMA_FIT_DRAFT_MIN_CACHE_FRAC")) {
        min_frac = std::max(0.0, atof(env));
    }

    llama_model_params   mp = mparams;
    llama_context_params cp = cparams;
    std::vector<float> ts(params.tensor_split, params.tensor_split + 128);
    std::vector<llama_model_tensor_buft_override> tbo = params.tensor_buft_overrides;
    if (mp.tensor_split == params.tensor_split) {
        mp.tensor_split = ts.data();
    }
    if (mp.tensor_buft_overrides != nullptr && !params.tensor_buft_overrides.empty()) {
        mp.tensor_buft_overrides = tbo.data();
    }
    std::vector<size_t> margins = params.fit_params_target;
    common_fit_set_moe_margin_auto(!params.fit_params_target_set);
    const size_t requested = cparams.moe_cache_size;
    cp.moe_cache_size = std::max<size_t>(requested, (size_t) 1 << 40);

    LOG_INF("%s: --fit with a draft model: fitting the main model and its MoE cache first\n", __func__);
    const auto status = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(), tbo.empty() ? nullptr : tbo.data(),
        margins.data(), params.fit_params_min_ctx, nullptr, log_level);
    if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
        LOG_INF("%s: the main model alone does not fit, the draft model is placed on the CPU\n", __func__);
        return true;
    }
    // k19-place8: a small device whose cache was sized for decode keeps no device drafter (as the placement before it, where none fit):
    // its verify windows are multi-token hybrid batches, open on k19-nanfix (plans/k19_lookup8.md section 4)
    // k19-kvlong (LLAMA_FIT_PLACE8_DRAFT=1, opt-in, measurement): the draft model is weighed against that cache by the rules below
    // as on a mid-size device (what a small drafter buys against the cache it takes, G4 10 / 12 GB at long contexts)
    static const bool place8_draft = getenv("LLAMA_FIT_PLACE8_DRAFT") != nullptr && atoi(getenv("LLAMA_FIT_PLACE8_DRAFT")) != 0;
    // k19-mtpkeep (LLAMA_FIT_DRAFT_HOST_EXPS, default on; 0 = off): only where the draft would be left out otherwise, for an MTP draft
    // model whose experts are not already placed by the user (-cmoed / -otd)
    static const bool hx_env = getenv("LLAMA_FIT_DRAFT_HOST_EXPS") == nullptr || atoi(getenv("LLAMA_FIT_DRAFT_HOST_EXPS")) != 0;
    bool hx_user = false;
    for (const llama_model_tensor_buft_override * p = mparams_dft.tensor_buft_overrides; p != nullptr && p->pattern != nullptr && !hx_user; ++p) {
        hx_user = strstr(p->pattern, "exps") != nullptr;
    }
    const bool hx_ok = hx_env && hx != nullptr && cparams_dft.ctx_type == LLAMA_CONTEXT_TYPE_MTP && !hx_user;
    bool hx_only = false;
    if (common_fit_place8() && !common_fit_place8_mid() && !place8_draft) {
        if (!hx_ok) {
            LOG_INF("%s: small device with the decode-sized MoE cache (k19-place8): the draft model is placed on the CPU\n", __func__);
            return true;
        }
        hx_only = true;   // the small device keeps no draft at its file's placement: only the host-expert placement is weighed
    }
    const size_t pot   = cp.moe_cache_size;            // memory the cache gets once the layers are placed
    const size_t cache = std::min(requested, pot);     // what it would hold without the draft model
    const size_t free_ = pot - cache;                  // memory the cache leaves unused

    // k19-mtpkeep: the host-expert placement, measured and weighed: it may take at most LLAMA_FIT_DRAFT_HOST_EXPS_MAX_CACHE_FRAC (0.5)
    // of the cache and must leave the experts of at least LLAMA_FIT_DRAFT_HOST_EXPS_MIN_TOKENS (1.0) tokens per layer cached; the
    // layers-first rule of the caller then still drops it where it would cost target layers
    auto try_hx = [&]() -> bool {
        if (!hx_ok) {
            return false;
        }
        static const uint32_t hx_ub = getenv("LLAMA_FIT_DRAFT_HOST_EXPS_UB") ? (uint32_t) std::max(1, atoi(getenv("LLAMA_FIT_DRAFT_HOST_EXPS_UB"))) : 64u;
        static const double hx_frac = getenv("LLAMA_FIT_DRAFT_HOST_EXPS_MAX_CACHE_FRAC") ? std::max(0.0, atof(getenv("LLAMA_FIT_DRAFT_HOST_EXPS_MAX_CACHE_FRAC"))) : 0.5;
        static const double hx_tok  = getenv("LLAMA_FIT_DRAFT_HOST_EXPS_MIN_TOKENS") ? std::max(0.0, atof(getenv("LLAMA_FIT_DRAFT_HOST_EXPS_MIN_TOKENS"))) : 1.0;
        std::vector<llama_model_tensor_buft_override> ov;
        for (const llama_model_tensor_buft_override * p = mparams_dft.tensor_buft_overrides; p != nullptr && p->pattern != nullptr; ++p) {
            ov.push_back(*p);
        }
        ov.push_back(llm_ffn_exps_cpu_override());
        ov.push_back({ nullptr, nullptr });
        llama_model_params mp_hx = mparams_dft;
        mp_hx.tensor_buft_overrides = ov.data();
        llama_context_params cp_hx = cparams_dft;
        cp_hx.n_ubatch = std::min(cp_hx.n_ubatch, hx_ub);
        cp_hx.type_k   = cparams.type_k;
        cp_hx.type_v   = cparams.type_v;
        double d = 0.0;
        try {
            common_draft_measure_env split_env;
            std::vector<ggml_backend_dev_t> devs_hx;
            uint32_t ngl_hx = 0, nct_hx = 0, nex_hx = 0;
            const auto dmds = common_get_device_memory_data(path_dft.c_str(), &mp_hx, &cp_hx, devs_hx, ngl_hx, nct_hx, nex_hx, log_level);
            for (size_t i = 0; i < devs_hx.size() && i < dmds.size(); i++) {
                d += dmds[i].model + dmds[i].context + dmds[i].compute;
            }
        } catch (const std::exception & e) {
            LOG_INF("%s: k19-mtpkeep: the draft with its experts in host memory could not be measured (%s)\n", __func__, e.what());
            return false;
        }
        const double take_hx = std::max(0.0, d - (double) free_);
        const double left_hx = std::max(0.0, (double) cache - take_hx);
        const double host_hx = common_moe_host_expert_bytes(params.model.path, mp.tensor_buft_overrides, mp.n_gpu_layers);
        double tok = -1.0;
        common_moe_slot_bytes sbt;
        if (host_hx > 0.0 && common_moe_slot_bytes_of(params.model.path, sbt) && sbt.sum > 0 && sbt.n_used > 0 && sbt.n_layer > 0) {
            const double host_layers = std::min<double>((double) sbt.n_layer, host_hx/((double) sbt.sum*(double) std::max<int64_t>(sbt.n_expert, 1)) + 0.5);
            tok = left_hx/((double) sbt.sum*(double) sbt.n_used*std::max(host_layers, 1.0));
        }
        const bool ok = d > 0.0 && take_hx <= hx_frac*(double) cache && (tok < 0.0 || tok >= hx_tok);
        LOG_INF("%s: k19-mtpkeep: the MTP draft with its experts in host memory (n_ubatch %u, K / V %s / %s) needs %.0f MiB of device memory: "
            "the cache %.0f -> %.0f MiB, the experts of %.1f tokens per layer -> %s (at most %.0f%% of the cache, at least %.2f tokens; "
            "LLAMA_FIT_DRAFT_HOST_EXPS=0 off)\n", __func__, cp_hx.n_ubatch, ggml_type_name(cp_hx.type_k), ggml_type_name(cp_hx.type_v),
            d/MiB, cache/MiB, left_hx/MiB, tok, ok ? "draft model on the GPU" : "refused", 100.0*hx_frac, hx_tok);
        if (ok) {
            hx->on       = true;
            hx->n_ubatch = cp_hx.n_ubatch;
            hx->type_k   = cp_hx.type_k;
            hx->type_v   = cp_hx.type_v;
        }
        return ok;
    };
    if (hx_only) {
        return !try_hx();
    }

    cparams_dft.n_ctx = cp.n_ctx;
    std::vector<ggml_backend_dev_t> devs;
    uint32_t ngl = 0, nct = 0, nex = 0;
    size_t dft = 0;
    try {
        common_draft_measure_env split_env;
        const auto dmds = common_get_device_memory_data(path_dft.c_str(), &mparams_dft, &cparams_dft, devs, ngl, nct, nex, log_level);
        for (size_t i = 0; i < devs.size() && i < dmds.size(); i++) {
            dft += dmds[i].model + dmds[i].context + dmds[i].compute;
        }
    } catch (const std::exception & e) {
        // k15-small4: a draft that needs the target's context to be created (a gemma-4 assistant shares the target's KV cache and token
        // embeddings, "requires ctx_other") cannot be measured on its own. Keeping the old placement put it on the GPU unmeasured:
        // T4 4 GB, gemma-4-26B-A4B UD-Q3_K_M + its Q8_0 assistant, the fit left 258 MiB free and the draft's 425 MiB of weights failed
        // to allocate, so the server exited. Its weights are the tensor data of its GGUF, its context the target's (shared); its compute
        // buffer is LLAMA_FIT_DRAFT_EST_COMPUTE_MIB (default 128; the gemma-4 assistant measured 76.5 MiB at n_ubatch 512 once created).
        // LLAMA_FIT_DRAFT_ESTIMATE=0: the old placement
        const char * env_est = getenv("LLAMA_FIT_DRAFT_ESTIMATE");
        size_t weights = 0;
        if (env_est == nullptr || atoi(env_est) != 0) {
            gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
            gguf_context_ptr gctx(gguf_init_from_file(path_dft.c_str(), gp));
            if (gctx) {
                for (int64_t i = 0; i < gguf_get_n_tensors(gctx.get()); i++) {
                    weights += gguf_get_tensor_size(gctx.get(), i);
                }
            }
        }
        if (weights == 0) {
            LOG_WRN("%s: failed to measure the draft model, keeping the old placement: %s\n", __func__, e.what());
            return false;
        }
        const char * env_c = getenv("LLAMA_FIT_DRAFT_EST_COMPUTE_MIB");
        size_t compute = (size_t) (env_c != nullptr ? std::max(0.0, atof(env_c)) : 128.0) * 1024 * 1024;
        // a DFlash / DSpark draft (it borrows the target's token embeddings and lm head) is priced from its GGUF: the K/V cache and the
        // larger of its two graphs (llama_context::sched_reserve), the feature batch of n_ubatch rows (the encoder input of n_extract
        // target layers, its output and the K/V of a layer) and the noise block of block_size rows per sequence on the target's lm head
        // (logits and the selector's copy). The 128 MiB default held neither the 2000 MiB compute buffer of the token graph at n_ubatch
        // 2048 (G4 32 GB, Qwen3.8-Flash-Next + DFlash2: cudaMalloc failed, drafting off) nor a full-context K/V cache
        const size_t kv_dflash = env_c == nullptr ? common_fit_dflash_estimate(path_dft, cparams_dft, compute, params.model.path, cparams.n_ubatch) : 0;
        if (kv_dflash > 0 || compute == 0) {
            LOG_INF("%s: DFlash draft estimate: K/V %.2f MiB, compute %.2f MiB\n", __func__, kv_dflash/MiB, compute/MiB);
        }
        compute += kv_dflash;
        dft = weights + compute;
        if (needs_target != nullptr) {
            *needs_target = true;
        }
        LOG_INF("%s: the draft model cannot be measured on its own (%s): estimated as its weights %.0f MiB + %.0f MiB of compute buffer "
            "(LLAMA_FIT_DRAFT_EST_COMPUTE_MIB; LLAMA_FIT_DRAFT_ESTIMATE=0 keeps the old placement)\n", __func__, e.what(), weights/MiB, compute/MiB);
    }
    const bool   cap_ok = (double) dft <= (double) free_ + max_frac*(double) cache;
    const double take   = std::max(0.0, (double) dft - (double) free_);   // what the draft takes from the cache
    const double host   = take > 0.0 ? common_moe_host_expert_bytes(params.model.path, mp.tensor_buft_overrides, mp.n_gpu_layers) : 0.0;
    const double left   = std::max(0.0, (double) cache - take);
    const bool   floor_ok = min_frac <= 0.0 || host <= 0.0 || left > min_frac*host;
    bool         gpu    = cap_ok && floor_ok;
    // k18 (LLAMA_FIT_DRAFT_MIN_TOKENS, default 4; 0 = the cap and floor alone, as before): a draft the cap or the floor refuses still gets
    // the GPU when the cache it leaves holds the experts of at least that many tokens per layer (n_expert_used slots per token: a verify
    // window of the draft plus the tail the cache serves on the device). The floor existed because prompts lost their layer-major groups
    // under the hybrid gate; those now run beside the gate (common_moe_cache_auto_ubatch, k18), and the decode gain of drafting comes from
    // reading the resident weights once per verified window. G4, Qwen3.8-Flash-Next UD-IQ4_XS + its Q8_0 MTP head forced on: 16 GB
    // (cache 8271 -> 4576 MiB, 4.1 tokens per layer) decode chat / 2k / 8k 89.9 / 89.3 / 85.6 -> 139.1 / 103.8 / 118.5 tok/s; 12 GB
    // (3549 -> 479 MiB, 0.4 tokens) +29 / -9 / +6%: refused here
    int64_t min_tok = 4;
    if (const char * env = getenv("LLAMA_FIT_DRAFT_MIN_TOKENS")) {
        min_tok = std::max(0, atoi(env));
    }
    double tok_left = -1.0;
    bool   tok_ok   = false;
    // only where prompts keep their layer-major groups: below the share they run beside the gate, or above the gate
    const bool lm_ok = host > 0.0 && (left < common_lm_hybrid_frac()*host || left > common_moe_hybrid_max_frac()*host);
    if (!gpu && min_tok > 0 && lm_ok) {
        common_moe_slot_bytes sbt;
        if (common_moe_slot_bytes_of(params.model.path, sbt) && sbt.sum > 0 && sbt.n_used > 0 && sbt.n_layer > 0) {
            // the expert layers on the host, all of the largest slot (the fit's own estimate of a slot)
            const double host_layers = std::min<double>((double) sbt.n_layer, host/((double) sbt.sum*(double) std::max<int64_t>(sbt.n_expert, 1)) + 0.5);
            tok_left = left/((double) sbt.sum*(double) sbt.n_used*std::max(host_layers, 1.0));
            tok_ok   = tok_left >= (double) min_tok;
            gpu      = tok_ok;
        }
    }
    // k18 (LLAMA_FIT_DRAFT_MIN_TOKENS_REQUANT, default 1; LLAMA_DRAFT_REQUANT=off disables, an explicit type forces it in every fit):
    // a draft refused at its file's types is measured again with its FFN / expert matrices at q3_k and its output matrix at q4_k
    // (requantized at load, see llama_model_loader general.requant_type) and gets the GPU when the cache it then leaves holds the experts
    // of at least that many tokens per layer; the load then uses those types (LLAMA_DRAFT_REQUANT / _OUTPUT are set for it). Drafts are
    // verified: the output is the target's, only the acceptance moves. G4, Qwen3.8-Flash-Next UD-IQ4_XS 12 GB + its Q8_0 MTP drafter:
    // refused (0.4 tokens) -> q3_k / q4_k admitted (1.x tokens), request time chat / 2k / 8k 3.30 / 4.03 / 4.82 -> 2.39 / 4.01 / 4.46 s
    // k19: matrices whose rows q3_k's blocks do not divide go to LLAMA_DRAFT_REQUANT_FALLBACK (q4_0, llama_model_loader::create_tensor),
    // so the measure here sees them too. Same head at ctx 32768: 2009 MiB (0.9 tokens, refused) with its ffn_down_exps kept at Q8_0
    const char * env_rq = getenv("LLAMA_DRAFT_REQUANT");
    const bool rq_off = env_rq != nullptr && (strcmp(env_rq, "off") == 0 || strcmp(env_rq, "0") == 0 || strcmp(env_rq, "none") == 0);
    const bool rq_set = env_rq != nullptr && !rq_off && env_rq[0] != 0;
    // the 1-token line holds where a verify step costs little more than a single token. The fit runs before any model is loaded, so a
    // probe cannot measure that: a cost model gives it instead, the expected distinct experts a verify window of n_max + 1 tokens reads
    // per layer over those of one token, with independent routing (n_expert / n_used * (1 - (1 - n_used / n_expert)^(n_max + 1))). At most
    // LLAMA_FIT_DRAFT_REQUANT_MAX_VERIFY_RATIO (3.5) keeps the 1-token line, above it LLAMA_FIT_DRAFT_MIN_TOKENS applies at the smaller
    // size too. Calibrated on two G4 cells (the model ranks them, its scale is a heuristic: real routing is correlated):
    //  - Qwen3.8-Flash-Next UD-IQ4_XS 12 GB (512 experts, 10 used) + its Q8_0 MTP drafter, n_max 2: model 2.94, measured verify step
    //    1.9x a single token at 1.4 tokens per layer, request time chat 3.32 -> 2.44 s: admitted at 1 token
    //  - Qwen3.6-35B-A3B UD-Q4_K_XL 7 GB (256 experts, 8 used) + Qwen3.5-0.8B Q8_0, draft-simple n_max 4: model 4.70, measured 38.8 ms per
    //    verify step against 5.7 ms per token without the drafter, chat 175 -> 60.5 tok/s at 2.8 tokens: refused
    common_moe_slot_bytes sbt_v;
    double v_ratio = -1.0;
    if (common_moe_slot_bytes_of(params.model.path, sbt_v) && sbt_v.n_used > 0 && sbt_v.n_expert > sbt_v.n_used) {
        const double q = 1.0 - (double) sbt_v.n_used/(double) sbt_v.n_expert;
        // (k19-g4tail2: the server's g4tail rule widens the windows only where its placed cache takes the prefetch band, else they
        // stay at the drafts the drafter was admitted for, fit_n_max; a floor from the wider windows left the drafter out instead)
        const int    n_fit = params.speculative.draft.fit_n_max > 0 ? params.speculative.draft.fit_n_max : common_speculative_n_max(&params.speculative);
        const int    n = std::max(1, n_fit) + 1;
        v_ratio = (double) sbt_v.n_expert/(double) sbt_v.n_used*(1.0 - std::pow(q, n));
    }
    double v_max = 3.5;
    if (const char * e = getenv("LLAMA_FIT_DRAFT_REQUANT_MAX_VERIFY_RATIO")) {
        v_max = std::max(0.0, atof(e));
    }
    // k19-np12b (LLAMA_FIT_DRAFT_MIN_TOKENS_REQUANT_MULTI, default 0.75; 1 = as before): with several slots (llama-server -np > 1, Studio's
    // -np 4 --kv-unified) the floor is lower. Their replay rows and compute take 545 MiB of the 12 GB fit of Qwen3.8-Flash-Next UD-IQ4_XS:
    // its q3_k MTP drafter leaves the experts of 0.8 tokens per layer there (1.3 at -np 1) and was refused, chat 3.375 s vs 2.700 s
    // admitted (G4 k19sr). LLAMA_FIT_DRAFT_MIN_TOKENS_REQUANT (fractional) sets the floor for every slot count
    double rq_min_tok = v_ratio >= 0.0 && v_ratio <= v_max ? 1.0 : (double) min_tok;
    if (rq_min_tok == 1.0 && params.n_parallel > 1) {
        const char * em = getenv("LLAMA_FIT_DRAFT_MIN_TOKENS_REQUANT_MULTI");
        rq_min_tok = em != nullptr ? std::max(0.0, atof(em)) : 0.75;
    }
    if (const char * e = getenv("LLAMA_FIT_DRAFT_MIN_TOKENS_REQUANT")) {
        rq_min_tok = std::max(0.0, atof(e));
    }
    // k19-draftcost (LLAMA_FIT_DRAFT_REQUANT_SHARE, default 0.10; 0 = only the rescue above): a draft admitted at its file's types that
    // takes more than this share of the cache it would get is requantized the same way, which leaves a larger cache. Strata's MTP layer
    // holds 871 MiB (Q2_0 experts). G4 Qwen3.8-Flash-Next UD-IQ4_XS 32 GB ctx 32768 + its Q8_0 MTP drafter (3109 MiB, 12.5% of the
    // 24800 MiB cache): 1608 MiB requantized, cache 21687 -> 23187 MiB, decode chat / 8k / 30k +4.5 / +18 / +20%. Its Q4_K_M drafter at
    // 16 GB (2229 MiB, 31%) was admitted as is with a smaller cache than the rescued Q8_0 one. Drafts are verified: only the acceptance
    // and the cache size move (same text, full card: the requantized drafter accepts 4.20 / 3.88 tokens per pass against Q8_0's 4.13 / 3.88)
    double rq_share = 0.10;
    if (const char * e = getenv("LLAMA_FIT_DRAFT_REQUANT_SHARE")) {
        rq_share = std::max(0.0, atof(e));
    }
    const bool rq_heavy = gpu && rq_share > 0.0 && cache > 0 && take > rq_share*(double) cache;
    double rq_tok = -1.0, rq_dft = 0.0;
    bool   rq_taken = false;
    if ((!gpu || rq_heavy) && !rq_off && !rq_set && rq_min_tok > 0.0 && host > 0.0 && (needs_target == nullptr || !*needs_target)) {
        std::vector<llama_model_kv_override> kvo;
        if (mparams_dft.kv_overrides != nullptr) {
            for (const llama_model_kv_override * p = mparams_dft.kv_overrides; p->key[0] != 0; ++p) {
                kvo.push_back(*p);
            }
        }
        for (const auto & kv : { std::make_pair("general.requant_type", "q3_k"), std::make_pair("general.requant_output_type", "q4_k") }) {
            llama_model_kv_override o = {};
            snprintf(o.key, sizeof(o.key), "%s", kv.first);
            o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            snprintf(o.val_str, sizeof(o.val_str), "%s", kv.second);
            kvo.push_back(o);
        }
        kvo.emplace_back();
        kvo.back().key[0] = 0;
        llama_model_params mp_rq = mparams_dft;
        mp_rq.kv_overrides = kvo.data();
        try {
            common_draft_measure_env split_env;
            std::vector<ggml_backend_dev_t> devs_rq;
            uint32_t ngl2 = 0, nct2 = 0, nex2 = 0;
            const auto dmds = common_get_device_memory_data(path_dft.c_str(), &mp_rq, &cparams_dft, devs_rq, ngl2, nct2, nex2, log_level);
            for (size_t i = 0; i < devs_rq.size() && i < dmds.size(); i++) {
                rq_dft += dmds[i].model + dmds[i].context + dmds[i].compute;
            }
        } catch (const std::exception & e) {
            rq_dft = 0.0;
        }
        common_moe_slot_bytes sbt;
        if (rq_dft > 0.0 && rq_dft < (double) dft && common_moe_slot_bytes_of(params.model.path, sbt) && sbt.sum > 0 && sbt.n_used > 0 && sbt.n_layer > 0) {
            const double left_rq = std::max(0.0, (double) cache - std::max(0.0, rq_dft - (double) free_));
            const bool   lm_rq   = left_rq < common_lm_hybrid_frac()*host || left_rq > common_moe_hybrid_max_frac()*host;
            const double host_layers = std::min<double>((double) sbt.n_layer, host/((double) sbt.sum*(double) std::max<int64_t>(sbt.n_expert, 1)) + 0.5);
            rq_tok = left_rq/((double) sbt.sum*(double) sbt.n_used*std::max(host_layers, 1.0));
            if (lm_rq && (rq_heavy || rq_tok >= rq_min_tok)) {
                gpu = true;
                rq_taken = true;
                common_set_env("LLAMA_DRAFT_REQUANT", "q3_k");
                common_set_env("LLAMA_DRAFT_REQUANT_OUTPUT", "q4_k");
            }
        }
    }
    LOG_INF("%s: draft model needs %.0f MiB of device memory; the MoE cache gets %.0f MiB (%.0f MiB left over), %.0f MiB with the draft "
        "(%.1f%% of %.0f MiB of host experts) -> draft model on the %s (it may take at most %.0f%% of the cache, LLAMA_FIT_DRAFT_MAX_CACHE_FRAC, "
        "and must leave more than %.0f%% of the host experts cached, LLAMA_FIT_DRAFT_MIN_CACHE_FRAC; LLAMA_FIT_DRAFT_LAST=0 for the old order)\n",
        __func__, dft/MiB, cache/MiB, free_/MiB, left/MiB, host > 0.0 ? 100.0*left/host : 0.0, host/MiB, gpu ? "GPU" : "CPU",
        100.0*max_frac, 100.0*min_frac);
    if (tok_left >= 0.0) {
        LOG_INF("%s: the cache the draft leaves holds the experts of %.1f tokens per layer -> %s (at least %d, LLAMA_FIT_DRAFT_MIN_TOKENS; 0: "
            "the cap and the floor alone)\n", __func__, tok_left, tok_ok ? "draft model on the GPU" : "refused", (int) min_tok);
    }
    if (rq_tok >= 0.0) {
        LOG_INF("%s: draft requantized at load (q3_k FFN / experts, LLAMA_DRAFT_REQUANT_FALLBACK where q3_k blocks do not divide the rows, q4_k output): %.0f MiB of device memory, the cache it leaves holds the experts "
            "of %.1f tokens per layer -> %s (at least %.2f, LLAMA_FIT_DRAFT_MIN_TOKENS_REQUANT(_MULTI): verify window cost %.2fx a token, %s %.2f "
            "LLAMA_FIT_DRAFT_REQUANT_MAX_VERIFY_RATIO; LLAMA_DRAFT_REQUANT=off disables)\n", __func__,
            rq_dft/MiB, rq_tok, rq_taken ? (rq_heavy ? "draft model on the GPU, requantized (it took "
            "more than LLAMA_FIT_DRAFT_REQUANT_SHARE of the cache)" : "draft model on the GPU, requantized") : (gpu ? "kept at its types" : "refused"), rq_min_tok, v_ratio,
            v_ratio >= 0.0 && v_ratio <= v_max ? "at most" : "over", v_max);
    } else if (!gpu && rq_off) {
        LOG_INF("%s: draft refused at its file's types; requantization off (LLAMA_DRAFT_REQUANT=off)\n", __func__);
    }
    if (!gpu && !rq_set && (needs_target == nullptr || !*needs_target)) {
        gpu = try_hx();
    }
    return !gpu;
}

// speculative decoding: the draft model and its context are created after the main context (common_speculative_init_result), which
// until then sees the device memory that --fit gave them as free. What the main context places in free memory while it is created
// (the automatic MoE cache lending decision, the seed graph, a staging arena without a lender) took it: Qwen3.8-Flash-Next UD-IQ4_XS
// with an MTP draft on 32 GB, --fit on --moe-cache-mib 200000: the cache did not lend its 1275 MiB of staging copies because 5 GiB
// looked free, and the draft context then failed to allocate its 778 MiB compute buffer. The draft is measured at the params it is
// created with (as in common_moe_cache_auto_ubatch) and that memory is kept free in the main context while it is created
// (llama_context_params::mem_reserve). LLAMA_DRAFT_MEM_RESERVE=0 disables.
static void common_draft_mem_reserve(const common_params & params, const llama_model_params & mparams, const llama_context_params & cparams,
        std::vector<ggml_backend_dev_t> & devs, std::vector<size_t> & bytes) {
    devs.clear();
    bytes.clear();
    if (const char * env = getenv("LLAMA_DRAFT_MEM_RESERVE"); env != nullptr && atoi(env) == 0) {
        return;
    }
    const bool has_draft = params.speculative.has_dft() && !params.speculative.draft.fit_off;
    const bool spec_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end() && !params.speculative.draft.fit_off;
    if (!has_draft && !spec_mtp) {
        return;
    }
    common_params        params_dft  = common_base_params_to_speculative(params);
    llama_model_params   mparams_dft = has_draft ? common_model_params_to_llama(params_dft) : mparams;
    llama_context_params cparams_dft = common_context_params_to_llama(params_dft);
    if (spec_mtp) {
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }
    cparams_dft.n_rs_seq = 0;
    cparams_dft.n_ctx    = cparams.n_ctx;
    uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;
    try {
        common_draft_measure_env split_env(has_draft);
        const common_device_memory_data_vec dmd = common_get_device_memory_data(has_draft ? params_dft.model.path.c_str() : params.model.path.c_str(),
            &mparams_dft, &cparams_dft, devs, hp_ngl, hp_nct, hp_nex, params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        // the weights of a draft model are loaded after the main context too, an MTP context runs on the weights of the main model
        for (size_t i = 0; i < devs.size() && i + 1 < dmd.size(); ++i) {
            bytes.push_back((size_t) ((has_draft ? dmd[i].model : 0) + dmd[i].context + dmd[i].compute));
        }
        devs.resize(bytes.size());
    } catch (const std::exception & e) {
        devs.clear();
        bytes.clear();
        // a DFlash / DSpark draft cannot be created without its target: its estimate (common_fit_dflash_estimate, the weights from its
        // GGUF) is kept free on the first GPU it would use. Without it the main context of Qwen3.8-Flash-Next + DFlash2 at 32 / 16 GB
        // (G4) placed its staging arena in the draft's memory, and its pp graph then failed to allocate once the draft was loaded
        size_t compute = 0;
        const size_t kv = has_draft && mparams_dft.n_gpu_layers != 0 ? common_fit_dflash_estimate(params_dft.model.path, cparams_dft, compute, params.model.path, cparams.n_ubatch) : 0;
        ggml_backend_dev_t dev = nullptr;
        if (kv > 0) {
            if (!params_dft.devices.empty()) {
                dev = params_dft.devices[0]; // nullptr: the draft is not offloaded
            } else {
                for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        dev = d;
                    }
                }
            }
        }
        size_t weights = 0;
        if (dev != nullptr) {
            gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
            gguf_context_ptr gctx(gguf_init_from_file(params_dft.model.path.c_str(), gp));
            for (int64_t i = 0; gctx && i < gguf_get_n_tensors(gctx.get()); i++) {
                weights += gguf_get_tensor_size(gctx.get(), i);
            }
        }
        if (dev != nullptr && weights > 0) {
            devs.push_back(dev);
            bytes.push_back(weights + kv + compute);
            COM_INF("DFlash draft: %.2f MiB kept free on %s for it (estimate: weights %.2f, K/V %.2f, compute %.2f MiB)\n",
                bytes.back()/1048576.0, ggml_backend_dev_name(dev), weights/1048576.0, kv/1048576.0, compute/1048576.0);
        } else {
            COM_WRN("memory measure of the draft for the memory the main context keeps free failed, nothing kept free: %s\n", e.what());
        }
    }
}

// true if the model has sliding-window attention layers (<arch>.attention.sliding_window > 0): their KV cache holds the window plus one ubatch
static bool common_gguf_has_swa(const std::string & path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), gp));
    if (!ctx) {
        return true; // unknown: as with the window
    }
    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return true;
    }
    const int64_t kid = gguf_find_key(ctx.get(), (std::string(gguf_get_val_str(ctx.get(), arch_id)) + ".attention.sliding_window").c_str());
    if (kid < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx.get(), kid)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx.get(), kid) > 0;
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx.get(), kid) > 0;
        default:               return true;
    }
}

static std::string common_gguf_arch(const std::string & path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), gp));
    if (!ctx) {
        return "";
    }
    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    return arch_id >= 0 && gguf_get_kv_type(ctx.get(), arch_id) == GGUF_TYPE_STRING ? std::string(gguf_get_val_str(ctx.get(), arch_id)) : "";
}

static bool common_moe_resident_ubatch_try(common_params & params, llama_model_params & mparams, llama_context_params & cparams,
        const common_fit_extra_model * extra, ggml_log_level log_level, uint32_t u, bool quiet);

// k9-prompt-pass: the prompt ubatch of a MoE model whose experts are all on the device (see the call in common_init_result)
// k15-lfm2: where 2048 does not keep every expert on the device, 1024 is tried next (LFM2.5 on a 16 GB G4 tier: 512 -> 1024; a ubatch
// size only regroups the same products, as -ub does: the equivalence rule of plans/k11.md). LLAMA_MOE_RESIDENT_UB_LADDER=0: 2048 only;
// =2: for every model (default: LFM2 / LFM2-MoE)
static void common_moe_resident_ubatch(common_params & params, llama_model_params & mparams, llama_context_params & cparams,
        const common_fit_extra_model * extra, ggml_log_level log_level) {
    const char * env = getenv("LLAMA_MOE_RESIDENT_UB");
    const int64_t want = env != nullptr ? atoll(env) : 2048;
    const uint32_t ub_def = cparams.n_ubatch;
    if (want <= (int64_t) ub_def || (params.n_ctx > 0 && want > params.n_ctx) || (params.n_batch_set && (int64_t) cparams.n_batch < want)) {
        return;
    }
    const char * env_l = getenv("LLAMA_MOE_RESIDENT_UB_LADDER");
    const int ladder = env_l != nullptr ? atoi(env_l) : 1;
    bool ladder_on = ladder >= 2;
    if (ladder == 1 && env == nullptr) {
        const std::string arch = common_gguf_arch(params.model.path);
        ladder_on = arch == "lfm2" || arch == "lfm2moe";
    }
    if (common_moe_resident_ubatch_try(params, mparams, cparams, extra, log_level, (uint32_t) want, ladder_on)) {
        return;
    }
    if (ladder_on && want/2 > (int64_t) ub_def) {
        if (!common_moe_resident_ubatch_try(params, mparams, cparams, extra, log_level, (uint32_t) (want/2), false)) {
            return;
        }
        COM_INF("MoE experts on the device: n_ubatch %u does not keep every expert on the device, %u does (LLAMA_MOE_RESIDENT_UB_LADDER=0: %u only)\n",
            (uint32_t) want, (uint32_t) (want/2), (uint32_t) want);
    }
}

// one candidate ubatch u: true (and params / cparams set) if the fit at u keeps every expert on the device
static bool common_moe_resident_ubatch_try(common_params & params, llama_model_params & mparams, llama_context_params & cparams,
        const common_fit_extra_model * extra, ggml_log_level log_level, uint32_t u, bool quiet) {
    const uint32_t ub_def = cparams.n_ubatch;
    common_moe_slot_bytes sbt;
    if (!common_moe_slot_bytes_of(params.model.path, sbt) || sbt.n_expert <= 1) {
        return true; // not a MoE model (nothing to try)
    }
    const char * env_swa = getenv("LLAMA_MOE_RESIDENT_UB_SWA");
    if ((env_swa == nullptr || atoi(env_swa) == 0) && common_gguf_has_swa(params.model.path)) {
        COM_INF("MoE experts on the device: n_ubatch stays %u (sliding-window layers: their KV cache grows with n_ubatch; LLAMA_MOE_RESIDENT_UB_SWA=1 "
            "to grow it anyway)\n", ub_def);
        return true;
    }
    // the fit again at the larger ubatch, on copies: every expert must stay on the device
    llama_model_params   mp = mparams;
    llama_context_params cp = cparams;
    cp.n_ubatch = u;
    cp.n_batch  = std::max(cp.n_batch, u);
    std::vector<float> ts(params.tensor_split, params.tensor_split + 128);
    std::vector<llama_model_tensor_buft_override> tbo = params.tensor_buft_overrides;
    if (mp.tensor_split == params.tensor_split) {
        mp.tensor_split = ts.data();
    }
    if (mp.tensor_buft_overrides != nullptr && !params.tensor_buft_overrides.empty()) {
        mp.tensor_buft_overrides = tbo.data();
    }
    std::vector<size_t> margins = params.fit_params_target;
    common_fit_set_moe_margin_auto(!params.fit_params_target_set);
    const auto status = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(), tbo.empty() ? nullptr : tbo.data(),
        margins.data(), params.fit_params_min_ctx, extra, log_level, params.fit_kv_resident);
    if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS || cp.n_ubatch != u || cp.n_ctx != cparams.n_ctx || mp.n_gpu_layers != mparams.n_gpu_layers ||
            common_experts_on_offloaded_host(params, mp.tensor_buft_overrides, mp.n_gpu_layers) ||
            (tbo.size() != params.tensor_buft_overrides.size()) ||
            !std::equal(tbo.begin(), tbo.end(), params.tensor_buft_overrides.begin(), [](const auto & a, const auto & b) {
                return a.pattern == b.pattern && a.buft == b.buft; })) {
        if (!quiet) {
            COM_INF("MoE experts on the device: n_ubatch stays %u (at n_ubatch %u the fit does not keep every expert on the device)\n", ub_def, u);
        }
        return false;
    }
    // the draft context keeps the ubatch it was fitted with (the MTP context reads the prompt over one layer; a draft model's compute
    // buffer would grow for nothing)
    if (extra != nullptr && params.speculative.draft.n_ubatch <= 0) {
        params.speculative.draft.n_ubatch = (int32_t) extra->cparams->n_ubatch;
    }
    params.n_ubatch  = (int32_t) u;
    cparams.n_ubatch = u;
    if (cparams.n_batch < u) {
        params.n_batch  = (int32_t) u;
        cparams.n_batch = u;
    }
    COM_INF("MoE experts on the device: n_ubatch = %u (default %u; draft context %d; LLAMA_MOE_RESIDENT_UB=0 keeps the default)\n", u, ub_def,
        extra != nullptr ? params.speculative.draft.n_ubatch : -1);
    return true;
}

// the request --moe-cache-mib auto stands for: larger than any device, so that the fit gives the cache what the layers leave
// (the fit caps the cache at the bytes of the host experts and at the free device memory)
static constexpr size_t COMMON_MOE_CACHE_AUTO_BYTES = (size_t) 1 << 40;

// k9-prompt-pass: unified memory: every offload device an integrated GPU (e.g. AMD Strix Halo gfx1151, GGML_BACKEND_DEVICE_TYPE_IGPU
// from the device's integrated attribute); LLAMA_MOE_UMA=1 / 0 forces the decision
static bool common_moe_uma(const common_params & params) {
    if (const char * e = getenv("LLAMA_MOE_UMA")) {
        return atoi(e) != 0;
    }
    int n_gpu = 0, n_igpu = 0;
    const auto count = [&](ggml_backend_dev_t dev) {
        const auto t = ggml_backend_dev_type(dev);
        n_gpu  += t == GGML_BACKEND_DEVICE_TYPE_GPU;
        n_igpu += t == GGML_BACKEND_DEVICE_TYPE_IGPU;
    };
    if (!params.devices.empty()) {
        for (ggml_backend_dev_t dev : params.devices) {
            if (dev != nullptr) {
                count(dev);
            }
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            count(ggml_backend_dev_get(i));
        }
    }
    return n_igpu > 0 && n_gpu == 0;
}

bool common_moe_cache_auto_resolve(common_params & params, bool fit) {
    if (!params.moe_cache_auto) {
        return false;
    }
    params.moe_cache_auto = false;
    if (!fit) {
        COM_WRN("%s: --moe-cache-mib auto needs --fit on, not using a MoE cache\n", __func__);
        params.moe_cache_size = 0;
        return false;
    }
    params.moe_cache_size = COMMON_MOE_CACHE_AUTO_BYTES;
    return true;
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    // k19-int3fix2: the verify graphs a server prewarms beside a drafter (counted by the fits that carry a draft model or MTP heads): up
    // to 3 sampler chains (greedy, masked greedy, top-k) x the verify widths 1 .. n_max + 1 (see common_fit_set_prewarm_graphs)
    common_fit_set_prewarm_graphs(params.warmup ? 3*(std::max(1, common_speculative_n_max(&params.speculative)) + 1) : 0);
    // k19-l4gap2: the verify window's rows for the disk-tier k19-placemid ubatch (common/fit.cpp)
    common_fit_set_verify_rows(common_speculative_n_max(&params.speculative) + 1);
    // k19-uma: the parameters as given, for the launch again without a MoE cache (see below)
    const common_params params_given = params;
    bool uma_no_cache = false;
uma_restart:
    // k15-up2: built-in MTP heads (no -md model) for the context's prefetch depth rule (llama_context::sched_new), before any fit probe
    common_set_env("LLAMA_SPEC_MTP_HEADS", !params.speculative.has_dft() && std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP; }) ? "1" : "0");
    // --moe-cache-mib auto: the fit sizes the cache (the fitted size is kept in params, so that a reload from params, e.g. the server
    // waking from sleep, gets the same cache)
    const bool moe_cache_auto = common_moe_cache_auto_resolve(params, params.fit_params);
    int64_t moe_margin_gain = 0; // the part of the fitted MoE cache that came from the reduced margin beside it (common_fit_moe_margin_gain)

    // --weight-stream on means what LLAMA_WEIGHT_STREAM=1 does for the automatic expert pinning of the model loader (which reads the
    // environment, it has no weight stream parameter): the experts stay mapped, so the weight stream and its RAM tier serve them instead
    // of a pinned copy taking that RAM. An explicit LLAMA_MOE_PIN_EXPERTS still wins
    // k18 (LLAMA_MOE_LOWRAM_LEND=1 keeps lending): where D1 streams the experts through the RAM tier, the device cache does not lend slots
    // to prompt staging. A lent slot's expert must move to RAM, which in this tier has no room for it (RAM + device slots hold every
    // expert with ~1 GiB to spare), so each prompt dropped ~2000 experts that the decode then read back from the drive (G4 fndl 48 GiB,
    // 8k prompt: 7.5 GB read, decode 28-32 tok/s). Without the lend the fit gives the prompt graphs their own buffer (device cache 21.4 ->
    // 20.1 GiB), RAM hits 99.9% of the misses, and both prompt and decode are faster (8k e2e 83-89 -> 63 s, 16k 130-138 -> 87 s, decode
    // after 8k 28-32 -> 52 tok/s). Set before the fit, which reads it; an explicit LLAMA_MOE_CACHE_LEND wins
    // (before the fit there are no expert overrides and the cache size may be auto: only the D1 memory test, on every expert of the model)
    if (params.load_mode == LLAMA_LOAD_MODE_NONE && params.moe_cache_size != 0 && getenv("LLAMA_MOE_CACHE_LEND") == nullptr &&
            (getenv("LLAMA_MOE_LOWRAM_LEND") == nullptr || atoi(getenv("LLAMA_MOE_LOWRAM_LEND")) == 0) &&
            (getenv("LLAMA_MOE_LOWRAM_MAP") == nullptr || atoi(getenv("LLAMA_MOE_LOWRAM_MAP")) != 0) &&
            common_moe_lowram_short(params)) {
        COM_INF("%s: MoE cache: the experts are streamed through the RAM tier: the device cache does not lend slots to prompts "
            "(LLAMA_MOE_LOWRAM_LEND=1 lends)\n", __func__);
        common_set_env("LLAMA_MOE_CACHE_LEND", "0");
    }
    if (params.weight_stream == 2 && getenv("LLAMA_WEIGHT_STREAM") == nullptr && getenv("LLAMA_MOE_PIN_EXPERTS") == nullptr) {
        COM_INF("%s: --weight-stream on: automatic expert pinning off (LLAMA_MOE_PIN_EXPERTS=1 or 2 to pin anyway)\n", __func__);
        common_set_env("LLAMA_MOE_PIN_EXPERTS", "0");
    }

    // with --fit the MoE cache load mode is decided after the fit has placed the experts
    // k9-prompt-pass: unified memory (every offload device an integrated GPU, e.g. AMD Strix Halo gfx1151): spilled experts are not copied
    // over a PCIe link, so the discrete-GPU rule that keeps n_ubatch at 512 while the cache does not hold every expert does not apply there.
    // Measured on gfx1151 (Qwen3.6-35B-A3B UD-Q3_K_M + MTP, 8 GB budget, plans/k9_platforms.md): -ub 2048 vs the default 2k prompt
    // 1252 -> 1373 (Linux) / 1125 -> 1185 (Windows), decode after it 65.0 -> 72.3 / 57.7 -> 66.8, chat decode equal. Without -ub and with
    // a MoE cache asked for, n_ubatch = LLAMA_MOE_UMA_UB (2048) there, set before the fit and handled as a given -ub (the fit, the draft
    // context see it as such). LLAMA_MOE_UMA_UB=0 off; LLAMA_MOE_UMA=1 / 0 forces the unified-memory decision.
    // On Windows n_batch stays as it is there (params.moe_uma_keep_batch, common_moe_cache_auto_ubatch): gfx1151, same cell, 8k prompt, layer-major
    // groups of 4 x 2048 vs one ubatch per graph: Windows 1257 vs 1368 tok/s (+1.5 GB working set), Linux 1508 vs 1458 (amd-ci kpp run)
    if (!model_only && !params.embedding && (moe_cache_auto || params.moe_cache_size > 0)) {
        const char * env_ub  = getenv("LLAMA_MOE_UMA_UB");
        const int64_t uma_ub = env_ub != nullptr ? atoll(env_ub) : 2048;
        const bool uma = common_moe_uma(params);
        // LLAMA_MOE_UMA_KEEP_BATCH=1 / 0: keep n_batch on unified memory (default: Windows only)
#ifdef _WIN32
        bool keep_batch = true;
#else
        bool keep_batch = false;
#endif
        if (const char * e = getenv("LLAMA_MOE_UMA_KEEP_BATCH")) {
            keep_batch = atoi(e) != 0;
        }
        params.moe_uma_keep_batch = uma && keep_batch;
        if (uma && !params.n_ubatch_set && uma_ub > params.n_ubatch && (params.n_ctx <= 0 || uma_ub <= params.n_ctx) && (!params.n_batch_set || params.n_batch >= uma_ub)) {
            COM_INF("%s: MoE cache on unified memory: n_ubatch = %" PRId64 " (default %d; LLAMA_MOE_UMA_UB=0 keeps the default)\n", __func__,
                uma_ub, params.n_ubatch);
            params.n_ubatch     = (int32_t) uma_ub;
            params.n_ubatch_set = true;
            params.n_batch      = std::max<int32_t>(params.n_batch, (int32_t) uma_ub);
        }
    }

    // k19-mgpu: a reload from params (the server waking from sleep) fits again from -sm layer and decides the split mode again
    if (params.split_mode_auto) {
        params.split_mode      = LLAMA_SPLIT_MODE_LAYER;
        params.split_mode_auto = false;
    }

    auto mparams = common_model_params_to_llama_impl(params, params.fit_params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
        COM_TRC("%s", "fitting params to device memory ...\n");
        common_fit_set_ubatch_user(params.n_ubatch_set); // k19-fitfaoff
        common_fit_faoff_ubatch_reset();
        COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n");

        // the draft context is created from the same base params and follows the main context, fit both together
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

        common_params params_dft = common_base_params_to_speculative(params);

        auto mparams_dft = common_model_params_to_llama(params_dft);
        auto cparams_dft = common_context_params_to_llama(params_dft);
        if (spec_mtp) {
            cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        }
        cparams_dft.n_rs_seq = 0;

        const common_fit_extra_model extra = {
            /*.path_model   =*/ params_dft.model.path.c_str(),
            /*.mparams      =*/ &mparams_dft,
            /*.cparams      =*/ &cparams_dft,
            /*.shares_model =*/ !has_draft, // an MTP context runs on the weights of the main model
        };
        const ggml_log_level fit_log_level = params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR;

        // a separate draft model next to a MoE cache goes last: the main model's layers and its cache are placed first, and the draft
        // model gets the GPU only if it fits in the memory the cache does not use, or takes at most LLAMA_FIT_DRAFT_MAX_CACHE_FRAC
        // (default 0.25), leaving more than LLAMA_FIT_DRAFT_MIN_CACHE_FRAC (0.10) of the host experts cached, of the memory the cache would get; otherwise it runs on the CPU, or, if the caller can run without it
        // (fit_can_drop), it is left out and drafting is off unless LLAMA_FIT_DRAFT_CPU=1 (drafting on the CPU is slower than none)
        // the old order (the draft model first, the main model around it) with LLAMA_FIT_DRAFT_LAST=0, or when -ngld or -devd is set

        // the fit of the main model on copies of the params at a K/V type, K/V placement (kvo: on the device) and n_ubatch (0: as given),
        // with the MTP drop rule below applied (the KV auto rules: k16-parity, k19-kvlong)
        std::vector<uint8_t> exps;
        const int64_t n_layer = params.fit_params && !model_only ? common_model_n_layer(params.model.path, exps) : 0;
        struct kv_probe {
            bool     ok      = false;
            uint32_t n_ctx   = 0;
            int64_t  layers  = 0;
            bool     mtp     = false;
            int64_t  n_host  = 0; // tensor buffer overrides the fit added (experts in system memory)
            int64_t  cache   = 0;
            bool     kvo     = true; // K/V on the device (offload_kqv)
            uint32_t ub      = 0;     // n_ubatch after the fit
            uint32_t ub_req  = 0;     // n_ubatch asked for (0: as given)
            ggml_type type   = GGML_TYPE_COUNT; // K/V type
            int32_t  win     = -1;    // k19-kvwin4: KV window ring positions (-1: none)
        };
        // k19-kvwin4: the KV window ring (n_win, n_sinks) a probe fits with (-1: as given)
        int32_t probe_win = -1, probe_win_sinks = 0;
        const auto probe = [&](ggml_type t, bool kvo, uint32_t ub) {
            kv_probe r;
            const auto one = [&](bool with_extra, kv_probe & o) {
                llama_model_params   mp = mparams;
                llama_context_params cp = cparams;
                if (probe_win >= 0) {
                    cp.kv_win       = probe_win;
                    cp.kv_win_sinks = probe_win_sinks;
                }
                std::vector<float>   ts(params.tensor_split, params.tensor_split + 128);
                std::vector<llama_model_tensor_buft_override> tbo = params.tensor_buft_overrides;
                std::vector<size_t>  mg = params.fit_params_target;
                mp.tensor_split = ts.data();
                if (mparams.tensor_buft_overrides != nullptr && mparams.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                    mp.tensor_buft_overrides = tbo.data();
                }
                cp.type_k = t;
                cp.type_v = t;
                cp.offload_kqv = kvo;
                if (ub > 0) {
                    cp.n_ubatch = ub;
                    cp.n_batch  = std::max(cp.n_batch, ub);
                }
                if (!with_extra) {
                    mp.load_mtp = params.load_mtp;
                    cp.n_rs_seq = 0;
                }
                int64_t n0 = 0;
                for (const auto & o2 : tbo) {
                    n0 += o2.pattern != nullptr;
                }
                llama_model_params   mp_dft = mparams_dft; // the fit may write the draft's params: copies
                llama_context_params cp_dft = cparams_dft;
                cp_dft.offload_kqv = kvo;
                const common_fit_extra_model ex = { extra.path_model, &mp_dft, &cp_dft, extra.shares_model };
                common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                const auto st = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(), tbo.empty() ? nullptr : tbo.data(),
                    mg.data(), params.fit_params_min_ctx, with_extra ? &ex : nullptr, fit_log_level, params.fit_kv_resident);
                o.ok     = st == COMMON_PARAMS_FIT_STATUS_SUCCESS;
                o.kvo    = kvo;
                o.ub     = cp.n_ubatch;
                o.ub_req = ub;
                o.type   = t;
                o.win    = probe_win;
                o.n_ctx  = cp.n_ctx;
                o.layers = mp.n_gpu_layers == llama_model_default_params().n_gpu_layers || mp.n_gpu_layers < 0 ? n_layer + 1 : mp.n_gpu_layers;
                o.mtp    = with_extra && spec_mtp && !has_draft;
                int64_t n1 = 0;
                for (const auto & o2 : tbo) {
                    n1 += o2.pattern != nullptr;
                }
                o.n_host = n1 - n0;
                o.cache  = cp.moe_cache_size >= COMMON_MOE_CACHE_AUTO_BYTES ? 0 : (int64_t) cp.moe_cache_size + common_fit_moe_margin_gain();
            };
            // a separate draft model is placed after the main model and its cache (see common_fit_draft_on_cpu): only the main model here
            one(spec_mtp && !has_draft, r);
            const char * env_md = getenv("LLAMA_FIT_MTP_DROP");
            if (r.ok && spec_mtp && !has_draft && params.speculative.draft.fit_can_drop && r.layers < n_layer + 1 &&
                    mparams.n_gpu_layers == llama_model_default_params().n_gpu_layers && (env_md == nullptr || atoi(env_md) != 0)) {
                kv_probe r2;
                one(false, r2);
                if (r2.ok && r2.layers >= n_layer + 1) {
                    r = r2; // the MTP drop rule below would drop the drafter
                }
            }
            return r;
        };

        // k16-parity: q8_0 K/V above 8K context where it buys device residency (Strata's setup: fp16 K/V up to 8K context, int8 above,
        // 8-bit codes + an fp16 scale per 64 values; ours: q8_0, 8-bit codes + an fp16 scale per 32). Only without -ctk / -ctv (or their
        // LLAMA_ARG_ variables), with flash attention not off, and only when the f16 placement spills (layers on the CPU, the MTP drafter
        // dropped, or experts in system memory). Both K/V types are fitted on copies, each with the MTP drop rule below applied, and
        // q8_0 is kept when it gives more context, more layers on the device, the MTP drafter that f16 drops, the experts on the device,
        // or a MoE cache larger by max(64 MiB, 1/8); otherwise the fit runs as before with f16. The rest of the fit then runs on the chosen
        // type as usual. Lossy (KV quantisation). LLAMA_KV_AUTO_Q8=0: off (f16 as before); LLAMA_KV_AUTO_Q8_MIN_CTX (8192): the context
        // above which it applies
        {
            const char * env_aq = getenv("LLAMA_KV_AUTO_Q8");
            const char * env_mc = getenv("LLAMA_KV_AUTO_Q8_MIN_CTX");
            const char * env_ar = getenv("LLAMA_KV_AUTO_Q8_ARCHS");
            const int     aq_mode = env_aq != nullptr ? atoi(env_aq) : 1;
            const int64_t min_ctx = env_mc != nullptr ? atoll(env_mc) : 8192;
            // k16-parity G4 (k16p_g4c, q3m 4 GB + MTP): the two trades below win decode but lose the prompt with q8_0 K/V (20k: decode
            // +12..19% over f16, 2k / 8k prompt -9% / -34%; 32k: +5..15%, -40% / -60%; 16k "a larger MoE cache": decode -4..-9%), so by
            // default q8_0 is kept only for more context, more layers or every expert on the device. LLAMA_KV_AUTO_Q8_TRADE=1: also for the
            // MTP drafter that f16 drops and for a MoE cache larger by max(64 MiB, 1/8)
            const bool aq_trade = getenv("LLAMA_KV_AUTO_Q8_TRADE") != nullptr && atoi(getenv("LLAMA_KV_AUTO_Q8_TRADE")) != 0;
            // (k19-kvwin: not beside the KV window ring, whose cache types are given)
            bool aq_on = aq_mode != 0 && !model_only && !params.cache_type_k_set && !params.cache_type_v_set && params.kv_win < 0 &&
                params.cache_type_k == GGML_TYPE_F16 && params.cache_type_v == GGML_TYPE_F16 &&
                params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && !params.embedding &&
                (params.n_ctx <= 0 || (int64_t) params.n_ctx > min_ctx);
            // only the architectures whose q8_0 K/V was measured inside the reassociation band (LLAMA_KV_AUTO_Q8=2: every architecture;
            // LLAMA_KV_AUTO_Q8_ARCHS=a,b,...: this list). qwen35moe (Qwen3.6-35B-A3B) and qwen4exp (Qwen3.8-Flash-Next): teacher-forced KLD
            // of q8_0 vs f16 K/V at positions 8K-16K inside the -ub 256 / -fa off band. Not lfm2moe (LFM2.5-24B-A2B): on short prompts the
            // first token's KLD is 6x the -ub 64 band (0.27 vs 0.044, top-1 0.80 vs 0.97)
            if (aq_on && aq_mode != 2) {
                const std::string arch = common_gguf_arch(params.model.path);
                const std::string list = std::string(",") + (env_ar != nullptr ? env_ar : "qwen35moe,qwen4exp") + ",";
                aq_on = !arch.empty() && list.find("," + arch + ",") != std::string::npos;
            }
            if (aq_on) {
                const kv_probe a = probe(GGML_TYPE_F16, cparams.offload_kqv, 0);
                const bool spilled = a.ok && (a.layers < n_layer + 1 || (spec_mtp && !has_draft && !a.mtp) || a.n_host > 0);
                if (a.ok && spilled && (int64_t) a.n_ctx > min_ctx && n_layer > 0) {
                    const kv_probe b = probe(GGML_TYPE_Q8_0, cparams.offload_kqv, 0);
                    const int64_t mib = 1024*1024;
                    const char * why = nullptr;
                    if (b.ok) {
                        if (b.n_ctx > a.n_ctx) {
                            why = "more context";
                        } else if (b.n_ctx == a.n_ctx && b.layers > a.layers) {
                            why = "more layers on the device";
                        } else if (b.n_ctx == a.n_ctx && b.layers == a.layers && b.mtp == a.mtp && a.n_host > 0 && b.n_host == 0) {
                            why = "every expert on the device";
                        } else if (aq_trade && b.n_ctx == a.n_ctx && b.layers == a.layers && b.mtp && !a.mtp) {
                            why = "the MTP drafter stays";
                        } else if (aq_trade && b.n_ctx == a.n_ctx && b.layers == a.layers && b.mtp == a.mtp && b.n_host > 0 &&
                                b.cache >= a.cache + std::max<int64_t>(64*mib, a.cache/8)) {
                            why = "a larger MoE cache";
                        }
                    }
                    if (why != nullptr) {
                        params.cache_type_k = GGML_TYPE_Q8_0;
                        params.cache_type_v = GGML_TYPE_Q8_0;
                        cparams.type_k      = GGML_TYPE_Q8_0;
                        cparams.type_v      = GGML_TYPE_Q8_0;
                    }
                    COM_INF("%s: --fit: K/V cache %s at ctx %u: %s (f16: %" PRId64 " / %" PRId64 " layers, MTP %d, MoE cache %" PRId64 " MiB; "
                        "q8_0: %" PRId64 " layers, MTP %d, MoE cache %" PRId64 " MiB%s; LLAMA_KV_AUTO_Q8=0 keeps f16)\n", __func__,
                        why != nullptr ? "q8_0" : "f16", a.n_ctx, why != nullptr ? why : "q8_0 buys no device residency", a.layers, n_layer + 1,
                        (int) a.mtp, a.cache/mib, b.layers, (int) b.mtp, b.cache/mib, b.ok ? "" : ", fit failed");
                }
            }
        }

        // k19-kvlong: a long context on a small card. The K/V and the compute buffers grow with n_ctx (the flash attention mask and the QSA
        // selection masks with n_ctx x n_ubatch), and the fit moves whole layers to the host to make room for them: Qwen3.8-Flash-Next at
        // ctx 262144 on 10 GB (q8_0 K/V) kept 17 of 49 layers on the device, with dense weights 5597 MiB, the compute buffer 4127 MiB at
        // n_ubatch 2048 and the K/V 4192 MiB. When the placement spills layers, two other trades are fitted first:
        // - a smaller n_ubatch (halved down to LLAMA_FIT_LONGCTX_UB_MIN, 512): the prompt runs in more, smaller steps.
        //   LLAMA_FIT_LONGCTX_UB: 2 (default since k19-defaults1116) also a given -ub (Studio passes -ub 2048: G4 Flash-Next 12 GB ctx
        //   262144 kept 30 of 49 layers at 29 tok/s, 10 GB 17 of 49 at 23), 1 only without -ub, 0 never;
        // - opt-in (LLAMA_FIT_LONGCTX_KV_HOST=1): the K/V in pinned host memory (-nkvo; the recurrent states stay on the device). The
        //   attention still runs on the GPU from a copy of the cells in use, so each decode step copies the whole used K/V of every
        //   attention layer: G4 10 GB, Flash-Next ctx 262144, n_ubatch 1024: every layer on the device and a 3 GB MoE cache, chat
        //   21.4 -> 31.4 tok/s and the 8k prompt 523 -> 918 tok/s, but decode after it 19.9 -> 18.8, and lower still as the context
        //   grows (a QSA layer reads only its selected cells: a gather of those would remove that cost).
        // - opt-in (LLAMA_FIT_LONGCTX_KV4=1) when that still spills layers, or leaves no MoE cache beside host experts: a 4-bit
        //   K/V (q4_0, LLAMA_FIT_LONGCTX_KV4_TYPE) with the same n_ubatch ladder, only without -ctk / -ctv / --kv-window (or LLAMA_KV_WIN)
        //   and for LLAMA_FIT_LONGCTX_KV4_ARCHS (qwen35moe,qwen4exp). k19-kvwin4: in KV window mode by default, the last
        //   LLAMA_FIT_LONGCTX_KV4_WIN (16384) positions and the first LLAMA_FIT_LONGCTX_KV4_SINKS (4) also in a q8_0 ring beside the q4_0
        //   cache (LLAMA_FIT_LONGCTX_KV4_WIN=0: the plain 4-bit cache). G4 10 GB, Flash-Next ctx 262144: q4_0 at n_ubatch 512 keeps every
        //   layer and a 2 GB MoE cache on the device (compute buffer 583 MiB against 1841 MiB with q8_0): chat 21.4 -> 83.8 tok/s, decode
        //   after an 8k prompt 19.9 -> 78.5. The window size is the smallest that kept every paired KLD CI against q8_0 at 0 (Flash-Next,
        //   ctx 262144, whole card, long48 = 48 prompts of 2.1k / 8.2k and long20k = 16 of 20k tokens; first-token KLD and top-1, decode
        //   KLD): 1024 failed the long48 first token (+.0174 [+.0053, +.0334]), 8192 the long20k decode (+.0026 [+.0003, +.0047]); 16384
        //   passed all (ring 210 MiB at n_ubatch 512). Past the window the cache is 4-bit: longer prompts than these are not gated.
        // A trade is kept only when it puts more layers on the device (then the larger MoE cache, then the MTP drafter); the K/V in host
        // memory is preferred over the smaller n_ubatch when both give the same layers (LLAMA_FIT_LONGCTX_PREFER=ub: the other way).
        // Only with flash attention not off, the K/V placement and -ngl not given, a MoE model and n_ctx above LLAMA_FIT_LONGCTX_MIN_CTX
        // (32768). Exact for the K/V placement; a smaller n_ubatch changes the prompt's batch shapes. LLAMA_FIT_LONGCTX=0: off.
        {
            const auto env_i = [](const char * name, int64_t def) {
                const char * e = getenv(name);
                return e != nullptr && e[0] != 0 ? (int64_t) atoll(e) : def;
            };
            const int64_t lc_mode  = env_i("LLAMA_FIT_LONGCTX", 1);
            const int64_t lc_min   = env_i("LLAMA_FIT_LONGCTX_MIN_CTX", 32768);
            const int64_t lc_host  = env_i("LLAMA_FIT_LONGCTX_KV_HOST", 0);
            const int64_t lc_ub    = env_i("LLAMA_FIT_LONGCTX_UB", 2);
            const int64_t lc_ubmin = std::max<int64_t>(64, env_i("LLAMA_FIT_LONGCTX_UB_MIN", 512));
            const char *  lc_pref  = getenv("LLAMA_FIT_LONGCTX_PREFER");
            const bool    pref_ub  = lc_pref != nullptr && strcmp(lc_pref, "ub") == 0;
            bool has_exps = false;
            for (const uint8_t e : exps) {
                has_exps = has_exps || e != 0;
            }
            // not on the disk tier (the host experts do not fit in the available RAM and the model is not on a memory file system): there a
            // prompt ubatch reads the experts it misses from the disk, so a smaller n_ubatch reads them more often. Colab L4 10 GB,
            // Flash-Next from the VM disk (57.6 GB of experts, 51 GB of RAM), ctx 262144, n_ubatch 2048 -> 512: decode after an 8k prompt
            // 1.3 -> 1.8 tok/s but the prompt 18 -> 8 tok/s. LLAMA_FIT_LONGCTX_DISK=1 lowers it there too
            const bool disk_tier = [&] {
                if (env_i("LLAMA_FIT_LONGCTX_DISK", 0) != 0) {
                    return false;
                }
                const uint64_t exp_b = common_moe_expert_bytes(params.model.path);
                const uint64_t avail = llama_available_memory();
                return exp_b > 0 && avail > 0 && (double) exp_b > 0.8*(double) avail && !common_fit_file_in_ram_fs(params.model.path);
            }();
            const bool ub_may = !disk_tier && (lc_ub == 2 || (lc_ub == 1 && !params.n_ubatch_set));
            const bool lc_on = lc_mode != 0 && !model_only && !params.embedding && n_layer > 0 && has_exps &&
                params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && cparams.offload_kqv && !params.no_kv_offload &&
                mparams.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                params.n_ctx > 0 && (int64_t) params.n_ctx > lc_min &&
                (lc_host != 0 || ub_may || env_i("LLAMA_FIT_LONGCTX_KV4", 0) != 0) &&
                [] {
                    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                            return true;
                        }
                    }
                    return false;
                }();
            if (lc_on) {
                const ggml_type t = cparams.type_k == cparams.type_v ? cparams.type_k : GGML_TYPE_COUNT;
                const kv_probe a = t != GGML_TYPE_COUNT ? probe(t, true, 0) : kv_probe{};
                // also where every layer fits but experts stay in host memory, with LLAMA_FIT_LONGCTX_UB=2 (or LLAMA_FIT_LONGCTX_CACHE=1):
                // at a long context the compute buffer takes what the MoE cache and a drafter would get. G4 16 GB, Flash-Next ctx 262144,
                // -ub 2048: 4924 MiB of compute buffer, no MoE cache, no drafter, chat 37 tok/s (12 GB with n_ubatch 512: 83)
                const bool lc_cache = a.ok && a.layers >= n_layer + 1 && a.n_host > 0 && ub_may &&
                    (lc_ub == 2 || env_i("LLAMA_FIT_LONGCTX_CACHE", 0) != 0);
                if (a.ok && (a.layers < n_layer + 1 || lc_cache)) {
                    // better: more layers, then more MoE cache, then the MTP drafter
                    const auto better = [](const kv_probe & x, const kv_probe & y) {
                        if (!x.ok) {
                            return false;
                        }
                        if (!y.ok) {
                            return true;
                        }
                        if (x.n_ctx != y.n_ctx) {
                            return x.n_ctx > y.n_ctx;
                        }
                        if (x.layers != y.layers) {
                            return x.layers > y.layers;
                        }
                        if (x.mtp != y.mtp) {
                            return x.mtp;
                        }
                        return x.cache > y.cache + std::max<int64_t>(64ll*1024*1024, y.cache/8);
                    };
                    std::vector<kv_probe> tried;
                    kv_probe host;
                    if (lc_host != 0) {
                        host = probe(t, false, 0);
                        tried.push_back(host);
                    }
                    kv_probe lowub;
                    if (ub_may) {
                        // halve n_ubatch until every layer is on the device (or the floor)
                        for (uint32_t ub = cparams.n_ubatch/2; (int64_t) ub >= lc_ubmin; ub /= 2) {
                            const kv_probe c = probe(t, true, ub);
                            tried.push_back(c);
                            if (better(c, lowub)) {
                                lowub = c;
                            }
                            if (c.ok && c.layers >= n_layer + 1) {
                                break;
                            }
                        }
                    }
                    kv_probe both;
                    const bool host_full = host.ok && host.layers >= n_layer + 1;
                    const bool ub_full   = lowub.ok && lowub.layers >= n_layer + 1;
                    if (lc_host != 0 && ub_may && !host_full && !ub_full) {
                        for (uint32_t ub = cparams.n_ubatch/2; (int64_t) ub >= lc_ubmin; ub /= 2) {
                            const kv_probe c = probe(t, false, ub);
                            tried.push_back(c);
                            if (better(c, both)) {
                                both = c;
                            }
                            if (c.ok && c.layers >= n_layer + 1) {
                                break;
                            }
                        }
                    }
                    // the preferred trade first: the other one must be strictly better to win
                    kv_probe best = a;
                    const kv_probe & first  = pref_ub ? lowub : host;
                    const kv_probe & second = pref_ub ? host  : lowub;
                    for (const kv_probe * c : std::initializer_list<const kv_probe *>{ &first, &second, &both }) {
                        if (better(*c, best)) {
                            best = *c;
                        }
                    }
                    // the 4-bit K/V step (opt-in): the same ladder with the K/V on the device (k19-kvwin4: in KV window mode)
                    const bool kv4_on = env_i("LLAMA_FIT_LONGCTX_KV4", 0) != 0 && !params.cache_type_k_set && !params.cache_type_v_set &&
                        params.kv_win < 0 && getenv("LLAMA_KV_WIN") == nullptr && getenv("LLAMA_KV_WIN_SINKS") == nullptr &&
                        (t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0);
                    if (kv4_on && best.ok && (best.layers < n_layer + 1 || (best.n_host > 0 && best.cache == 0))) {
                        const char * e_t = getenv("LLAMA_FIT_LONGCTX_KV4_TYPE");
                        ggml_type t4 = GGML_TYPE_Q4_0;
                        if (e_t != nullptr && e_t[0] != 0) {
                            for (int i = 0; i < GGML_TYPE_COUNT; i++) {
                                const char * n = ggml_type_name((ggml_type) i);
                                if (n != nullptr && strcmp(n, e_t) == 0) {
                                    t4 = (ggml_type) i;
                                }
                            }
                        }
                        const char * e_ar = getenv("LLAMA_FIT_LONGCTX_KV4_ARCHS");
                        const std::string arch = common_gguf_arch(params.model.path);
                        const std::string archs = std::string(",") + (e_ar != nullptr ? e_ar : "qwen35moe,qwen4exp") + ",";
                        if (!arch.empty() && archs.find("," + arch + ",") != std::string::npos) {
                            const int64_t w4 = env_i("LLAMA_FIT_LONGCTX_KV4_WIN", 16384);
                            probe_win       = w4 > 0 ? (int32_t) w4 : -1;
                            probe_win_sinks = w4 > 0 ? (int32_t) std::max<int64_t>(0, env_i("LLAMA_FIT_LONGCTX_KV4_SINKS", 4)) : 0;
                            kv_probe best4;
                            for (uint32_t ub = cparams.n_ubatch; ub >= 1 && (ub == cparams.n_ubatch || ((int64_t) ub >= lc_ubmin && ub_may)); ub /= 2) {
                                const kv_probe c = probe(t4, true, ub == cparams.n_ubatch ? 0 : ub);
                                tried.push_back(c);
                                if (better(c, best4)) {
                                    best4 = c;
                                }
                                if (c.ok && c.layers >= n_layer + 1) {
                                    break;
                                }
                            }
                            probe_win = -1;
                            probe_win_sinks = 0;
                            if (better(best4, best)) {
                                best = best4;
                            }
                        }
                    }
                    std::string list;
                    for (const auto & c : tried) {
                        list += string_format("%s%s%s %s ub %u: %s", list.empty() ? "" : "; ", ggml_type_name(c.type),
                            c.win >= 0 ? string_format(" + q8_0 window %d", c.win).c_str() : "", c.kvo ? "K/V device" : "K/V host", c.ub,
                            c.ok ? string_format("%" PRId64 " layers, MoE cache %" PRId64 " MiB", c.layers, c.cache/(1024*1024)).c_str() : "fit failed");
                    }
                    const bool changed = best.ok && (best.kvo != a.kvo || best.ub_req != 0 || best.type != a.type || best.win >= 0);
                    if (changed) {
                        if (best.win >= 0) {
                            // the KV window ring beside the 4-bit cache (as --kv-window / --kv-sinks)
                            params.kv_win        = best.win;
                            params.kv_win_sinks  = (int32_t) std::max<int64_t>(0, env_i("LLAMA_FIT_LONGCTX_KV4_SINKS", 4));
                            cparams.kv_win       = params.kv_win;
                            cparams.kv_win_sinks = params.kv_win_sinks;
                        }
                        if (!best.kvo) {
                            params.no_kv_offload    = true;
                            cparams.offload_kqv     = false;
                            cparams_dft.offload_kqv = false;
                        }
                        if (best.type != a.type) {
                            params.cache_type_k = best.type;
                            params.cache_type_v = best.type;
                            cparams.type_k      = best.type;
                            cparams.type_v      = best.type;
                        }
                        if (best.ub_req != 0) {
                            // given from here on, so the MoE cache rules do not grow it back
                            params.n_ubatch     = (int32_t) best.ub_req;
                            params.n_ubatch_set = true;
                            cparams.n_ubatch    = best.ub_req;
                        }
                    }
                    COM_INF("%s: --fit: ctx %u: %s (as given: %" PRId64 " / %" PRId64 " layers, MoE cache %" PRId64 " MiB, n_ubatch %u; %s; "
                        "LLAMA_FIT_LONGCTX=0 keeps it)\n", __func__, a.n_ctx,
                        !changed ? "no long-context trade puts more on the device" :
                        string_format("K/V %s%s %s, n_ubatch %u: %" PRId64 " layers, MoE cache %" PRId64 " MiB", ggml_type_name(best.type),
                            best.win >= 0 ? string_format(" + q8_0 window of the last %d and first %d positions", best.win,
                                (int) std::max<int64_t>(0, env_i("LLAMA_FIT_LONGCTX_KV4_SINKS", 4))).c_str() : "",
                            best.kvo ? "on the device" : "in host memory", best.ub, best.layers, best.cache/(1024*1024)).c_str(),
                        a.layers, n_layer + 1, a.cache/(1024*1024), a.ub, list.c_str());
                }
            }
        }

        bool draft_cpu = false;
        bool draft_needs_target = false;
        {
            const char * env_last = getenv("LLAMA_FIT_DRAFT_LAST");
            const bool   last     = has_draft && params.moe_cache_size > 0 && params.speculative.draft.n_gpu_layers == -1 &&
                params.speculative.draft.devices.empty() &&
                (env_last == nullptr || atoi(env_last) != 0);
            if (last) {
                const bool rq_before = getenv("LLAMA_DRAFT_REQUANT") != nullptr;
                common_fit_draft_hx hx;
                draft_cpu = common_fit_draft_on_cpu(params, mparams, cparams, mparams_dft, cparams_dft, params_dft.model.path, fit_log_level,
                    &draft_needs_target, &hx);
                if (!draft_cpu && hx.on) {
                    // k19-mtpkeep: the draft runs with its experts in host memory: the draft params (the context the server creates) and the
                    // copies the rest of the fit measures it with (the extra model of the layers-first rule)
                    for (auto * v : { &params.speculative.draft.tensor_buft_overrides, &params_dft.tensor_buft_overrides }) {
                        if (!v->empty() && v->back().pattern == nullptr) {
                            v->pop_back();
                        }
                        v->push_back(llm_ffn_exps_cpu_override());
                        v->push_back({ nullptr, nullptr });
                    }
                    mparams_dft.tensor_buft_overrides     = params_dft.tensor_buft_overrides.data();
                    params.speculative.draft.n_ubatch     = (int32_t) hx.n_ubatch;
                    params.speculative.draft.cache_type_k = hx.type_k;
                    params.speculative.draft.cache_type_v = hx.type_v;
                    cparams_dft.n_ubatch = hx.n_ubatch;
                    cparams_dft.type_k   = hx.type_k;
                    cparams_dft.type_v   = hx.type_v;
                }
                if (!draft_cpu && !rq_before && getenv("LLAMA_DRAFT_REQUANT") != nullptr) {
                    // k18: the draft was admitted requantized: the rest of the fit (the extra model) measures it at those types, as it loads
                    // (the overrides are added in place: the extra model holds params_dft's path and &mparams_dft)
                    if (!params_dft.kv_overrides.empty() && params_dft.kv_overrides.back().key[0] == 0) {
                        params_dft.kv_overrides.pop_back();
                    }
                    for (const auto & kv : { std::make_pair("general.requant_type", "LLAMA_DRAFT_REQUANT"),
                                             std::make_pair("general.requant_output_type", "LLAMA_DRAFT_REQUANT_OUTPUT") }) {
                        const char * v = getenv(kv.second);
                        if (v == nullptr || v[0] == 0) {
                            continue;
                        }
                        llama_model_kv_override o = {};
                        snprintf(o.key, sizeof(o.key), "%s", kv.first);
                        o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                        snprintf(o.val_str, sizeof(o.val_str), "%s", v);
                        params_dft.kv_overrides.push_back(o);
                    }
                    params_dft.kv_overrides.emplace_back();
                    params_dft.kv_overrides.back().key[0] = 0;
                    mparams_dft.kv_overrides = params_dft.kv_overrides.data();
                }
            }
        }
        const char * env_dcpu  = getenv("LLAMA_FIT_DRAFT_CPU");
        // k15-small4: a draft that runs on the target's context (a gemma-4 assistant reads the target's KV cache) cannot run on the CPU
        // while that cache is on the device: T4 4 GB with LLAMA_FIT_DRAFT_CPU=1 aborted in its first graph reservation (a CPU context
        // with a pre-allocated CUDA tensor). It is left out instead
        if (draft_cpu && draft_needs_target && env_dcpu != nullptr && atoi(env_dcpu) != 0) {
            COM_WRN("%s: LLAMA_FIT_DRAFT_CPU=1: the draft model runs on the target's KV cache and cannot run on the CPU beside it, it is left out\n", __func__);
        }
        const bool   draft_off = draft_cpu && params.speculative.draft.fit_can_drop && (env_dcpu == nullptr || atoi(env_dcpu) == 0 || draft_needs_target);
        // k15-ov: the context's prefetch depth rule treats a target whose draft --fit dropped as running without one (no device draft
        // competes with the deeper staging); cparams.spec_draft stays set, as it also turns on layer-major prompts (llama_context::sched_new)
        common_set_env("LLAMA_FIT_DRAFT_DROPPED", draft_off ? "1" : "0");
        if (draft_cpu) {
            // no device at all (-devd none): with only n_gpu_layers = 0 its context still reserves a device compute buffer for the
            // offloaded ops of large batches (2.4 GiB for the MTP drafter at -ub 2048), which pushed the main layers off the GPU
            params.speculative.draft.devices      = { nullptr };
            params.speculative.draft.n_gpu_layers = 0;
            mparams_dft.devices      = params.speculative.draft.devices.data();
            mparams_dft.n_gpu_layers = 0;
            params.speculative.draft.fit_on_cpu   = true;
            params.speculative.draft.fit_off      = draft_off;
            if (draft_off) {
                // fitted and run as without a draft model, the caller turns drafting off
                cparams.n_rs_seq = 0;
                mparams.load_mtp = params.load_mtp;
                // k19-rollback: or drafts n-gram windows instead (LLAMA_SPEC_DROP_NGRAM)
                common_fit_drop_ngram(params, cparams);
            }
        }

        // k15-pp4: the state capture snapshots (layer-major prompts beside a draft, llama_context::cap_planned_bytes) are counted by the
        // fit; where counting them costs device layers, the capture is dropped (llama_context_params::n_state_capture = 0) before any
        // layer: G4 q3m 4 GB + MTP, ctx 16384: two snapshots (126 MiB) took the small-device placement's whole 114 MiB cache, and 14 of
        // 42 layers went to the CPU. LLAMA_FIT_CAPTURE_SHRINK=0 keeps the count. Placement only: the server then checkpoints as without
        // capture (LLAMA_LM_CKPT_CAPTURE=0), whose output it equals
        const llama_model_params   mparams_in  = mparams;
        const llama_context_params cparams_in  = cparams;
        const std::vector<float>   ts_in(params.tensor_split, params.tensor_split + 128);
        const std::vector<llama_model_tensor_buft_override> tbo_in = params.tensor_buft_overrides;
        const std::vector<size_t>  margins_in  = params.fit_params_target;
        auto run_fit = [&](int32_t n_cap) {
            mparams = mparams_in;
            cparams = cparams_in;
            cparams.n_state_capture = n_cap;
            std::copy(ts_in.begin(), ts_in.end(), params.tensor_split);
            params.tensor_buft_overrides = tbo_in;
            params.fit_params_target     = margins_in;
            if (mparams_in.tensor_buft_overrides != nullptr && !params.tensor_buft_overrides.empty()) {
                mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
            }
            common_fit_set_moe_margin_auto(!params.fit_params_target_set);
            return common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split,
                params.tensor_buft_overrides.data(),
                params.fit_params_target.data(),
                params.fit_params_min_ctx,
                (has_draft || spec_mtp) && !draft_off ? &extra : nullptr,
                fit_log_level,
                params.fit_kv_resident);
        };
        common_params_fit_status fit_status = run_fit(cparams.n_state_capture);
        {
            // LLAMA_FIT_CAPTURE_SHRINK: 2 (default) no capture; 1 one snapshot first, then none; 0 keeps the count. G4 q3m 4 GB + MTP,
            // ctx 16384 (k15g4s1): one counted snapshot keeps the 42 layers but its 61 MiB leave the first layer-major group 98 MiB free
            // (split at 2), 2k / 8k prompt 3395 / 3266 tok/s; no capture gives k14's groups and md5, 5264 / 4812 (k14 5229 / 5429)
            static const int shrink = getenv("LLAMA_FIT_CAPTURE_SHRINK") == nullptr ? 2 : atoi(getenv("LLAMA_FIT_CAPTURE_SHRINK"));
            std::vector<uint8_t> exps;
            const int64_t n_layer = common_model_n_layer(params.model.path, exps);
            const auto layers = [&]() { return mparams.n_gpu_layers < 0 ? n_layer + 1 : (int64_t) mparams.n_gpu_layers; };
            const bool counted = getenv("LLAMA_FIT_CAPTURE") != nullptr && atoi(getenv("LLAMA_FIT_CAPTURE")) != 0;   // llama_context::cap_planned_bytes
            if (counted && shrink > 0 && cparams_in.n_state_capture < 0 && cparams_in.spec_draft && !draft_off && n_layer > 0 &&
                    getenv("LLAMA_STATE_CAPTURE_MAX") == nullptr && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && layers() < n_layer + 1) {
                const int64_t l_full = layers();
                int64_t best_l = l_full;
                int32_t best_n = -1;
                for (const int32_t n : { 1, 0 }) {
                    if (n == 1 && shrink == 2) {
                        continue;
                    }
                    const auto st = run_fit(n);
                    if (st == COMMON_PARAMS_FIT_STATUS_SUCCESS && layers() > best_l) {
                        best_l = layers();
                        best_n = n;
                        if (best_l >= n_layer + 1) {
                            break;
                        }
                    }
                }
                if (best_n != (int32_t) cparams.n_state_capture) {
                    fit_status = run_fit(best_n);
                }
                COM_INF("%s: --fit: the state capture snapshots would move %" PRId64 " layers to the CPU: %s (%" PRId64 " of %" PRId64 " layers "
                    "in device memory; LLAMA_FIT_CAPTURE_SHRINK=0 keeps them)\n", __func__, n_layer + 1 - l_full,
                    best_n < 0 ? "kept, fewer snapshots do not keep more layers" : best_n == 0 ? "no capture" : "one snapshot",
                    layers(), n_layer + 1);
            }
            // k15-pp5: with the snapshots not counted (LLAMA_FIT_CAPTURE unset, the default) they are allocated at the first capture from
            // what the fit left free. On a small-margin placement (common_fit_small_device: the margin dropped below the default beside a
            // cache of at most 256 MiB) or one that did not keep every layer in device memory there is no room for them: G4 q3m 4 GB + MTP,
            // ctx 16384 ran out of device memory at the first prompt save in 2 of 2 launches (k15g4c8), with LLAMA_STATE_CAPTURE_MAX=1 too.
            // The capture is dropped there as above (the server checkpoints as with LLAMA_LM_CKPT_CAPTURE=0; md5 = k14); the placement
            // is the fit's, unchanged. LLAMA_FIT_CAPTURE_GUARD=0 keeps the capture
            // k19-tierfix-fit: also where --fit dropped the draft model: cparams.spec_draft stays set (layer-major prompts), so the
            // server still captures and the snapshots are allocated from the margin all the same. G4 capped at 8 GB, Qwen3.8-Flash-Next
            // UD-IQ4_XS + the dropped Q8_0 MTP drafter: the 8k prompt asked for 112.57 MiB snapshots with 110.56 MiB free (3 times in
            // a prompt). LLAMA_FIT_CAPTURE_GUARD_DROPPED=0 keeps the capture beside a dropped draft
            static const bool guard      = getenv("LLAMA_FIT_CAPTURE_GUARD") == nullptr || atoi(getenv("LLAMA_FIT_CAPTURE_GUARD")) != 0;
            static const bool guard_drop = getenv("LLAMA_FIT_CAPTURE_GUARD_DROPPED") == nullptr || atoi(getenv("LLAMA_FIT_CAPTURE_GUARD_DROPPED")) != 0;
            if (!counted && guard && cparams_in.n_state_capture < 0 && cparams.n_state_capture < 0 && cparams_in.spec_draft && (!draft_off || guard_drop) &&
                    n_layer > 0 && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && (common_fit_small_device() || layers() < n_layer + 1)) {
                cparams.n_state_capture = 0;
                COM_INF("%s: --fit: %s%s: no state capture (its snapshots are not counted and would not fit; LLAMA_FIT_CAPTURE_GUARD=0 keeps it)\n",
                    __func__, common_fit_small_device() ? "small device" : "layers in system memory", draft_off ? ", draft model dropped" : "");
            }
        }
        // k15-small4 (after the k15-c capture shrink, from the same inputs): an MTP drafter (the target's own NextN layers, no -md) that costs whole layers on a small device is dropped, as a
        // draft model that does not fit is (fit_can_drop). Its context (KV, compute buffer at the draft ubatch), the NextN weights and the
        // recurrent rollback state (n_rs_seq) are fitted with the target; where they push layers to the CPU, every token runs those layers
        // (attention, recurrent state) on the CPU: T4 4 GB, Qwen3.6-35B-A3B UD-Q4_K_XL + MTP kept 22 of 42 layers on the device (decode
        // 4-5 tok/s), without the drafter every layer and a 476 MiB MoE cache. Only when the fit with the drafter moved layers off the
        // device is the fit run again without it; the drafter goes if that one keeps every layer on the device.
        // LLAMA_FIT_MTP_DROP=0: keep the drafter and the layer placement
        const char * env_mtp_drop = getenv("LLAMA_FIT_MTP_DROP");
        int64_t moe_gain_first = common_fit_moe_margin_gain();
        bool moe_gain_mtp_trial = false;
        if (spec_mtp && !has_draft && !draft_off && params.speculative.draft.fit_can_drop && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS &&
                mparams.n_gpu_layers != llama_model_default_params().n_gpu_layers && mparams_in.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                (env_mtp_drop == nullptr || atoi(env_mtp_drop) != 0)) {
            llama_model_params   mp = mparams_in;
            llama_context_params cp = cparams_in;
            std::vector<float>   ts = ts_in;
            std::vector<llama_model_tensor_buft_override> tbo = tbo_in;
            std::vector<size_t>  mg = margins_in;
            mp.tensor_split = ts.data();
            if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                mp.tensor_buft_overrides = tbo.data();
            }
            mp.load_mtp = params.load_mtp;
            // k17-small: the trial is the fit of a launch without the drafter (common_fit_plain_launch); LLAMA_FIT_DROP_PLAIN=0: the
            // drafting context's parameters minus its rollback states, as before
            static const bool drop_plain = getenv("LLAMA_FIT_DROP_PLAIN") == nullptr || atoi(getenv("LLAMA_FIT_DROP_PLAIN")) != 0;
            common_params params_plain = params;
            if (drop_plain) {
                common_fit_plain_launch(params_plain, cp);
                common_set_env("LLAMA_SPEC_MTP_HEADS", "0");
            } else {
                cp.n_rs_seq = 0;
                cp.n_state_capture = cparams.n_state_capture;   // k15-c: the capture count the shrink above chose
            }
            common_fit_set_moe_margin_auto(!params.fit_params_target_set);
            const common_params_fit_status st2 = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(), tbo.empty() ? nullptr : tbo.data(),
                mg.data(), params.fit_params_min_ctx, nullptr, fit_log_level, params.fit_kv_resident);
            if (st2 == COMMON_PARAMS_FIT_STATUS_SUCCESS && mp.n_gpu_layers == llama_model_default_params().n_gpu_layers) {
                COM_WRN("%s: --fit: the MTP drafter would move layers to the CPU (%d layers on the device with it, all without it): "
                    "speculative decoding is off (LLAMA_FIT_MTP_DROP=0 keeps it)\n", __func__, mparams.n_gpu_layers);
                std::copy(ts.begin(), ts.end(), params.tensor_split);
                std::copy(tbo.begin(), tbo.end(), params.tensor_buft_overrides.begin());
                params.fit_params_target = mg;
                const bool tbo_ours = mp.tensor_buft_overrides != nullptr && mp.tensor_buft_overrides == tbo.data();
                mparams = mp;
                cparams = cp;
                mparams.tensor_split = params.tensor_split;
                if (tbo_ours) {   // the fit points the overrides at the buffer it wrote: the copy now lives in params
                    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                }
                fit_status = st2;
                params.speculative.draft.fit_off = true;
                if (drop_plain) {
                    llama_context_params cp_plain = cparams;
                    common_fit_plain_launch(params, cp_plain);
                }
            } else {
                if (drop_plain) {
                    common_set_env("LLAMA_SPEC_MTP_HEADS", "1");
                }
                COM_INF("%s: --fit: the MTP drafter stays (without it the fit does not keep every layer on the device either)\n", __func__);
                // the fit with the drafter stands: its outputs were not touched (the trial ran on copies), only its margin gain is the trial's
                moe_gain_mtp_trial = true;
            }
        }
        // k19-q38t: before the rule below drops a separate MTP drafter that pushes layers of a dense target to a host with few threads,
        // the drafter is fitted again squeezed: its context reads the prompt in ubatches of LLAMA_FIT_DRAFT_SQUEEZE_UB (256) tokens (its
        // compute buffer scales with its ubatch: one block, 569 MiB at -ub 2048 for Qwen3.8-27B's MTP GGUF; it only fills its KV from
        // the target's rows, a few percent of the prompt time) and, where that alone does not keep every layer, beside a margin of
        // LLAMA_FIT_DRAFT_SQUEEZE_MARGIN_MIB (512, the floor Studio keeps on every card) instead of the default 1024 (not below a
        // --fit-target given). The first that keeps every target layer on the device is taken; otherwise the rule below decides as
        // before. Kaggle T4, Qwen3.8-27B UD-Q3_K_XL + its MTP GGUF, ctx 16384: the drafter stays with every layer on the device
        // instead of being dropped. Drafting changes the text as on any GPU where the drafter fits (the exact contract,
        // LLAMA_MTP_EXACT=1 GGML_CUDA_VERIFY_REASSOC=0, keeps the no-draft text). LLAMA_FIT_DRAFT_SQUEEZE=0: off
        bool draft_squeezed = false;
        {
            const char * e_sq = getenv("LLAMA_FIT_DRAFT_SQUEEZE");
            const char * e_sd = getenv("LLAMA_FIT_DRAFT_SPILL_DROP");
            const char * e_st = getenv("LLAMA_FIT_DRAFT_SPILL_MIN_THREADS");
            const char * e_ub = getenv("LLAMA_FIT_DRAFT_SQUEEZE_UB");
            const char * e_mm = getenv("LLAMA_FIT_DRAFT_SQUEEZE_MARGIN_MIB");
            const int     min_threads = e_st != nullptr ? std::max(0, atoi(e_st)) : 8;
            const int32_t n_threads   = params.cpuparams.n_threads;
            const uint32_t sq_ub      = e_ub != nullptr ? (uint32_t) std::max(1, atoi(e_ub)) : 256u;
            const size_t   sq_margin  = (size_t) (e_mm != nullptr ? std::max(0, atoi(e_mm)) : 512) * 1024 * 1024;
            // k19-q38r2: a third step beside a margin of LLAMA_FIT_DRAFT_SQUEEZE_MARGIN2_MIB (448). With block 0 no longer left on the
            // host by the NextN slot (llama-model.cpp), Qwen3.8-27B UD-Q3_K_XL + its MTP GGUF at ctx 16384 needs 14313 of 14806 MiB on a
            // Kaggle T4 for every layer (493 MiB free): 19 MiB short of 512. Where no step keeps every layer, the 512 MiB step's fit is
            // still taken when it leaves only one block on the host (what the drafter ran beside before: the old slot left block 0 there)
            const char * e_m2 = getenv("LLAMA_FIT_DRAFT_SQUEEZE_MARGIN2_MIB");
            const size_t   sq_margin2 = (size_t) (e_m2 != nullptr ? std::max(0, atoi(e_m2)) : 448) * 1024 * 1024;
            if ((e_sq == nullptr || atoi(e_sq) != 0) && (e_sd == nullptr || atoi(e_sd) != 0) && has_draft && spec_mtp && !draft_off &&
                    !params.speculative.draft.fit_off && params.speculative.draft.fit_can_drop && params.speculative.draft.n_ubatch <= 0 &&
                    fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS &&
                    mparams.n_gpu_layers != llama_model_default_params().n_gpu_layers &&
                    mparams_in.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                    n_threads > 0 && n_threads < min_threads && [&] {
                        std::vector<uint8_t> exps;
                        return common_model_n_layer(params.model.path, exps) > 0 && !exps.empty() &&
                            std::all_of(exps.begin(), exps.end(), [](uint8_t x) { return x == 0; });
                    }()) {
                llama_context_params cp_dft_sq = cparams_dft;
                cp_dft_sq.n_ubatch = std::min(cp_dft_sq.n_ubatch, sq_ub);
                common_fit_extra_model extra_sq = extra;
                extra_sq.cparams = &cp_dft_sq;
                const int32_t ngl_spilled = mparams.n_gpu_layers;
                // the fit of the 512 MiB step when it leaves exactly one block on the host (the fallback, see sq_margin2)
                struct sq_fit { llama_model_params mp; llama_context_params cp; std::vector<float> ts; std::vector<llama_model_tensor_buft_override> tbo;
                                std::vector<size_t> mg; common_params_fit_status st; bool set = false; } one_off;
                for (int step = 0; step < 3; ++step) {
                    std::vector<size_t> mg = margins_in;
                    if (step >= 1) {
                        if (params.fit_params_target_set) {
                            break;
                        }
                        const size_t mstep = step == 1 ? sq_margin : sq_margin2;
                        bool lower = false;
                        for (auto & m : mg) {
                            if (m > mstep) {
                                m = mstep;
                                lower = true;
                            }
                        }
                        if (!lower) {
                            continue;
                        }
                    }
                    llama_model_params   mp = mparams_in;
                    llama_context_params cp = cparams_in;
                    std::vector<float>   ts = ts_in;
                    std::vector<llama_model_tensor_buft_override> tbo = tbo_in;
                    mp.tensor_split = ts.data();
                    if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                        mp.tensor_buft_overrides = tbo.data();
                    }
                    common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                    const common_params_fit_status st = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(),
                        tbo.empty() ? nullptr : tbo.data(), mg.data(), params.fit_params_min_ctx, &extra_sq, fit_log_level, params.fit_kv_resident);
                    // every layer on the device: the fit leaves -ngl at its default ("no changes needed") or sets it to all of them
                    // (trunk + output, the target's NextN block is not loaded beside -md) without moving tensors to the host
                    const auto n_tbo = [](const std::vector<llama_model_tensor_buft_override> & v) {
                        size_t n = 0;
                        for (const auto & o : v) {
                            if (o.pattern == nullptr) {
                                break;
                            }
                            n++;
                        }
                        return n;
                    };
                    std::vector<uint8_t> exps_sq;
                    const int64_t n_layer_sq = common_model_n_layer(params.model.path, exps_sq);
                    const int64_t n_nextn_sq = std::max<int64_t>(0, common_gguf_u32(params.model.path, "nextn_predict_layers"));
                    const bool all_dev = mp.n_gpu_layers == llama_model_default_params().n_gpu_layers ||
                        (n_layer_sq > 0 && mp.n_gpu_layers >= n_layer_sq + 1 - (mp.load_mtp ? 0 : n_nextn_sq) && n_tbo(tbo) == n_tbo(tbo_in));
                    if (st == COMMON_PARAMS_FIT_STATUS_SUCCESS && !all_dev && step == 1 && n_layer_sq > 0 && n_tbo(tbo) == n_tbo(tbo_in) &&
                            mp.n_gpu_layers + 1 == n_layer_sq + 1 - (mp.load_mtp ? 0 : n_nextn_sq)) {
                        one_off = { mp, cp, ts, tbo, mg, st, true };
                    }
                    const bool fits_all     = st == COMMON_PARAMS_FIT_STATUS_SUCCESS && all_dev;
                    const bool take_one_off = !fits_all && step == 2 && one_off.set;
                    common_params_fit_status st_take = st;
                    if (take_one_off) {
                        // the copies are the 512 MiB step's own: re-point the fit's buffers at this step's locals
                        const bool tbo_one = one_off.mp.tensor_buft_overrides != nullptr && one_off.mp.tensor_buft_overrides == one_off.tbo.data();
                        mp = one_off.mp; cp = one_off.cp; ts = one_off.ts; tbo = one_off.tbo; mg = one_off.mg; st_take = one_off.st;
                        mp.tensor_split = ts.data();
                        if (tbo_one) {
                            mp.tensor_buft_overrides = tbo.data();
                        }
                        COM_WRN("%s: --fit: k19-q38r2: the MTP drafter stays with one block on the host (-ngl %d): every layer did not fit beside "
                            "a %zu MiB margin either (LLAMA_FIT_DRAFT_SQUEEZE_MARGIN2_MIB)\n", __func__, mp.n_gpu_layers, sq_margin2/(1024*1024));
                    } else if (fits_all) {
                        COM_WRN("%s: --fit: k19-q38t: the MTP drafter stays with every layer on the device (-ngl %d with tensors on the host before): "
                            "its context reads the prompt in ubatches of %u tokens%s (LLAMA_FIT_DRAFT_SQUEEZE=0 off)\n", __func__, ngl_spilled,
                            cp_dft_sq.n_ubatch, step == 1 ? ", beside a smaller device margin" : step == 2 ? ", beside a margin of LLAMA_FIT_DRAFT_SQUEEZE_MARGIN2_MIB" : "");
                    }
                    if (fits_all || take_one_off) {
                        std::copy(ts.begin(), ts.end(), params.tensor_split);
                        std::copy(tbo.begin(), tbo.end(), params.tensor_buft_overrides.begin());
                        params.fit_params_target = mg;
                        const bool tbo_ours = mp.tensor_buft_overrides != nullptr && mp.tensor_buft_overrides == tbo.data();
                        mparams = mp;
                        cparams = cp;
                        mparams.tensor_split = params.tensor_split;
                        if (tbo_ours) {
                            mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                        }
                        fit_status = st_take;
                        params.speculative.draft.n_ubatch = (int32_t) cp_dft_sq.n_ubatch;
                        cparams_dft.n_ubatch = cp_dft_sq.n_ubatch;
                        draft_squeezed = true;
                        break;
                    }
                }
            }
        }
        // k19-q38d: the k15-small4 rule above for a separate draft model (-md: an MTP head GGUF, a DFlash / EAGLE drafter, a small model)
        // fitted beside a target without a MoE cache, on a host with few threads. Where the drafter's weights, context and verify buffers
        // push whole target layers to the CPU, every token and every verify row runs those layers on the host's few threads: Kaggle T4
        // (2 threads), dense Qwen3.8-27B UD-Q3_K_XL + its MTP head GGUF, ctx 16384: 60 of 66 layers on the device with the drafter, MTP
        // decode 3.9 tok/s (717 ms per pass) against 11.5 without it (every layer on the device). With many host threads the drafts still
        // pay beside spilled layers (G4 48 threads, Qwen3.8-27B UD-Q4_K_XL capped at 16 GB: MTP 43.5 tok/s against 24.7 without), so the
        // rule needs fewer than LLAMA_FIT_DRAFT_SPILL_MIN_THREADS (8) host threads; the fit then runs again without the drafter and drafting
        // is off if that keeps every layer on the device. Exact: the text is the no-draft text. LLAMA_FIT_DRAFT_SPILL_DROP=0: keep the drafter
        {
            const char * e_sd = getenv("LLAMA_FIT_DRAFT_SPILL_DROP");
            const char * e_st = getenv("LLAMA_FIT_DRAFT_SPILL_MIN_THREADS");
            const int    min_threads = e_st != nullptr ? std::max(0, atoi(e_st)) : 8;
            const int32_t n_threads  = params.cpuparams.n_threads;
            if ((e_sd == nullptr || atoi(e_sd) != 0) && has_draft && !draft_off && !params.speculative.draft.fit_off && !draft_squeezed &&
                    params.speculative.draft.fit_can_drop && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS &&
                    mparams.n_gpu_layers != llama_model_default_params().n_gpu_layers &&
                    mparams_in.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                    n_threads > 0 && n_threads < min_threads && [&] {
                        // a target without expert tensors (no MoE cache: the spilled weights are whole layers)
                        std::vector<uint8_t> exps;
                        return common_model_n_layer(params.model.path, exps) > 0 && !exps.empty() &&
                            std::all_of(exps.begin(), exps.end(), [](uint8_t x) { return x == 0; });
                    }()) {
                llama_model_params   mp = mparams_in;
                llama_context_params cp = cparams_in;
                std::vector<float>   ts = ts_in;
                std::vector<llama_model_tensor_buft_override> tbo = tbo_in;
                std::vector<size_t>  mg = margins_in;
                mp.tensor_split = ts.data();
                if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                    mp.tensor_buft_overrides = tbo.data();
                }
                mp.load_mtp = params.load_mtp;   // a plain launch loads no MTP tensors (-md draft-mtp sets load_mtp on the target: +569 MiB on T4)
                common_params params_plain3 = params;
                common_fit_plain_launch(params_plain3, cp);
                const bool tbo_local0 = mp.tensor_buft_overrides != nullptr && mp.tensor_buft_overrides == tbo.data();
                const llama_model_params   mp0 = mp;
                const llama_context_params cp0 = cp;
                const std::vector<float>   ts0 = ts;
                const std::vector<llama_model_tensor_buft_override> tbo0 = tbo;
                common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                common_params_fit_status st4 = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(), tbo.empty() ? nullptr : tbo.data(),
                    mg.data(), params.fit_params_min_ctx, nullptr, fit_log_level, params.fit_kv_resident);
                // k19-spilldrop: at a long context the launch without the drafter can spill too (Kaggle T4, Qwen3.8-27B UD-Q3_K_XL + its
                // MTP GGUF, ctx 32768: 57 of 65 blocks with the drafter, 60 without at the 1024 MiB margin), and the rule above kept the
                // drafter beside 8 host blocks: 3.49 / 2.89 / 2.22 tok/s (chat / 8k / 16k), against 4.41 / 4.09 / 3.60 without it at 60
                // blocks and 8.70 / 9.86 / 9.51 without it with every block on the device beside a 256 MiB margin (205 MiB free at the
                // lowest during a 16k decode). So where the plain fit spills, it is tried again beside the margins of
                // LLAMA_FIT_DRAFT_SPILL_MARGINS_MIB (512,448,256,192; never above a --fit-target given; the drafter launch's plain fit
                // projects 14565 of 14806 MiB there, 14 MiB short of 256): the first that keeps every block
                // drops the drafter; where none does, the drafter still goes if the plain fit keeps more blocks on the device than the
                // fit with it (every host block costs each verify pass as much as a token). LLAMA_FIT_DRAFT_SPILL_LONG=0: the rule as before
                const char * e_sl = getenv("LLAMA_FIT_DRAFT_SPILL_LONG");
                const bool spill_long = e_sl == nullptr || atoi(e_sl) != 0;
                const bool all_plain0 = st4 == COMMON_PARAMS_FIT_STATUS_SUCCESS && mp.n_gpu_layers == llama_model_default_params().n_gpu_layers;
                const char * how = "all without it";
                if (spill_long && !all_plain0 && !params.fit_params_target_set) {
                    std::vector<size_t> steps;
                    {
                        const char * e_m = getenv("LLAMA_FIT_DRAFT_SPILL_MARGINS_MIB");
                        const std::string str = e_m != nullptr ? e_m : "512,448,256,192";
                        size_t pos = 0;
                        while (pos < str.size()) {
                            const size_t c = str.find(',', pos);
                            const std::string tok = str.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                            if (!tok.empty()) {
                                steps.push_back((size_t) std::max(0, atoi(tok.c_str())) * 1024 * 1024);
                            }
                            if (c == std::string::npos) {
                                break;
                            }
                            pos = c + 1;
                        }
                    }
                    const llama_model_params   mp_d  = mp;   // the plain fit beside the default margin
                    const llama_context_params cp_d  = cp;
                    const std::vector<float>   ts_d  = ts;
                    const std::vector<llama_model_tensor_buft_override> tbo_d = tbo;
                    const std::vector<size_t>  mg_d  = mg;
                    const common_params_fit_status st_d = st4;
                    bool found = false;
                    for (const size_t m_step : steps) {
                        std::vector<size_t> mgs = margins_in;
                        bool lower = false;
                        for (auto & m : mgs) {
                            if (m > m_step) {
                                m = m_step;
                                lower = true;
                            }
                        }
                        if (!lower) {
                            continue;
                        }
                        llama_model_params   mps = mp0;
                        llama_context_params cps = cp0;
                        std::vector<float>   tss = ts0;
                        std::vector<llama_model_tensor_buft_override> tbos = tbo0;
                        mps.tensor_split = tss.data();
                        if (tbo_local0) {
                            mps.tensor_buft_overrides = tbos.data();
                        }
                        common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                        const common_params_fit_status sts = common_fit_params(params.model.path.c_str(), &mps, &cps, tss.data(),
                            tbos.empty() ? nullptr : tbos.data(), mgs.data(), params.fit_params_min_ctx, nullptr, fit_log_level, params.fit_kv_resident);
                        if (sts == COMMON_PARAMS_FIT_STATUS_SUCCESS && mps.n_gpu_layers == llama_model_default_params().n_gpu_layers) {
                            const bool tbo_s = mps.tensor_buft_overrides != nullptr && mps.tensor_buft_overrides == tbos.data();
                            mp = mps; cp = cps; ts = tss; tbo = tbos; mg = mgs; st4 = sts;
                            mp.tensor_split = ts.data();
                            if (tbo_s) {
                                mp.tensor_buft_overrides = tbo.data();
                            }
                            how = "all without it beside a smaller device margin";
                            COM_INF("%s: --fit: k19-spilldrop: without the drafter every block fits beside a %zu MiB margin\n", __func__,
                                m_step/(1024*1024));
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        // back to the plain fit beside the default margin (the trials ran on copies)
                        const bool tbo_s = mp_d.tensor_buft_overrides != nullptr && mp_d.tensor_buft_overrides == tbo_d.data();
                        mp = mp_d; cp = cp_d; ts = ts_d; tbo = tbo_d; mg = mg_d; st4 = st_d;
                        mp.tensor_split = ts.data();
                        if (tbo_s) {
                            mp.tensor_buft_overrides = tbo.data();
                        }
                        if (st4 == COMMON_PARAMS_FIT_STATUS_SUCCESS && mp.n_gpu_layers > mparams.n_gpu_layers) {
                            how = "more without it";
                            found = true;
                        }
                    }
                    if (found) {
                        COM_INF("%s: --fit: k19-spilldrop: %d layers on the device without the drafter, %d with it (LLAMA_FIT_DRAFT_SPILL_LONG=0: "
                            "keep the drafter unless every layer fits without it)\n", __func__,
                            mp.n_gpu_layers == llama_model_default_params().n_gpu_layers ? -1 : mp.n_gpu_layers, mparams.n_gpu_layers);
                    }
                    if (!found) {
                        st4 = COMMON_PARAMS_FIT_STATUS_FAILURE; // keep the drafter (the condition below does not hold)
                    }
                }
                const bool drop = st4 == COMMON_PARAMS_FIT_STATUS_SUCCESS &&
                    (mp.n_gpu_layers == llama_model_default_params().n_gpu_layers || (spill_long && mp.n_gpu_layers > mparams.n_gpu_layers));
                if (drop) {
                    COM_WRN("%s: --fit: %d host threads: the draft model would move layers to the CPU (%d layers on the device with it, %s): "
                        "speculative decoding is off (LLAMA_FIT_DRAFT_SPILL_DROP=0 keeps it)\n", __func__, n_threads, mparams.n_gpu_layers, how);
                    std::copy(ts.begin(), ts.end(), params.tensor_split);
                    std::copy(tbo.begin(), tbo.end(), params.tensor_buft_overrides.begin());
                    params.fit_params_target = mg;
                    const bool tbo_ours = mp.tensor_buft_overrides != nullptr && mp.tensor_buft_overrides == tbo.data();
                    mparams = mp;
                    cparams = cp;
                    mparams.tensor_split = params.tensor_split;
                    if (tbo_ours) {
                        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                    }
                    fit_status = st4;
                    params.speculative.draft.fit_off = true;
                    common_set_env("LLAMA_FIT_DRAFT_DROPPED", "1");
                    llama_context_params cp_plain = cparams;
                    common_fit_plain_launch(params, cp_plain);
                }
            }
        }
        // k19-kvlong (the order on a capped card: target layers, then the compute buffer at a sane n_ubatch, then K/V, then the drafter,
        // then the MoE cache): a separate draft model beside a MoE target is admitted against the memory of the cache
        // (common_fit_draft_on_cpu), but the fit that then measures it with its context can move target layers to the host. G4 capped at
        // 16 GB, Qwen3.8-Flash-Next UD-IQ4_XS ctx 262144 + its requantized Q8_0 MTP GGUF: 41 of 49 layers with it (every layer without it),
        // decode 37 tok/s, against 83 at 12 GB with every layer and no drafter. Where the fit with the drafter keeps fewer layers on the
        // device than the fit without it, the drafter is left out (drafting off, the plain fit): 37.2 / 35.3 -> 89.7 / 81.8 tok/s chat / 8k
        // there, prompt 1170 -> 5760. LLAMA_FIT_DRAFT_LAYERS_FIRST_UB=N (opt-in) first fits an MTP drafter again with its context
        // reading the prompt in ubatches of N tokens and keeps it if that keeps as many layers: at N = 256 it stayed beside a 3.8 GB
        // cache instead of 5.7 GB and decoded 74.2 / 65.2, slower than without it. Only beside a MoE cache, where -ngl and -ub of the
        // drafter are not given. LLAMA_FIT_DRAFT_LAYERS_FIRST=0: off
        {
            const char * e_lf = getenv("LLAMA_FIT_DRAFT_LAYERS_FIRST");
            const auto   lay  = [&](const llama_model_params & m) {
                return m.n_gpu_layers < 0 ? n_layer + 1 : (int64_t) m.n_gpu_layers;
            };
            if ((e_lf == nullptr || atoi(e_lf) != 0) && has_draft && !draft_off && !params.speculative.draft.fit_off &&
                    params.speculative.draft.fit_can_drop && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && params.moe_cache_size > 0 &&
                    n_layer > 0 && std::any_of(exps.begin(), exps.end(), [](uint8_t x) { return x != 0; }) &&
                    mparams_in.n_gpu_layers == llama_model_default_params().n_gpu_layers && lay(mparams) < n_layer + 1) {
                // the plain fit (no drafter), as k19-spilldrop runs it
                llama_model_params   mp = mparams_in;
                llama_context_params cp = cparams_in;
                std::vector<float>   ts = ts_in;
                std::vector<llama_model_tensor_buft_override> tbo = tbo_in;
                std::vector<size_t>  mg = margins_in;
                mp.tensor_split = ts.data();
                if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                    mp.tensor_buft_overrides = tbo.data();
                }
                mp.load_mtp = params.load_mtp;
                common_params params_plain = params;
                common_fit_plain_launch(params_plain, cp);
                common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                const common_params_fit_status st_p = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(),
                    tbo.empty() ? nullptr : tbo.data(), mg.data(), params.fit_params_min_ctx, nullptr, fit_log_level, params.fit_kv_resident);
                const int64_t l_with  = lay(mparams);
                const int64_t l_plain = st_p == COMMON_PARAMS_FIT_STATUS_SUCCESS ? lay(mp) : -1;
                if (l_plain > l_with) {
                    const char * e_ub  = getenv("LLAMA_FIT_DRAFT_LAYERS_FIRST_UB");
                    const uint32_t lub = e_ub != nullptr ? (uint32_t) std::max(0, atoi(e_ub)) : 0u;
                    bool kept = false;
                    const char * e_lft = getenv("LLAMA_FIT_DRAFT_LAYERS_FIRST_TUB");
                    const char * e_lfm = getenv("LLAMA_FIT_DRAFT_LAYERS_FIRST_TUB_MIN");
                    const uint32_t lft_min = e_lfm != nullptr ? (uint32_t) std::max(32, atoi(e_lfm)) : 256u;
                    // k19-mtpkeep (LLAMA_FIT_DRAFT_HOST_EXPS_LF, default on; LLAMA_FIT_DRAFT_HOST_EXPS=0 turns it off with the rest): before
                    // the drafter goes, an MTP draft model is fitted again with its experts in host memory, n_ubatch 64 and the target's
                    // K / V types (see common_fit_draft_hx), at the target's own n_ubatch; it stays if that keeps as many layers as the plain
                    // fit. G4 16 GB / 262K, Qwen3.8-Flash-Next UD-IQ4_XS + its Q8_0 MTP GGUF: 41 of 49 layers at its file's placement, so
                    // Studio's MTP never drafted there
                    {
                        const char * e_hx  = getenv("LLAMA_FIT_DRAFT_HOST_EXPS");
                        const char * e_hxl = getenv("LLAMA_FIT_DRAFT_HOST_EXPS_LF");
                        bool exps_set = false;
                        for (const auto & o : params.speculative.draft.tensor_buft_overrides) {
                            exps_set = exps_set || (o.pattern != nullptr && strstr(o.pattern, "exps") != nullptr);
                        }
                        if (!kept && spec_mtp && has_draft && !exps_set && (e_hx == nullptr || atoi(e_hx) != 0) && (e_hxl == nullptr || atoi(e_hxl) != 0)) {
                            static const uint32_t hx_ub = getenv("LLAMA_FIT_DRAFT_HOST_EXPS_UB") ? (uint32_t) std::max(1, atoi(getenv("LLAMA_FIT_DRAFT_HOST_EXPS_UB"))) : 64u;
                            std::vector<llama_model_tensor_buft_override> ov_hx;
                            for (const auto & o : params_dft.tensor_buft_overrides) {
                                if (o.pattern != nullptr) {
                                    ov_hx.push_back(o);
                                }
                            }
                            ov_hx.push_back(llm_ffn_exps_cpu_override());
                            ov_hx.push_back({ nullptr, nullptr });
                            llama_model_params   mp_dft_hx = mparams_dft;
                            llama_context_params cp_dft_hx = cparams_dft;
                            mp_dft_hx.tensor_buft_overrides = ov_hx.data();
                            cp_dft_hx.n_ubatch = std::min(cp_dft_hx.n_ubatch, hx_ub);
                            cp_dft_hx.type_k   = cparams.type_k;
                            cp_dft_hx.type_v   = cparams.type_v;
                            common_fit_extra_model extra_hx = extra;
                            extra_hx.mparams = &mp_dft_hx;
                            extra_hx.cparams = &cp_dft_hx;
                            llama_model_params   mq = mparams_in;
                            llama_context_params cq = cparams_in;
                            std::vector<float>   tq = ts_in;
                            std::vector<llama_model_tensor_buft_override> bq = tbo_in;
                            std::vector<size_t>  gq = margins_in;
                            mq.tensor_split = tq.data();
                            if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                                mq.tensor_buft_overrides = bq.data();
                            }
                            common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                            const common_params_fit_status st_q = common_fit_params(params.model.path.c_str(), &mq, &cq, tq.data(),
                                bq.empty() ? nullptr : bq.data(), gq.data(), params.fit_params_min_ctx, &extra_hx, fit_log_level, params.fit_kv_resident);
                            const bool hx_keep = st_q == COMMON_PARAMS_FIT_STATUS_SUCCESS && lay(mq) >= l_plain && cq.n_ubatch == cp.n_ubatch;
                            COM_INF("%s: --fit: k19-mtpkeep: the MTP draft with its experts in host memory (n_ubatch %u, K / V %s / %s): %" PRId64
                                " of %" PRId64 " layers on the device with it, %" PRId64 " without it, target n_ubatch %u -> %s (LLAMA_FIT_DRAFT_HOST_EXPS_LF=0 off)\n",
                                __func__, cp_dft_hx.n_ubatch, ggml_type_name(cp_dft_hx.type_k), ggml_type_name(cp_dft_hx.type_v),
                                st_q == COMMON_PARAMS_FIT_STATUS_SUCCESS ? lay(mq) : (int64_t) -1, n_layer + 1, l_plain, cq.n_ubatch,
                                hx_keep ? "the drafter stays" : "not kept");
                            if (hx_keep) {
                                std::copy(tq.begin(), tq.end(), params.tensor_split);
                                std::copy(bq.begin(), bq.end(), params.tensor_buft_overrides.begin());
                                params.fit_params_target = gq;
                                const bool tbo_q = mq.tensor_buft_overrides != nullptr && mq.tensor_buft_overrides == bq.data();
                                mparams = mq;
                                cparams = cq;
                                mparams.tensor_split = params.tensor_split;
                                if (tbo_q) {
                                    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                                }
                                fit_status = st_q;
                                for (auto * v : { &params.speculative.draft.tensor_buft_overrides, &params_dft.tensor_buft_overrides }) {
                                    if (!v->empty() && v->back().pattern == nullptr) {
                                        v->pop_back();
                                    }
                                    v->push_back(llm_ffn_exps_cpu_override());
                                    v->push_back({ nullptr, nullptr });
                                }
                                mparams_dft.tensor_buft_overrides     = params_dft.tensor_buft_overrides.data();
                                params.speculative.draft.n_ubatch     = (int32_t) cp_dft_hx.n_ubatch;
                                params.speculative.draft.cache_type_k = cp_dft_hx.type_k;
                                params.speculative.draft.cache_type_v = cp_dft_hx.type_v;
                                cparams_dft.n_ubatch = cp_dft_hx.n_ubatch;
                                cparams_dft.type_k   = cp_dft_hx.type_k;
                                cparams_dft.type_v   = cp_dft_hx.type_v;
                                kept = true;
                            }
                        }
                    }
                    if (!kept && lub > 0 && spec_mtp && params.speculative.draft.n_ubatch <= 0 && cparams_dft.n_ubatch > lub) {
                        llama_context_params cp_dft_lf = cparams_dft;
                        cp_dft_lf.n_ubatch = lub;
                        common_fit_extra_model extra_lf = extra;
                        extra_lf.cparams = &cp_dft_lf;
                        llama_model_params   mq = mparams_in;
                        llama_context_params cq = cparams_in;
                        std::vector<float>   tq = ts_in;
                        std::vector<llama_model_tensor_buft_override> bq = tbo_in;
                        std::vector<size_t>  gq = margins_in;
                        mq.tensor_split = tq.data();
                        if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                            mq.tensor_buft_overrides = bq.data();
                        }
                        common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                        const common_params_fit_status st_q = common_fit_params(params.model.path.c_str(), &mq, &cq, tq.data(),
                            bq.empty() ? nullptr : bq.data(), gq.data(), params.fit_params_min_ctx, &extra_lf, fit_log_level, params.fit_kv_resident);
                        if (st_q == COMMON_PARAMS_FIT_STATUS_SUCCESS && lay(mq) >= l_plain) {
                            COM_WRN("%s: --fit: k19-kvlong: the draft model would move layers to the CPU (%" PRId64 " of %" PRId64 " layers on the "
                                "device with it, %" PRId64 " without it): it stays with its context reading the prompt in ubatches of %u tokens "
                                "(LLAMA_FIT_DRAFT_LAYERS_FIRST_UB; LLAMA_FIT_DRAFT_LAYERS_FIRST=0 off)\n", __func__, l_with, n_layer + 1,
                                l_plain, lub);
                            std::copy(tq.begin(), tq.end(), params.tensor_split);
                            std::copy(bq.begin(), bq.end(), params.tensor_buft_overrides.begin());
                            params.fit_params_target = gq;
                            const bool tbo_q = mq.tensor_buft_overrides != nullptr && mq.tensor_buft_overrides == bq.data();
                            mparams = mq;
                            cparams = cq;
                            mparams.tensor_split = params.tensor_split;
                            if (tbo_q) {
                                mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                            }
                            fit_status = st_q;
                            params.speculative.draft.n_ubatch = (int32_t) lub;
                            cparams_dft.n_ubatch = lub;
                            kept = true;
                        }
                    }
                    // k19-kvlong (LLAMA_FIT_DRAFT_LAYERS_FIRST_TUB=1, opt-in): before the drafter goes, the target's n_ubatch is halved (down to
                    // LLAMA_FIT_DRAFT_LAYERS_FIRST_TUB_MIN, 256) and the fit with the drafter runs again; it stays if that keeps as many layers
                    // as the plain fit. On a small device the decode-sized cache must be positive beside the prompt compute buffers before the
                    // k19-place8 step lends it their memory: G4 10 GB Qwen3.8-Flash-Next ctx 262144 + DFlash2 Q4_0 (495 MiB) went 335 -> -160 MiB
                    // at ub 512, so the small-device placement was skipped and 8 layers spilled, while a smaller target ubatch frees that room
                    if (!kept && (e_lft == nullptr ? false : atoi(e_lft) != 0)) {
                        for (uint32_t ub_t = cparams_in.n_ubatch/2; !kept && ub_t >= lft_min; ub_t /= 2) {
                            llama_model_params   mq = mparams_in;
                            llama_context_params cq = cparams_in;
                            std::vector<float>   tq = ts_in;
                            std::vector<llama_model_tensor_buft_override> bq = tbo_in;
                            std::vector<size_t>  gq = margins_in;
                            mq.tensor_split = tq.data();
                            if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                                mq.tensor_buft_overrides = bq.data();
                            }
                            cq.n_ubatch = ub_t;
                            common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                            const common_params_fit_status st_q = common_fit_params(params.model.path.c_str(), &mq, &cq, tq.data(),
                                bq.empty() ? nullptr : bq.data(), gq.data(), params.fit_params_min_ctx, &extra, fit_log_level, params.fit_kv_resident);
                            if (st_q == COMMON_PARAMS_FIT_STATUS_SUCCESS && lay(mq) >= l_plain) {
                                COM_WRN("%s: --fit: k19-kvlong: the draft model would move layers to the CPU (%" PRId64 " of %" PRId64 " layers on the "
                                    "device with it, %" PRId64 " without it): it stays with the target's n_ubatch %u -> %u "
                                    "(LLAMA_FIT_DRAFT_LAYERS_FIRST_TUB)\n", __func__, l_with, n_layer + 1, l_plain, cparams_in.n_ubatch, ub_t);
                                std::copy(tq.begin(), tq.end(), params.tensor_split);
                                std::copy(bq.begin(), bq.end(), params.tensor_buft_overrides.begin());
                                params.fit_params_target = gq;
                                const bool tbo_q = mq.tensor_buft_overrides != nullptr && mq.tensor_buft_overrides == bq.data();
                                mparams = mq;
                                cparams = cq;
                                mparams.tensor_split = params.tensor_split;
                                if (tbo_q) {
                                    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                                }
                                fit_status = st_q;
                                params.n_ubatch     = (int32_t) ub_t;
                                params.n_ubatch_set = true;
                                kept = true;
                            }
                        }
                    }
                    if (!kept) {
                        COM_WRN("%s: --fit: k19-kvlong: the draft model would move layers to the CPU (%" PRId64 " of %" PRId64 " layers on the "
                            "device with it, %" PRId64 " without it): speculative decoding is off (LLAMA_FIT_DRAFT_LAYERS_FIRST=0 keeps it)\n",
                            __func__, l_with, n_layer + 1, l_plain);
                        std::copy(ts.begin(), ts.end(), params.tensor_split);
                        std::copy(tbo.begin(), tbo.end(), params.tensor_buft_overrides.begin());
                        params.fit_params_target = mg;
                        const bool tbo_ours = mp.tensor_buft_overrides != nullptr && mp.tensor_buft_overrides == tbo.data();
                        mparams = mp;
                        cparams = cp;
                        mparams.tensor_split = params.tensor_split;
                        if (tbo_ours) {
                            mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                        }
                        fit_status = st_p;
                        params.speculative.draft.fit_off = true;
                        common_set_env("LLAMA_FIT_DRAFT_DROPPED", "1");
                        llama_context_params cp_plain = cparams;
                        common_fit_plain_launch(params, cp_plain);
                        // the context runs as a launch without a drafter (cparams.spec_draft is off): its prompts keep their layer-major
                        // groups only with this mark, which the MoE cache rules set for a launch without -md (they still see the drafter
                        // here). Without it the G4 16 GB ctx 262144 8k prompt ran ubatch by ubatch, 1860 tok/s against 5743 without -md
                        // (same placement, same text). LLAMA_MOE_LAYER_MAJOR_NODRAFT=0 keeps it off
                        if (const char * e_nd = getenv("LLAMA_MOE_LAYER_MAJOR_NODRAFT"); e_nd == nullptr || atoi(e_nd) != 0) {
                            common_set_env("LLAMA_MOE_LAYER_MAJOR_AUTO_NODRAFT", "1");
                        }
                    }
                }
            }
        }
        // k17-small: on a small device with few host threads an MTP drafter that keeps every layer on the device can still take most of
        // the MoE cache (Qwen3.6-35B-A3B 4 GB, ctx 16384: UD-Q3_K_M cache 31 MiB with it, 699 MiB without; UD-Q4_K_XL 99 vs 535). There
        // the host computes nearly every expert of every drafted verify window on its few threads (Kaggle T4 half, 2 threads, ctx 32768:
        // no drafter + 297 MiB cache decoded 16.0-20.8 tok/s, the drafter beside a 33 MiB cache 14.6-16.6, plans/k15_kaggle.md k15smPC),
        // while hosts with many threads gain from the drafts (G4 48 threads, 16k: the drafter kept is k14's 153 tok/s). So below
        // LLAMA_FIT_MTP_CACHE_MIN_THREADS (4, the spill guard's threshold) host threads, on a small-device placement with the drafter, the
        // fit runs again without it; the drafter goes if that keeps every layer on the device and gives the cache at least
        // LLAMA_FIT_MTP_CACHE_GAIN_MIB (256) and twice its size. The drafter's own text equals the target's where the verify is exact.
        // On by default (LLAMA_FIT_MTP_CACHE_DROP=0 keeps the drafter). G4 KLD against a full-card reference, judged against today's MTP
        // 4 GB default under the same route tail skip setting (k17s_g4d, tail skip off): decode mean / p99 0.00149 / 0.0287 short and
        // 0.00187 / 0.0299 long for this placement, 0.00136 / 0.0244 and 0.00208 / 0.0322 for today's default; with the default tail
        // skip both are further out (0.0061 / 0.0034 short), the skip being the accepted default (Strata's). Kaggle T4 2 threads: decode
        // +29..+49% (q3m 16k), +25% (q4x 16k)
        {
            const char * e_cd = getenv("LLAMA_FIT_MTP_CACHE_DROP");
            const char * e_ct = getenv("LLAMA_FIT_MTP_CACHE_MIN_THREADS");
            const char * e_cg = getenv("LLAMA_FIT_MTP_CACHE_GAIN_MIB");
            const int     min_threads = e_ct != nullptr ? std::max(0, atoi(e_ct)) : 4;
            const int64_t gain_mib    = e_cg != nullptr ? std::max<int64_t>(0, atoll(e_cg)) : 256;
            const int32_t n_threads   = params.cpuparams.n_threads;
            const bool    cache_auto  = moe_cache_auto && cparams.moe_cache_size < COMMON_MOE_CACHE_AUTO_BYTES;
            if ((e_cd == nullptr || atoi(e_cd) != 0) && spec_mtp && !has_draft && !draft_off && !params.speculative.draft.fit_off &&
                    params.speculative.draft.fit_can_drop && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && cache_auto &&
                    mparams.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                    mparams_in.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                    n_threads > 0 && n_threads < min_threads && common_fit_small_device()) {
                const int64_t mib       = 1024*1024;
                const int64_t gain_with = moe_gain_mtp_trial ? moe_gain_first : common_fit_moe_margin_gain();
                const int64_t c_with    = (int64_t) cparams.moe_cache_size + gain_with;
                llama_model_params   mp = mparams_in;
                llama_context_params cp = cparams_in;
                std::vector<float>   ts = ts_in;
                std::vector<llama_model_tensor_buft_override> tbo = tbo_in;
                std::vector<size_t>  mg = margins_in;
                mp.tensor_split = ts.data();
                if (mparams_in.tensor_buft_overrides == params.tensor_buft_overrides.data()) {
                    mp.tensor_buft_overrides = tbo.data();
                }
                mp.load_mtp = params.load_mtp;
                static const bool drop_plain2 = getenv("LLAMA_FIT_DROP_PLAIN") == nullptr || atoi(getenv("LLAMA_FIT_DROP_PLAIN")) != 0;
                common_params params_plain2 = params;
                if (drop_plain2) {
                    common_fit_plain_launch(params_plain2, cp);
                    common_set_env("LLAMA_SPEC_MTP_HEADS", "0");
                } else {
                    cp.n_rs_seq = 0;
                    cp.n_state_capture = cparams.n_state_capture;
                }
                common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                const common_params_fit_status st3 = common_fit_params(params.model.path.c_str(), &mp, &cp, ts.data(), tbo.empty() ? nullptr : tbo.data(),
                    mg.data(), params.fit_params_min_ctx, nullptr, fit_log_level, params.fit_kv_resident);
                const int64_t gain_wo = common_fit_moe_margin_gain();
                // (k19-vramlend: without the unbacked KV a lazy grant added to the cache of the trial, so the keep / drop decision is the
                // one of LLAMA_KV_LAZY=0)
                const int64_t c_wo    = st3 == COMMON_PARAMS_FIT_STATUS_SUCCESS && cp.moe_cache_size < COMMON_MOE_CACHE_AUTO_BYTES ?
                    (int64_t) (cp.moe_cache_size - std::min(cp.moe_cache_size, common_fit_kv_lazy_grant(cp.moe_cache_size))) + gain_wo : -1;
                const bool drop = st3 == COMMON_PARAMS_FIT_STATUS_SUCCESS && mp.n_gpu_layers == llama_model_default_params().n_gpu_layers &&
                    c_wo >= c_with + gain_mib*mib && c_wo >= 2*c_with;
                if (drop) {
                    COM_WRN("%s: --fit: small device, %d host threads: the MTP drafter would take %" PRId64 " of %" PRId64 " MiB of the MoE cache "
                        "(every layer on the device either way): speculative decoding is off (LLAMA_FIT_MTP_CACHE_DROP=0 keeps it)\n", __func__,
                        n_threads, (c_wo - c_with)/mib, c_wo/mib);
                    std::copy(ts.begin(), ts.end(), params.tensor_split);
                    std::copy(tbo.begin(), tbo.end(), params.tensor_buft_overrides.begin());
                    params.fit_params_target = mg;
                    const bool tbo_ours = mp.tensor_buft_overrides != nullptr && mp.tensor_buft_overrides == tbo.data();
                    mparams = mp;
                    cparams = cp;
                    mparams.tensor_split = params.tensor_split;
                    if (tbo_ours) {
                        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                    }
                    fit_status = st3;
                    params.speculative.draft.fit_off = true;
                    moe_gain_mtp_trial = false;   // the margin gain is the trial's (the last fit)
                    if (drop_plain2) {
                        llama_context_params cp_plain = cparams;
                        common_fit_plain_launch(params, cp_plain);
                    }
                } else {
                    if (drop_plain2) {
                        common_set_env("LLAMA_SPEC_MTP_HEADS", "1");
                    }
                    COM_INF("%s: --fit: small device, %d host threads: the MTP drafter stays (MoE cache %" PRId64 " MiB with it, %" PRId64 " without)\n",
                        __func__, n_threads, c_with/mib, c_wo/mib);
                    moe_gain_first     = gain_with;
                    moe_gain_mtp_trial = true;    // the fit with the drafter stands, its margin gain was recorded before the trial
                }
            }
        }
        // k19-fitplain: the k19-spilldrop step-down for a plain launch (no draft model, no MTP heads asked) of a dense model. Where the
        // fit at the default 1024 MiB margin moves whole layers to the CPU, every token runs those layers on the host: Kaggle T4,
        // Qwen3.8-27B UD-Q3_K_XL, ctx 32768: 5 of 65 blocks on the CPU, 4.41 / 4.09 / 3.60 tok/s (chat / 8k / 16k), against 8.70 / 9.86
        // / 9.51 with every block on the device beside a 256 MiB margin (157-159 MiB free at the lowest with 192 MiB). So the fit runs
        // again beside the margins of LLAMA_FIT_PLAIN_SPILL_MARGINS_MIB (512,448,256,192), and the first that keeps every layer on the
        // device is taken; where none does, the default placement stays. Only on devices that drive no display (the CUDA backend's
        // ggml_backend_dev_headless: no kernel run time limit and NVML reports no display active or connected): a desktop's compositor
        // and other apps take and give back device memory while the server runs, which the smaller margin does not cover. An unknown
        // signal (no NVML, another backend, Windows WDDM) keeps 1024 MiB. Not with a --fit-target or -ngl given (the user's),
        // nor for a model with experts (their own small-device margin, k13-small). Placement only: the text is the one of the same
        // launch with -fitt at that margin. LLAMA_FIT_PLAIN_SPILL=0: off; LLAMA_FIT_PLAIN_HEADLESS=1 / 0: take / refuse the step-down
        // whatever the signal (a desktop user who knows the headroom, or a test)
        {
            const char * e_ps = getenv("LLAMA_FIT_PLAIN_SPILL");
            const bool   on   = e_ps == nullptr || atoi(e_ps) != 0;
            // a placement keeps every weight on the device: every layer, and no tensor override beyond the given ones (the
            // KV-resident placement moves FFN tensors of every-layer placements to system memory by overrides)
            const auto n_tbo = [](const llama_model_params & mp) {
                size_t n = 0;
                for (const llama_model_tensor_buft_override * t = mp.tensor_buft_overrides; t != nullptr && t->pattern != nullptr; t++) {
                    n++;
                }
                return n;
            };
            const size_t n_tbo_in = n_tbo(mparams_in);
            const auto all_on_device = [&](const llama_model_params & mp) {
                return mp.n_gpu_layers == llama_model_default_params().n_gpu_layers && n_tbo(mp) <= n_tbo_in;
            };
            if (on && !model_only && !has_draft && !spec_mtp && !params.fit_params_target_set &&
                    fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && !all_on_device(mparams) &&
                    mparams_in.n_gpu_layers == llama_model_default_params().n_gpu_layers && [&] {
                        std::vector<uint8_t> exps;
                        return common_model_n_layer(params.model.path, exps) > 0 && !exps.empty() &&
                            std::all_of(exps.begin(), exps.end(), [](uint8_t x) { return x == 0; });
                    }()) {
                std::vector<ggml_backend_dev_t> devs;
                if (!params.devices.empty()) {
                    devs = params.devices;
                } else {
                    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
                        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                            devs.push_back(dev);
                        }
                    }
                }
                const std::vector<int> headless = common_fit_devices_headless(devs);
                const char * e_hl = getenv("LLAMA_FIT_PLAIN_HEADLESS");
                std::string sig;
                for (int h : headless) {
                    sig += sig.empty() ? "" : ",";
                    sig += h == 1 ? "headless" : h == 0 ? "display" : "unknown";
                }
                if (!common_fit_plain_spill_gate(headless, e_hl)) {
                    COM_INF("%s: --fit: k19-fitplain: %d layers on the device beside the default margin; device display signal [%s]: "
                        "the margin stays (LLAMA_FIT_PLAIN_HEADLESS=1 steps it down)\n", __func__, mparams.n_gpu_layers, sig.c_str());
                } else {
                    const char * e_m = getenv("LLAMA_FIT_PLAIN_SPILL_MARGINS_MIB");
                    const std::vector<std::vector<size_t>> trials = common_fit_margin_trials(margins_in,
                        common_fit_margin_steps_mib(e_m != nullptr ? e_m : "512,448,256,192"));
                    for (const std::vector<size_t> & mgs : trials) {
                        llama_model_params   mps  = mparams_in;
                        llama_context_params cps  = cparams_in;
                        std::vector<float>   tss  = ts_in;
                        std::vector<llama_model_tensor_buft_override> tbos = tbo_in;
                        std::vector<size_t>  mgt  = mgs;
                        mps.tensor_split = tss.data();
                        if (mparams_in.tensor_buft_overrides != nullptr && !tbo_in.empty()) {
                            mps.tensor_buft_overrides = tbos.data();
                        }
                        common_fit_set_moe_margin_auto(!params.fit_params_target_set);
                        const common_params_fit_status sts = common_fit_params(params.model.path.c_str(), &mps, &cps, tss.data(),
                            tbos.empty() ? nullptr : tbos.data(), mgt.data(), params.fit_params_min_ctx, nullptr, fit_log_level,
                            params.fit_kv_resident);
                        if (sts != COMMON_PARAMS_FIT_STATUS_SUCCESS || !all_on_device(mps)) {
                            continue;
                        }
                        COM_WRN("%s: --fit: k19-fitplain: %d layers on the device beside the default margin (-1: every layer, tensors moved), every layer beside a %zu MiB "
                            "margin on a device with no display [%s]: the smaller margin is used (LLAMA_FIT_PLAIN_SPILL=0 keeps the default)\n",
                            __func__, mparams.n_gpu_layers, mgt[0]/(1024*1024), sig.c_str());
                        params.tensor_buft_overrides = tbos;
                        std::copy(tss.begin(), tss.end(), params.tensor_split);
                        params.fit_params_target = mgt;
                        const bool tbo_ours = mps.tensor_buft_overrides != nullptr && mps.tensor_buft_overrides == tbos.data();
                        mparams = mps;
                        cparams = cps;
                        mparams.tensor_split = params.tensor_split;
                        if (tbo_ours) {
                            mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                        }
                        fit_status = sts;
                        break;
                    }
                    if (!all_on_device(mparams)) {
                        COM_INF("%s: --fit: k19-fitplain: no margin of the step list keeps every layer on the device, the default "
                            "placement stays\n", __func__);
                    }
                }
            }
        }
        moe_margin_gain = fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS ? (moe_gain_mtp_trial ? moe_gain_first : common_fit_moe_margin_gain()) : 0;

        // --moe-cache-mib auto that the fit did not size (a failed or skipped fit, no host experts on a device) is no cache, as with
        // the release: the unbounded request must never reach the context
        if (moe_cache_auto && (fit_status != COMMON_PARAMS_FIT_STATUS_SUCCESS || cparams.moe_cache_size >= COMMON_MOE_CACHE_AUTO_BYTES)) {
            COM_WRN("%s: --moe-cache-mib auto: the fit did not size the MoE cache, not using one\n", __func__);
            cparams.moe_cache_size = 0;
        } else if (moe_cache_auto) {
            COM_INF("%s: --moe-cache-mib auto: MoE cache %zu MiB\n", __func__, cparams.moe_cache_size/(1024*1024));
        }
        // k19-mtpprice: a separate MTP drafter reads the prompt in ubatches of at most common_draft_ubatch_max() only beside a MoE cache
        // (common_base_params_to_speculative), and the fit prices it that way while it plans one. Where the fit then ends without a cache
        // but with experts in host memory (layers moved to the CPU), the drafter's context would be created at the target's n_ubatch, a
        // compute buffer the fit never priced: G4 12 GB, Qwen3.8-Flash-Next ctx 131072 -ub 2048 + its requantized MTP GGUF, priced 2021
        // MiB at n_ubatch 512, the context's 1443 MiB compute buffer at 2048 failed ("failed to create MTP context", drafting off). The
        // drafter keeps the n_ubatch it was priced at. LLAMA_FIT_DRAFT_UB_PIN=0 off
        {
            const char * e_pin   = getenv("LLAMA_FIT_DRAFT_UB_PIN");
            const bool   mtp_dft = params.speculative.has_dft() && std::find(params.speculative.types.begin(),
                params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
            bool host_exps = false;
            for (size_t i = 0; mparams.tensor_buft_overrides != nullptr && mparams.tensor_buft_overrides[i].pattern != nullptr && !host_exps; ++i) {
                host_exps = mparams.tensor_buft_overrides[i].buft != nullptr && ggml_backend_buft_is_host(mparams.tensor_buft_overrides[i].buft);
            }
            const uint32_t cap  = common_draft_ubatch_max();
            const uint32_t ub_d = common_draft_n_ubatch(params, cparams.n_ubatch);
            if ((e_pin == nullptr || atoi(e_pin) != 0) && moe_cache_auto && cparams.moe_cache_size == 0 && host_exps && mtp_dft &&
                    !params.speculative.draft.fit_off && !params.speculative.draft.fit_on_cpu && params.speculative.draft.n_ubatch <= 0 &&
                    cap > 0 && ub_d > cap) {
                params.speculative.draft.n_ubatch = (int32_t) cap;
                COM_INF("%s: --fit: the MTP drafter keeps n_ubatch %u, as priced beside the planned MoE cache (no cache now, experts in host "
                    "memory; LLAMA_FIT_DRAFT_UB_PIN=0 off)\n", __func__, cap);
            }
        }
        // k17-small (lazy KV): where the fit keeps a drafter (a draft model or the MTP heads) the KV cache is backed whole at load and the
        // MoE cache keeps its floor (common_fit does not give it the unbacked KV beside a drafter): G4 q4x 4 GB 16k + MTP ran out of memory
        // at the first request 3/3 with the bigger cache, q3m 16k + MTP was 5-17% slower. Without a drafter (none asked, or dropped by the
        // fit) lazy KV stays on. LLAMA_KV_LAZY_FIT_DRAFT=1 keeps it on beside a drafter
        {
            const char * e_kv = getenv("LLAMA_KV_LAZY");
            const char * e_fd = getenv("LLAMA_KV_LAZY_FIT_DRAFT");
            // k19-vramlend: unset = auto (on where the fit gave the MoE cache the unbacked KV), LLAMA_KV_LAZY=0 off (the kill switch)
            const bool lazy_auto = e_kv != nullptr && strcmp(e_kv, "auto") == 0;   // (opt-in variant: LLAMA_KV_LAZY=auto)
            const bool lazy_on   = lazy_auto || (e_kv != nullptr && atoi(e_kv) != 0);
            const bool draft_on = !params.speculative.draft.fit_off && !params.speculative.draft.fit_on_cpu && (params.speculative.has_dft() ||
                std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end());
            const bool suspend  = lazy_on && draft_on && !(e_fd != nullptr && atoi(e_fd) != 0);
            common_set_env("LLAMA_KV_LAZY_SUSPEND", suspend ? "1" : "0");
            if (suspend) {
                // the whole KV is backed now: a cache the fit raised by the unbacked KV (a fit that ran without the drafter) gives it back
                const size_t grant = common_fit_kv_lazy_grant(cparams.moe_cache_size);
                if (grant > 0 && cparams.moe_cache_size > grant) {
                    cparams.moe_cache_size -= grant;
                }
                COM_INF("%s: lazy KV: a drafter is kept, the KV cache is backed whole at load, MoE cache %zu MiB (%zu MiB of unbacked KV "
                    "given back; LLAMA_KV_LAZY_FIT_DRAFT=1 keeps lazy KV)\n", __func__, cparams.moe_cache_size/(1024*1024), grant/(1024*1024));
            }
            // k19-vramlend: the automatic mode backs the KV lazily only where this fit gave the MoE cache the KV it leaves unbacked (the
            // cache gives it back as the context grows); elsewhere (no MoE cache, several devices, a fit that granted nothing) the KV cache
            // is backed whole at load as with LLAMA_KV_LAZY=0
            const size_t grant_now = suspend || !lazy_on ? 0 : common_fit_kv_lazy_grant(cparams.moe_cache_size);
            common_set_env("LLAMA_KV_LAZY_GRANTED", grant_now > 0 ? "1" : "0");
            if (grant_now > 0) {
                COM_INF("%s: lazy KV %s: the KV cache backs the cells in use, the MoE cache (%zu MiB) holds the %zu MiB of KV cells not backed "
                    "yet and gives them back as the context grows (LLAMA_KV_LAZY=0 to disable)\n", __func__, lazy_auto ? "on (auto)" : "on",
                    cparams.moe_cache_size/(1024*1024), grant_now/(1024*1024));
            }
        }

        // the fit may have reduced the MoE cache (it gets the memory left after the layers), the load mode below follows the fitted size
        params.moe_cache_size = cparams.moe_cache_size;

        // k19-uma: unified memory (common_moe_uma: every offload device an integrated GPU), --moe-cache-mib auto: a fitted cache of at
        // most the hybrid gate's share (LLAMA_MOE_CACHE_HYBRID_MAX_FRAC, 0.10) of the host experts is not used. There the device and the
        // host read the same DRAM: the host experts that ggml-cpu computes in place (upstream's --cpu-moe path) are not slower to reach
        // than the cache's copies, and the cache's fit costs the rest of the placement: gfx1151 Qwen3.6-35B-A3B UD-Q4_K_XL + MTP at an
        // emulated 4 GiB device (3.1% cached), the fit dropped the MTP drafter to keep the cache and the layers on the device, decode 43.1
        // tok/s vs 52.7 with --moe-cache-mib 0 (MTP kept, 35 of 42 layers on the device) and upstream's 50.7. The launch is fitted again
        // from the given parameters as with --moe-cache-mib 0. Discrete GPUs are unchanged; LLAMA_MOE_UMA_MIN_FRAC=0 keeps the cache
        if (moe_cache_auto && !uma_no_cache && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && cparams.moe_cache_size > 0 &&
                !model_only && common_moe_uma(params)) {
            const char * e_mf = getenv("LLAMA_MOE_UMA_MIN_FRAC");
            const double min_frac = e_mf != nullptr ? std::max(0.0, atof(e_mf)) : common_moe_hybrid_max_frac();
            const double host = common_moe_host_expert_bytes(params.model.path, mparams.tensor_buft_overrides, mparams.n_gpu_layers);
            if (min_frac > 0.0 && host > 0.0 && (double) cparams.moe_cache_size <= min_frac*host) {
                COM_INF("%s: --moe-cache-mib auto on unified memory: a MoE cache of %zu MiB holds %.1f%% of the host experts (at most %.1f%%): "
                    "no MoE cache, the host computes its experts (LLAMA_MOE_UMA_MIN_FRAC=0 keeps the cache)\n", __func__,
                    cparams.moe_cache_size/(1024*1024), 100.0*(double) cparams.moe_cache_size/host, 100.0*min_frac);
                params = params_given;
                params.moe_cache_auto = false;
                params.moe_cache_size = 0;
                uma_no_cache = true;
                goto uma_restart;
            }
        }

        // k19-fitfaoff (LLAMA_FIT_FAOFF_UB=0 off): without flash attention and without -ub the fit chose n_ubatch for the F16 copies of
        // the KQ matrix it reserves (common_fit_faoff_ubatch_plan); it stays as if given with -ub, so the n_ubatch growth rules below
        // (resident ubatch, the MoE cache's auto ubatch, up to 2048 and more), which do not price those copies, keep it. Kaggle 2x T4
        // dsf8 -fa off ctx 16384: -ub 2048 with its 4096 MiB reserve 4.5 / 2.5 tok/s chat / 8k, -ub 512 52.7 / 17.1, both without an abort
        if (fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && !params.n_ubatch_set && common_fit_faoff_ubatch() > 0) {
            const uint32_t ub = std::min<uint32_t>(cparams.n_ubatch, common_fit_faoff_ubatch());
            COM_INF("%s: --fit: flash attention off: n_ubatch %u kept (the KQ copies are reserved at it; LLAMA_FIT_FAOFF_UB=0 lets it grow)\n",
                __func__, ub);
            cparams.n_ubatch      = ub;
            params.n_ubatch       = (int32_t) ub;
            params.n_ubatch_set   = true;
        }

        // k9-prompt-pass: every expert on the device (the MoE cache was asked for and has nothing to hold): the prompt runs at
        // LLAMA_MOE_RESIDENT_UB (2048) instead of the default n_ubatch when the fit still places every expert on the device at that
        // ubatch, the KV cache does not grow with it (no sliding-window layers: their cache holds the window plus one ubatch, which every
        // later decode step reads) and the draft context keeps its ubatch. The MoE kernels of a 512-token ubatch run far below their
        // peak (G4, Qwen3.6-35B-A3B UD-Q4_K_XL 32 GB, -ub 2048: 2k / 8k prompt +49 / +52%). LLAMA_MOE_RESIDENT_UB=0 off, =N that ubatch;
        // LLAMA_MOE_RESIDENT_UB_SWA=1 also with sliding-window layers
        if (!model_only && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && (moe_cache_auto || params.moe_cache_size > 0) &&
                !params.n_ubatch_set && !params.embedding && !cparams.embeddings &&
                !common_experts_on_offloaded_host(params, mparams.tensor_buft_overrides, mparams.n_gpu_layers)) {
            common_moe_resident_ubatch(params, mparams, cparams, (has_draft || spec_mtp) && !draft_off ? &extra : nullptr, fit_log_level);
        }

        // decide on the params the model is loaded with, whatever the fit status: a failed or skipped fit leaves the user placement, which then decides like the path without --fit
        // the fitted overrides live in params.tensor_buft_overrides, which mparams points to
        mparams.load_mode = common_moe_cache_load_mode(params, mparams.load_mode, mparams.n_gpu_layers);

        // k16-fit: the memory the fit leaves free (its target) is also the headroom of the context's optional load-time allocations (the
        // MoE cache seed graph, an unlent staging arena): they only take memory beyond it (llama_context::sched_reserve_seed,
        // llama_moe_cache lend_decide). LLAMA_FIT_TARGET_HEADROOM=0: their fixed 512 MiB as before
        if (fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && (getenv("LLAMA_FIT_TARGET_HEADROOM") == nullptr || atoi(getenv("LLAMA_FIT_TARGET_HEADROOM")) != 0)) {
            size_t target = 0;
            for (size_t t : params.fit_params_target) {
                target = std::max(target, t);
            }
            common_set_env("LLAMA_FIT_TARGET_MIB", std::to_string(target/(1024*1024)));
        }

        // k19-mgpu (LLAMA_SPLIT_AUTO_TENSOR=0 off): on two or more GPUs, without -sm / -ts / -ngl / -ot given, where the fit kept the whole
        // model on the devices (no host experts, no MoE cache, the draft not on the CPU) and the architecture splits by tensor, the model
        // loads with -sm tensor when a measure of that placement (every layer, an even split) leaves each GPU its fit margin. Decode on
        // -sm layer runs the GPUs one after the other; -sm tensor runs both on half of every layer. Kaggle 2x T4, Qwen3.8-27B UD-Q3_K_XL
        // ctx 16384: no draft chat 13.2 -> 18.6 tok/s, MTP n2 chat 23.8 -> 31.4, 8k 22.0 -> 30.2; Qwen3.6-35B-A3B UD-Q4_K_XL + MTP
        // chat 101.7 -> 108.7, 8k 93.0 -> 105.8 (plans/k19_mgpu.md). A draft model follows the target's split mode (common_base_params_to_speculative)
        // MoE models too by default (k19-moetp; LLAMA_SPLIT_AUTO_TENSOR=3 = dense models only, the k19-mgpu default). The Qwen3.6-35B-A3B
        // UD-Q4_K_XL "out of band" verdict (srv_kld long48 first-token KLD mean 0.00184 vs the -fa off arm's 0.00145) was the gate, not the
        // split: the AllReduce adds the two F32 halves exactly (P2P) at quant-block-aligned splits, and a rounding-only change on the layer
        // path (GGML_CUDA_DISABLE_FUSION=1) lands at 0.00101 on the same prompts; the three arms' means overlap (bootstrap 95% CIs), tensor
        // vs -fa off paired p = 0.43, 24 / 48 prompts worse, median / geometric mean lower (0.00048 / 0.00047 vs 0.00065 / 0.00049): one
        // MoE routing flip anywhere in a 2100-8200 token prompt moves the first token by ~1e-3 whatever the size of the rounding change.
        // Kaggle 2x T4 q4x: MTP 101.9 -> 106.9 chat, 94.3 -> 106.0 8k; no draft 66.1 -> 81.4, 63.5 -> 79.9 (plans/k19_moetp.md)
        {
            static const char * e_auto = getenv("LLAMA_SPLIT_AUTO_TENSOR");
            const int  mode = e_auto == nullptr ? 1 : atoi(e_auto);
            std::vector<uint8_t> exps_t;
            common_model_n_layer(params.model.path, exps_t);
            const bool moe = std::any_of(exps_t.begin(), exps_t.end(), [](uint8_t e) { return e != 0; });
            const bool on  = mode > 0 && !(mode == 3 && moe);
            std::vector<ggml_backend_dev_t> gpus;
            if (!params.devices.empty()) {
                for (ggml_backend_dev_t d : params.devices) {
                    if (d != nullptr && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        gpus.push_back(d);
                    }
                }
            } else {
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        gpus.push_back(d);
                    }
                }
            }
            if (mode == 3 && moe && gpus.size() >= 2 && !params.split_mode_set && !model_only) {
                COM_INF("%s: --fit: MoE model: -sm layer (LLAMA_SPLIT_AUTO_TENSOR=3: -sm tensor for dense models only)\n", __func__);
            }
            const bool ts_user = std::any_of(ts_in.begin(), ts_in.end(), [](float x) { return x != 0.0f; });
            const bool ngl_user = mparams_in.n_gpu_layers != llama_model_default_params().n_gpu_layers;
            const bool tbo_none = std::none_of(params.tensor_buft_overrides.begin(), params.tensor_buft_overrides.end(),
                [](const llama_model_tensor_buft_override & o) { return o.pattern != nullptr; });
            // k19-2xt4 (LLAMA_SPLIT_AUTO_TENSOR_HOST=0 off): a MoE model whose -sm layer fit put routed experts on the host (its
            // -ot overrides and a MoE cache, none given by the user) is measured for -sm tensor too: where the tensor split holds the
            // whole model, the fit's host experts and cache are dropped and it loads split by tensor. Layer split places whole layers
            // per GPU, so a model of few large layers (DeepSeek-V4.1 8-layer fake: 2.64 GiB of experts per layer, 24.1 GiB) leaves
            // slack on both cards that the even tensor split uses. Kaggle 2x T4 dsf8 (plans/k19_2xt4board.md): -sm layer + 2 x 8.9 GB
            // MoE cache 65.6 / 63.1 tok/s (chat / 8k), 8k TTFT 11.5 s; -sm tensor 75.1 / 69.3, 5.05 s
            // srv_kld long48 on dsf8 (vs the layer + cache default): first-token KLD mean 0.0091 / p99 0.134 / top1 0.917, per position
            // 0.0027 / 0.133 / 0.9725, inside the -fa off band of the same launch (k19-fitfaoff, -fa off vs on: 0.0095 / 0.143 / 0.917 and
            // 0.0025 / 0.133 / 0.9736), so on by default; LLAMA_SPLIT_AUTO_TENSOR_HOST=0 = the layer fit's host experts and cache as before
            static const bool host_on = [] {
                const char * e = getenv("LLAMA_SPLIT_AUTO_TENSOR_HOST");
                return e == nullptr || atoi(e) != 0;
            }();
            const bool tbo_user_none = std::none_of(tbo_in.begin(), tbo_in.end(),
                [](const llama_model_tensor_buft_override & o) { return o.pattern != nullptr; });
            const bool fit_host = moe && host_on && tbo_user_none && (!tbo_none || cparams.moe_cache_size > 0) && cparams_in.moe_cache_size > 0 &&
                params.moe_cache_size > 0;
            if (on && !model_only && fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && gpus.size() >= 2 && !params.split_mode_set &&
                    mparams.split_mode == LLAMA_SPLIT_MODE_LAYER && !ts_user && !ngl_user && ((tbo_none && cparams.moe_cache_size == 0) || fit_host) &&
                    !draft_cpu) {
                llama_model_params   mp = mparams;
                llama_context_params cp = cparams;
                cp.moe_cache_size       = 0;   // k19-2xt4: a fit_host launch is measured without the fit's cache (no-op otherwise)
                mp.split_mode           = LLAMA_SPLIT_MODE_TENSOR;
                mp.n_gpu_layers         = llama_model_default_params().n_gpu_layers;
                mp.tensor_split         = nullptr;
                mp.tensor_buft_overrides = nullptr;
                int64_t need = -1;
                try {
                    std::vector<ggml_backend_dev_t> devs_t;
                    uint32_t ngl_t = 0, nct_t = 0, nex_t = 0;
                    const common_device_memory_data_vec dm = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs_t,
                        ngl_t, nct_t, nex_t, fit_log_level);
                    need = 0;
                    for (size_t i = 0; i + 1 < dm.size(); i++) { // the last entry is the host
                        // the no_alloc measure sizes each split weight and K/V tensor whole (the Meta buffer type's alloc size of the
                        // full tensor), so per GPU it is that total over the GPUs (+3% for the slices' rounding and the copied small
                        // tensors); the compute buffer is measured per GPU. q38s: 12240 MiB measured vs 5837 MiB loaded per T4
                        const int64_t split = (int64_t) (dm[i].model + dm[i].context);
                        need = std::max<int64_t>(need, split/(int64_t) gpus.size()*103/100 + (int64_t) dm[i].compute);
                    }
                } catch (const std::exception & e) {
                    COM_INF("%s: --fit: -sm tensor not measured (%s): -sm layer\n", __func__, e.what());
                    need = -1;
                }
                if (need > 0) {
                    // the draft model (-md) loads beside it with the same split: its weights per GPU plus its context
                    int64_t dft = 0;
                    if (has_draft && !draft_off) {
                        std::error_code ec;
                        const auto sz = std::filesystem::file_size(params.speculative.draft.mparams.path, ec);
                        dft = (ec ? 0 : (int64_t) sz/(int64_t) gpus.size()) + (int64_t) 512*1024*1024;
                    } else if (spec_mtp && !draft_off) {
                        dft = (int64_t) 256*1024*1024; // the MTP context of the built-in heads (their weights are in the measure)
                    }
                    bool fits = true;
                    int64_t free_min = INT64_MAX;
                    for (size_t i = 0; i < gpus.size(); i++) {
                        size_t f = 0, t = 0;
                        ggml_backend_dev_memory(gpus[i], &f, &t);
                        const int64_t margin = (int64_t) (i < params.fit_params_target.size() ? params.fit_params_target[i] : params.fit_params_target[0]);
                        free_min = std::min<int64_t>(free_min, (int64_t) f);
                        if ((int64_t) f - need - dft < margin) {
                            fits = false;
                        }
                    }
                    COM_INF("%s: --fit: whole model on %zu GPUs: %s (-sm tensor needs %" PRId64 " + %" PRId64 " MiB per GPU, least free %" PRId64
                        " MiB, margin %zu MiB; LLAMA_SPLIT_AUTO_TENSOR=0 keeps -sm layer)\n", __func__, gpus.size(), fits ? "-sm tensor" : "-sm layer",
                        need/(1024*1024), dft/(1024*1024), free_min/(1024*1024), params.fit_params_target[0]/(1024*1024));
                    if (fits && fit_host) {
                        // k19-2xt4: the tensor split replaces the fit's host experts and MoE cache
                        const size_t n_fit_tbo = (size_t) std::count_if(params.tensor_buft_overrides.begin(), params.tensor_buft_overrides.end(),
                            [](const llama_model_tensor_buft_override & o) { return o.pattern != nullptr; });
                        COM_INF("%s: --fit: -sm tensor holds the whole model: the layer fit's host experts (%zu overrides) and %zu MiB MoE "
                            "cache dropped (LLAMA_SPLIT_AUTO_TENSOR_HOST=0 keeps them)\n", __func__, n_fit_tbo,
                            cparams.moe_cache_size/(1024*1024));
                        params.tensor_buft_overrides = tbo_in;
                        mparams.tensor_buft_overrides = mparams_in.tensor_buft_overrides != nullptr && !params.tensor_buft_overrides.empty() ?
                            params.tensor_buft_overrides.data() : mparams_in.tensor_buft_overrides;
                        cparams.moe_cache_size = 0;
                        params.moe_cache_size  = 0;
                    }
                    if (fits) {
                        mparams.split_mode     = LLAMA_SPLIT_MODE_TENSOR;
                        mparams.n_gpu_layers   = mp.n_gpu_layers;
                        params.split_mode      = LLAMA_SPLIT_MODE_TENSOR;
                        params.split_mode_auto = true;
                        // F32 on the AllReduce wire on every host: off the P2P path (no peer access, e.g. some Kaggle 2x T4 VMs) the
                        // AllReduce defaults to a BF16 wire, a second text class besides the P2P path's exact F32 sum (q38s chat md5
                        // 75d741 vs f813a2); with F32 both paths add the same two halves, so the text does not depend on the host.
                        // An explicit GGML_CUDA_AR_BF16_THRESHOLD wins
                        if (getenv("GGML_CUDA_AR_BF16_THRESHOLD") == nullptr) {
                            common_set_env("GGML_CUDA_AR_BF16_THRESHOLD", "0");
                        }
                        params.n_gpu_layers    = mp.n_gpu_layers;
                        std::fill(params.tensor_split, params.tensor_split + 128, 0.0f);
                        mparams.tensor_split   = params.tensor_split;
                    }
                }
            }
        }
    }

    if (!model_only) {
        common_moe_cache_auto_ubatch(params, mparams, cparams);
    }

    // k11 (LLAMA_MOE_FIT_MARGIN): once n_ubatch, lending and the load mode follow the cache the fit sized with the full margin, the
    // cache takes the part of the default margin a spilled context does not need (see common_fit_moe_margin_gain)
    if (!model_only && moe_margin_gain > 0 && cparams.moe_cache_size > 0 && cparams.moe_cache_size < COMMON_MOE_CACHE_AUTO_BYTES) {
        cparams.moe_cache_size += (size_t) moe_margin_gain;
        params.moe_cache_size   = cparams.moe_cache_size;
        COM_INF("%s: MoE cache %zu MiB: +%" PRId64 " MiB of the fit margin beside it (LLAMA_MOE_FIT_MARGIN_MIB, LLAMA_MOE_FIT_MARGIN=0 off)\n",
            __func__, cparams.moe_cache_size/(1024*1024), moe_margin_gain/(1024*1024));
    } else {
        moe_margin_gain = 0;
    }

    // k19-df2cost: where the DFlash2 cached window band engages (a DFlash drafter, the cache holding LLAMA_DFLASH_CACHED_FRAC_MIN ..
    // LLAMA_DFLASH_CACHED_FRAC_MAX of the host experts, see common/speculative.cpp) the target keeps the verify graphs of two widths
    // (3 rows and the whole draft) and their CUDA graphs, which the fit did not price: G4 Qwen3.8-Flash-Next UD-IQ4_XS 32 GB, srv_kld long48
    // aborted with CUDA out of memory after a 2096-token prompt. The cache leaves LLAMA_DFLASH_CACHED_RESERVE_MIB (K19_DF2COST_RESERVE_MIB)
    // of device memory for them (0: none)
    if (!model_only && cparams.moe_cache_size > 0 && cparams.moe_cache_size < COMMON_MOE_CACHE_AUTO_BYTES &&
            std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != params.speculative.types.end()) {
        const char * ew  = getenv("LLAMA_DFLASH_WINDOW");
        const char * ecw = getenv("LLAMA_DFLASH_CACHED_WINDOW");
        const double lo  = getenv("LLAMA_DFLASH_CACHED_FRAC_MIN") ? atof(getenv("LLAMA_DFLASH_CACHED_FRAC_MIN")) : 0.20;
        const double hi  = getenv("LLAMA_DFLASH_CACHED_FRAC_MAX") ? atof(getenv("LLAMA_DFLASH_CACHED_FRAC_MAX")) : K19_DF2COST_FRAC_MAX;
        const int64_t rs = (getenv("LLAMA_DFLASH_CACHED_RESERVE_MIB") ? std::max<int64_t>(0, atoll(getenv("LLAMA_DFLASH_CACHED_RESERVE_MIB"))) :
            (int64_t) K19_DF2COST_RESERVE_MIB)*1024*1024;
        common_moe_slot_bytes sbt;
        if ((ew == nullptr || atoi(ew) != 0) && (ecw == nullptr || atoi(ecw) != 0) && rs > 0 && hi > lo &&
                common_moe_slot_bytes_of(params.model.path, sbt) && sbt.total > 0) {
            const double share = (double) cparams.moe_cache_size/(double) sbt.total;
            if (share >= lo && share < hi && (int64_t) cparams.moe_cache_size > 2*rs) {
                cparams.moe_cache_size -= (size_t) rs;
                params.moe_cache_size   = cparams.moe_cache_size;
                COM_INF("%s: MoE cache %zu MiB: -%" PRId64 " MiB for the DFlash2 cached window's second verify width (%.1f%% of the host "
                    "experts cached; LLAMA_DFLASH_CACHED_RESERVE_MIB)\n", __func__, cparams.moe_cache_size/(1024*1024), rs/(1024*1024), 100.0*share);
            }
        }
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // these decision models return a score for each token via the embeddings output
    // TODO: maybe improve this in the future
    const auto decision_type = common_get_decision_type(model);
    if (decision_type == COMMON_DECISION_TYPE_LAYA || decision_type == COMMON_DECISION_TYPE_KEV || decision_type == COMMON_DECISION_TYPE_CLEF ||
        decision_type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || decision_type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
        params.embedding    = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;

        cparams.embeddings            = true;
        cparams.pooling_type          = LLAMA_POOLING_TYPE_NONE;
        cparams.n_outputs_max         = cparams.n_batch;
        cparams.n_outputs_max_per_seq = 1;

        LOG_INF("%s", "decision model reads the embeddings output, enabling embedding mode\n");
    }

    // embeddings need the whole batch in one ubatch, so n_batch must not be larger than n_ubatch
    // (server.cpp does this check for --embedding, but before the model is loaded)
    if (cparams.embeddings && cparams.n_batch > cparams.n_ubatch) {
        LOG_WRN("embeddings enabled: setting n_batch = n_ubatch = %u\n", cparams.n_ubatch);
        cparams.n_batch = cparams.n_ubatch;
        params.n_batch  = params.n_ubatch;
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            COM_ERR("failed to load lora adapter '%s'\n", la.path.c_str());
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            COM_TRC("added %s logit bias = %f\n", common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    std::vector<ggml_backend_dev_t> reserve_devs;
    std::vector<size_t>             reserve_bytes;
    common_draft_mem_reserve(params, mparams, cparams, reserve_devs, reserve_bytes);
    if (!reserve_bytes.empty()) {
        cparams.mem_reserve_devs = reserve_devs.data();
        cparams.mem_reserve      = reserve_bytes.data();
        cparams.n_mem_reserve    = reserve_bytes.size();
    }

    // k19-rollback: the n-gram suffix drafter alone asked for the recurrent rollback rows (need_n_rs_seq) of a model whose rollback
    // keeps a large state per drafted token (GDN / Mamba hybrids), and the layer weights spill to host memory: the rows (7 x 8.4 MiB conv
    // rows + replay stores on G4 Qwen3.8-Flash-Next, 59 MiB at 8 GB, where the small-device placement has ~50 MiB left) go, and a
    // window that is not fully accepted is rolled back from the host copy of the state (llama_rs_snap_save, one sequence). Nothing stays
    // on the device for the drafter; where nothing repeats the launch runs as one without it. The fit priced the rows, so their memory
    // stays free. LLAMA_SPEC_SUFFIX_RS=1 keeps the rows (=0 never takes them, as before)
    if (cparams.n_rs_seq > 0 && getenv("LLAMA_SPEC_SUFFIX_RS") == nullptr && params.n_parallel == 1) {
        const auto & ty = params.speculative.types;
        const bool model_rows = std::any_of(ty.begin(), ty.end(), [](auto t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
                   t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        }) && !params.speculative.draft.fit_off;
        char arch[64] = {};
        const bool big_state = llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) > 0 && llama_arch_rollback_state(arch) == 2;
        if (!model_rows && big_state && params.speculative.suffix_n_max() > 0 && llama_model_weights_spill(model)) {
            COM_INF("%s: n-gram drafts beside host weights: no recurrent rollback rows on the device (%u asked), a window that is not "
                "fully accepted goes back to a host copy of the state (LLAMA_SPEC_SUFFIX_RS=1 keeps the rows)\n", __func__, cparams.n_rs_seq);
            cparams.n_rs_seq = 0;
        }
    }

    // -t given: the threads that compute missing experts on the host are the user's too (k11 LLAMA_MOE_HOST_THREADS)
    if (params.n_threads_set && getenv("LLAMA_MOE_HOST_THREADS") == nullptr) {
        common_set_env("LLAMA_MOE_HOST_THREADS", "0");
    }
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL && moe_margin_gain > 0 && cparams.moe_cache_size > (size_t) moe_margin_gain) {
        // the MoE cache got memory of the default margin (common_fit_moe_margin_gain) and the context does not fit: the cache of the
        // full margin
        cparams.moe_cache_size -= (size_t) moe_margin_gain;
        params.moe_cache_size   = cparams.moe_cache_size;
        COM_WRN("%s: the context does not fit beside the MoE cache with the reduced margin: retrying with a MoE cache of %zu MiB "
            "(LLAMA_MOE_FIT_MARGIN=0 keeps the full margin)\n", __func__, cparams.moe_cache_size/(1024*1024));
        lctx = llama_init_from_model(model, cparams);
    }
    cparams.mem_reserve_devs = nullptr;
    cparams.mem_reserve      = nullptr;
    cparams.n_mem_reserve    = 0;
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);

    set_process_priority(params.cpuparams.priority);

    pimpl->threadpools.init(lctx, params);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        // the history too: the greedy fast path rebuilds its penalty window from it (common_sampler_sample), as the chain from its state
        common_sampler_reset(pimpl->samplers[i].get());
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

// only for warmup and probe decodes, fill zeros as dummy input
static void common_batch_set_zero_state(common_batch & batch, const llama_model * model, std::vector<float> & zeros) {
    zeros.assign(llama_model_n_embd_out(model), 0.0f);
    for (int32_t i = 0; i < batch.size(); ++i) {
        batch.set_embd_state(i, { zeros.data(), 1, zeros.size() });
    }
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    // k19-g4pref: the MoE cache's default cross-layer prefetch stays off beside an MTP drafter (G4 Flash-Next Studio MTP n2 at depth 4,
    // 48 GB: 8k tok/s geomean 0.90 over 6 prompts with it on); an explicit LLAMA_MOE_PREFETCH or LLAMA_MOE_PREFETCH_AUTO is kept
    if (getenv("LLAMA_MOE_PREFETCH") == nullptr && getenv("LLAMA_MOE_PREFETCH_AUTO") == nullptr &&
            std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) !=
            params.speculative.types.end()) {
        common_set_env("LLAMA_MOE_PREFETCH_AUTO", "0");
    }
    common_init_result_ptr res(new common_init_result(params, model_only));

    llama_model * model = res->model();
    if (model == NULL) {
        COM_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return res;
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
            ok = false;
        } else if (!has_eos) {
            COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            common_batch batch = common_batch_get_one(lctx, tmp);
            llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get());
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            tmp.resize(std::min(tmp.size(), (size_t) params.n_batch));
            common_batch batch = common_batch_get_one(lctx, tmp);
            std::vector<float> zeros;
            common_batch_set_zero_state(batch, model, zeros);
            llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    std::string endpoint = common_get_env("MODEL_ENDPOINT");
    if (endpoint.empty()) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if (endpoint.empty()) {
        return "https://huggingface.co/";
    }
    if (endpoint.back() != '/') {
        endpoint += '/';
    }
    return endpoint;
}

char * common_get_model_or_exit(int argc, char * argv[]) {
    if (argc > 1) {
        return argv[1];
    }

    char * path = getenv("LLAMACPP_TEST_MODELFILE");
    if (!path || strlen(path) == 0) {
        fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
        exit(EXIT_SUCCESS);
    }

    return path;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    // k13: with frozen MoE-cache small batches (LLAMA_MOE_CPU_MISS_FROZEN=1 / LLAMA_SPEC_EXACT_FROZEN=1) the probe decode below runs for
    // recurrent rollback contexts too: its two tokens fill free cache slots before the freeze, so a context with drafts (rollback
    // snapshots, no probe before) started from another cache than the same context without them, and the texts differed
    const bool rs = llama_n_rs_seq(ctx) > 0;
    const auto env_on = [](const char * n) { const char * e = getenv(n); return e != nullptr && atoi(e) != 0; };
    if (rs && !(env_on("LLAMA_MOE_CPU_MISS_FROZEN") || env_on("LLAMA_SPEC_EXACT_FROZEN"))) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
        return COMMON_CONTEXT_SEQ_RM_TYPE_RS;
    }

    common_context_seq_rm_type res = rs ? COMMON_CONTEXT_SEQ_RM_TYPE_RS : COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret;
    {
        common_batch batch = common_batch_get_one(ctx, tmp);
        std::vector<float> zeros;
        common_batch_set_zero_state(batch, llama_get_model(ctx), zeros);
        ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    }
    if (ret != 0) {
        COM_ERR("llama_process() failed: %d\n", ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    // try to remove the last tokens
    if (!rs && !llama_memory_seq_rm(mem, 0, 1, -1)) {
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

static void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

static void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_memory::init(llama_context * ctx_tgt, llama_context * ctx_dft) {
    this->ctx_tgt = ctx_tgt;
    this->ctx_dft = ctx_dft;
}

void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
    if (ctx_dft) {
        common_context_seq_rm(ctx_dft, seq_id, p0, p1);
    }
}

void common_memory::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const {
    common_context_seq_cp(ctx_tgt, seq_id_src, seq_id_dst, p0, p1);
    if (ctx_dft) {
        common_context_seq_cp(ctx_dft, seq_id_src, seq_id_dst, p0, p1);
    }
}

void common_memory::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const {
    common_context_seq_add(ctx_tgt, seq_id, p0, p1, delta);
    if (ctx_dft) {
        common_context_seq_add(ctx_dft, seq_id, p0, p1, delta);
    }
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

// size of the model files, all the splits of a split model, 0 if unknown
static uint64_t common_model_file_bytes(const std::string & path) {
    std::vector<std::string> files = { path };
    static const std::regex re_split("-([0-9]{5})-of-([0-9]{5})\\.gguf$");
    std::smatch m;
    if (std::regex_search(path, m, re_split)) {
        const int n_split = std::stoi(m[2].str());
        const std::string prefix = path.substr(0, m.position(0));
        files.clear();
        for (int i = 1; i <= n_split; ++i) {
            char suffix[64];
            snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, n_split);
            files.push_back(prefix + suffix);
        }
    }
    uint64_t res = 0;
    for (const auto & f : files) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(f, ec);
        if (ec) {
            return 0;
        }
        res += size;
    }
    return res;
}

// whether the model file lives in memory already (tmpfs / ramfs, Linux): its pages are not reclaimable page cache, so they are neither in
// the available memory nor evicted (and read from the disk again) by a load that copies from them
static bool common_model_file_in_ram(const std::string & path) {
#if defined(__linux__)
    struct statfs sfs;
    if (statfs(path.c_str(), &sfs) == 0) {
        const unsigned long t = (unsigned long) sfs.f_type;
        return t == 0x01021994UL || t == 0x858458f6UL; // TMPFS_MAGIC, RAMFS_MAGIC
    }
#else
    GGML_UNUSED(path);
#endif
    return false;
}

// system memory usable without swapping, 0 if unknown (the CPU backend reports total memory as free on Linux and macOS)
// on Linux MemAvailable capped by the cgroup limits, without counting a cgroup's reclaimable page cache as used, see llama_available_memory
static uint64_t common_available_memory() {
    return llama_available_memory();
}

// the devices llama_model_load picks when none are given (src/llama.cpp): RPC servers, GPUs without a duplicate device id, integrated GPUs only without a discrete one
static std::vector<ggml_backend_dev_t> common_default_model_devices() {
    std::vector<ggml_backend_dev_t> rpc;
    std::vector<ggml_backend_dev_t> gpus;
    std::vector<ggml_backend_dev_t> igpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU: {
                if (ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)) == std::string("RPC")) {
                    rpc.push_back(dev);
                    break;
                }
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                bool dup = false;
                for (ggml_backend_dev_t g : gpus) {
                    ggml_backend_dev_props g_props;
                    ggml_backend_dev_get_props(g, &g_props);
                    dup = dup || (props.device_id && g_props.device_id && strcmp(props.device_id, g_props.device_id) == 0);
                }
                if (!dup) {
                    gpus.push_back(dev);
                }
                break;
            }
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                if (igpus.empty() || ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(igpus.back())) {
                    igpus.push_back(dev);
                }
                break;
            default:
                break;
        }
    }
    std::vector<ggml_backend_dev_t> devs = rpc;
    devs.insert(devs.end(), gpus.begin(), gpus.end());
    if (gpus.empty()) {
        devs.insert(devs.end(), igpus.begin(), igpus.end());
    }
    return devs;
}

// expert tensors a layer of the model has, from the tensor names of the GGUF
enum common_expert_tensor : uint8_t {
    COMMON_EXPERT_DOWN    = 1, // ffn_down_exps
    COMMON_EXPERT_GATE_UP = 2, // fused ffn_gate_up_exps
    COMMON_EXPERT_GATE    = 4, // ffn_gate_exps
    COMMON_EXPERT_UP      = 8, // ffn_up_exps
};

// number of repeating layers of the model (the <arch>.block_count key the loader reads into n_layer_all), -1 if unknown, and the expert tensors of each layer (common_expert_tensor bits)
// for a split model the metadata is read from the first split and the tensor names from every split, the expert tensors are left empty (unknown) if a split cannot be read
static int64_t common_model_n_layer(const std::string & path, std::vector<uint8_t> & experts) {
    experts.clear();
    std::string fname = path;
    std::string prefix;
    int n_split = 1;
    static const std::regex re_split("-([0-9]{5})-of-([0-9]{5})\\.gguf$");
    std::smatch m;
    if (std::regex_search(path, m, re_split)) {
        prefix  = path.substr(0, m.position(0));
        n_split = std::stoi(m[2].str());
        fname   = prefix + "-00001-of-" + m[2].str() + ".gguf";
    }
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    gguf_context_ptr gguf_ctx(gguf_init_from_file(fname.c_str(), gguf_params));
    if (!gguf_ctx) {
        return -1;
    }
    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return -1;
    }
    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    const int64_t id = gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str());
    if (id < 0) {
        return -1;
    }
    int64_t n_layer = -1;
    switch (gguf_get_kv_type(gguf_ctx.get(), id)) {
        case GGUF_TYPE_UINT32: n_layer = gguf_get_val_u32(gguf_ctx.get(), id); break;
        case GGUF_TYPE_INT32:  n_layer = gguf_get_val_i32(gguf_ctx.get(), id); break;
        default: break;
    }
    // the loader rejects more than LLAMA_MAX_LAYERS (512) layers, so a larger count is malformed: treat it as unknown rather than scan it
    if (n_layer <= 0 || n_layer > 512) {
        return -1;
    }
    std::vector<uint8_t> found(n_layer, 0);
    const auto scan = [&found, n_layer](const gguf_context * ctx) {
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            if (strncmp(name, "blk.", 4) != 0) {
                continue;
            }
            char * end = nullptr;
            const long il = strtol(name + 4, &end, 10);
            if (end == name + 4 || *end != '.' || il < 0 || il >= n_layer) {
                continue;
            }
            if (strstr(end, ".ffn_down_exps.") != nullptr) {
                found[il] |= COMMON_EXPERT_DOWN;
            } else if (strstr(end, ".ffn_gate_up_exps.") != nullptr) {
                found[il] |= COMMON_EXPERT_GATE_UP;
            } else if (strstr(end, ".ffn_gate_exps.") != nullptr) {
                found[il] |= COMMON_EXPERT_GATE;
            } else if (strstr(end, ".ffn_up_exps.") != nullptr) {
                found[il] |= COMMON_EXPERT_UP;
            }
        }
    };
    scan(gguf_ctx.get());
    // only the metadata of the other splits is read
    for (int i = 1; i < n_split; ++i) {
        std::vector<char> split_path(prefix.size() + 32);
        llama_split_path(split_path.data(), split_path.size(), prefix.c_str(), i, n_split);
        gguf_context_ptr split_ctx(gguf_init_from_file(split_path.data(), gguf_params));
        if (!split_ctx) {
            return n_layer;
        }
        scan(split_ctx.get());
    }
    experts = std::move(found);
    return n_layer;
}

// whether the tensor overrides (terminated by a null pattern, nullptr for none) put every routed expert of some layer in [il_begin, il_end) served by the MoE cache in a host buffer
// the MoE cache only serves a layer whose expert tensors are all host-resident, so an override of a single expert family is not enough
// the loader applies the first override whose pattern matches the tensor name (std::regex_search), so try representative expert names
static bool common_overrides_put_experts_on_host(const llama_model_tensor_buft_override * overrides, int64_t il_begin, int64_t il_end, const std::vector<uint8_t> & experts) {
    std::vector<std::regex> patterns;
    std::vector<bool> valid;
    bool any_host = false;
    for (size_t i = 0; overrides != nullptr && overrides[i].pattern != nullptr; ++i) {
        const llama_model_tensor_buft_override & ovr = overrides[i];
        // the loader rejects an invalid pattern anyway, here it matches nothing
        std::regex re;
        bool ok = true;
        try {
            re = std::regex(ovr.pattern);
        } catch (const std::regex_error &) {
            ok = false;
        }
        patterns.push_back(std::move(re));
        valid.push_back(ok);
        any_host = any_host || (ovr.buft != nullptr && ggml_backend_buft_is_host(ovr.buft));
    }
    if (!any_host) {
        return false;
    }
    // whether the first override matching the tensor name puts it in a host buffer
    const auto is_host = [&](int il, const char * family) {
        const std::string name = "blk." + std::to_string(il) + "." + family + ".weight";
        for (size_t i = 0; i < patterns.size(); ++i) {
            if (valid[i] && std::regex_search(name, patterns[i])) {
                return overrides[i].buft != nullptr && ggml_backend_buft_is_host(overrides[i].buft);
            }
        }
        return false;
    };
    // the expert tensors the MoE cache serves (ffn_*_exps of llama-arch.cpp): a layer has either separate gate/up/down experts or fused gate_up/down experts
    // only the tensors the GGUF has for a layer are checked, so a layer without experts (e.g. the leading dense layers of DeepSeek2) never counts
    // unknown expert tensors count a layer if either complete set is on the host
    // the layer index matters for per-layer patterns such as --n-cpu-moe
    il_end = std::min<int64_t>(il_end, 512);
    for (int64_t il = std::max<int64_t>(il_begin, 0); il < il_end; ++il) {
        const uint8_t has = experts.empty() ? (COMMON_EXPERT_DOWN | COMMON_EXPERT_GATE_UP | COMMON_EXPERT_GATE | COMMON_EXPERT_UP) :
                            il < (int64_t) experts.size() ? experts[il] : 0;
        if (!(has & COMMON_EXPERT_DOWN) || !is_host((int) il, "ffn_down_exps")) {
            continue;
        }
        const bool fused    = (has & COMMON_EXPERT_GATE_UP) && is_host((int) il, "ffn_gate_up_exps");
        const bool separate = (has & COMMON_EXPERT_GATE) && (has & COMMON_EXPERT_UP) && is_host((int) il, "ffn_gate_exps") && is_host((int) il, "ffn_up_exps");
        // ungated experts (e.g. Nemotron-H) have only up/down, which the cache serves as well; only when the GGUF says so
        const bool ungated  = !experts.empty() && (has & COMMON_EXPERT_UP) && !(has & (COMMON_EXPERT_GATE | COMMON_EXPERT_GATE_UP)) &&
                              is_host((int) il, "ffn_up_exps");
        if (fused || separate || ungated) {
            return true;
        }
    }
    return false;
}

// whether the overrides put a complete host expert set on a layer the model offloads to a device: the MoE cache rejects layers assigned to the CPU
// the placement of llama-model.cpp offloads the last n_gpu_layers of the n_layer repeating layers plus the output layer (all of them when n_gpu_layers < 0), so layers [max(n_layer + 1 - n_gpu_layers, 0), n_layer) are on a device
// the device list and split mode only choose which device serves an offloaded layer, not whether it is offloaded; an empty device list keeps every layer on the CPU but then no device can do the lookup either
// if the layer count cannot be read, every layer index up to 512 counts as before
static bool common_experts_on_offloaded_host(const common_params & params, const llama_model_tensor_buft_override * overrides, int32_t n_gpu_layers) {
    if (params.moe_cache_size <= 0 || !(n_gpu_layers < 0 || n_gpu_layers > 1)) {
        return false;
    }
    // without an override to a host buffer no expert is on the host: answer before reading the GGUF, whose metadata takes ~0.1 s to parse
    // (Gemma-4 26B-A4B, a 262k-token vocabulary) on every load that keeps every expert on a device
    bool any_host = false;
    for (size_t i = 0; overrides != nullptr && overrides[i].pattern != nullptr && !any_host; ++i) {
        any_host = overrides[i].buft != nullptr && ggml_backend_buft_is_host(overrides[i].buft);
    }
    if (!any_host) {
        return false;
    }
    std::vector<uint8_t> experts;
    const int64_t n_layer = common_model_n_layer(params.model.path, experts);
    if (n_layer < 0) {
        return common_overrides_put_experts_on_host(overrides, 0, 512, experts);
    }
    const int64_t n_gpu = n_gpu_layers < 0 ? n_layer + 1 : n_gpu_layers;
    return common_overrides_put_experts_on_host(overrides, std::max<int64_t>(n_layer + 1 - n_gpu, 0), n_layer, experts);
}

// k10: whether the MoE cache will take its default hybrid gate (llama_moe_cache: LLAMA_MOE_CACHE_HYBRID unset and the slots hold at most
// LLAMA_MOE_CACHE_HYBRID_MAX_FRAC (0.10) of the host experts' bytes). Estimated from the fitted cache size and the expert bytes of the layers
// whose experts the overrides (or n_gpu_layers) leave on the host, all expert layers of the same size
static bool common_moe_hybrid_gate_on(const common_params & params, const llama_model_tensor_buft_override * overrides, int32_t n_gpu_layers) {
    if (params.moe_cache_size <= 0 || getenv("LLAMA_MOE_CACHE_HYBRID") != nullptr || getenv("LLAMA_MOE_CACHE_FETCH") != nullptr) {
        return false;
    }
    // the CPU-miss rule instead of the gate: forced (LLAMA_MOE_CPU_MISS=2) or kept at small caches (LLAMA_MOE_CPU_MISS_GATE=0)
    const char * env_cm = getenv("LLAMA_MOE_CPU_MISS");
    const char * env_cg = getenv("LLAMA_MOE_CPU_MISS_GATE");
    if ((env_cm != nullptr && atoi(env_cm) == 2) || ((env_cm == nullptr || atoi(env_cm) != 0) && env_cg != nullptr && atoi(env_cg) == 0)) {
        return false;
    }
    // k19-exactmtp: LLAMA_MTP_EXACT=1 turns the cache's host path off (no hybrid gate) unless LLAMA_MOE_CPU_MISS=1 / LLAMA_MOE_EXACT_HOST=1
    {
        const char * x = getenv("LLAMA_MTP_EXACT");
        const char * k = getenv("LLAMA_MOE_EXACT_HOST");
        if (x != nullptr && atoi(x) != 0 && (k == nullptr || atoi(k) == 0) && (env_cm == nullptr || atoi(env_cm) == 0)) {
            return false;
        }
    }
    const double host = common_moe_host_expert_bytes(params.model.path, overrides, n_gpu_layers);
    return host > 0.0 && (double) params.moe_cache_size <= common_moe_hybrid_max_frac()*host;
}

// LLAMA_MOE_CACHE_HYBRID_MAX_FRAC (0.10): the share of the host experts at or below which the MoE cache takes its hybrid gate
static double common_moe_hybrid_max_frac() {
    double max_frac = 0.10;
    if (const char * e = getenv("LLAMA_MOE_CACHE_HYBRID_MAX_FRAC")) {
        max_frac = std::max(0.0, atof(e));
    }
    return max_frac;
}

// the bytes of the experts that the overrides (or n_gpu_layers) leave on the host, all expert layers of the same size (0: none, or unknown)
static double common_moe_host_expert_bytes(const std::string & path, const llama_model_tensor_buft_override * overrides, int32_t n_gpu_layers) {
    common_moe_slot_bytes sbt;
    std::vector<uint8_t> experts;
    const int64_t n_layer = common_model_n_layer(path, experts);
    if (n_layer <= 0 || (int64_t) experts.size() < n_layer || !common_moe_slot_bytes_of(path, sbt) || sbt.total <= 0) {
        return 0.0;
    }
    const int64_t n_gpu  = n_gpu_layers < 0 ? n_layer + 1 : n_gpu_layers;
    const int64_t il_gpu = std::max<int64_t>(n_layer + 1 - n_gpu, 0); // layers below run on the CPU: their experts are host experts
    int64_t n_exp = 0, n_host = 0;
    for (int64_t il = 0; il < n_layer; ++il) {
        if (experts[il] == 0) {
            continue;
        }
        n_exp++;
        n_host += il < il_gpu || common_overrides_put_experts_on_host(overrides, il, il + 1, experts);
    }
    if (n_exp == 0 || n_host == 0) {
        return 0.0;
    }
    return (double) sbt.total*(double) n_host/(double) n_exp;
}

// whether the weight stream is forced on (--weight-stream on, LLAMA_WEIGHT_STREAM=N != 0): it uploads the host weights read in place from the mapping, so it needs mmap
// (auto turns it on only when those weights do not fit in 80% of the available RAM, which the model size check below already rules out; the RAM tier of the MoE cache reads through it)
static bool common_weight_stream_forced(const common_params & params) {
    const char * env = getenv("LLAMA_WEIGHT_STREAM");
    if (env != nullptr) {
        return atoi(env) != 0;
    }
    return params.weight_stream == 2;
}

// the load mode for the MoE cache once the expert placement is known: n_gpu_layers is the one the model is loaded with
// n_gpu_layers: -1 is auto, <= -2 is all, 0 keeps every layer on the CPU and 1 offloads only the output layer: the MoE cache then has no repeating layer to serve, keep mmap
// with host experts the device lookup needs them pinned; with every expert on a device nothing reads the mapping after the load, and without mmap the device weights
// are uploaded from pinned staging buffers (Gemma-4 26B-A4B on an L4: 6.0 s vs 7.0-8.2 s with mmap): load without mmap in both cases, as before the host expert check
// mmap stays only where a feature that is on reads the mapping: the weight stream forced on (and the RAM tier behind it) or a device without the device lookup
// D1's memory test: the model's experts and 8 GiB do not fit in the available memory
static bool common_moe_lowram_short(const common_params & params, uint64_t * expert_bytes_out, uint64_t * avail_out) {
    common_moe_slot_bytes sb;
    const uint64_t expert_bytes = common_moe_slot_bytes_of(params.model.path, sb) && sb.total > 0 ? (uint64_t) sb.total : 0;
    uint64_t avail = common_available_memory();
    if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        size_t mem_free = 0, mem_total = 0;
        ggml_backend_dev_memory(cpu, &mem_free, &mem_total);
        const uint64_t m = std::min(mem_free, mem_total);
        avail = avail > 0 ? std::min<uint64_t>(avail, m) : m;
    }
    if (expert_bytes_out != nullptr) {
        *expert_bytes_out = expert_bytes;
    }
    if (avail_out != nullptr) {
        *avail_out = avail;
    }
    return expert_bytes > 0 && avail > 0 && expert_bytes + (UINT64_C(8) << 30) > avail;
}

// D1: a MoE cache with host experts on offloaded layers, where they and 8 GiB do not fit in the available memory (see common_moe_cache_load_mode)
static bool common_moe_lowram_maps(const common_params & params, int32_t n_gpu_layers, uint64_t * expert_bytes_out, uint64_t * avail_out) {
    if (params.moe_cache_size == 0 || !(n_gpu_layers < 0 || n_gpu_layers > 1) ||
            (getenv("LLAMA_MOE_LOWRAM_MAP") != nullptr && atoi(getenv("LLAMA_MOE_LOWRAM_MAP")) == 0) ||
            !common_experts_on_offloaded_host(params, params.tensor_buft_overrides.empty() ? nullptr : params.tensor_buft_overrides.data(), n_gpu_layers)) {
        return false;
    }
    return common_moe_lowram_short(params, expert_bytes_out, avail_out);
}

static enum llama_load_mode common_moe_cache_load_mode(const common_params & params, enum llama_load_mode load_mode, int32_t n_gpu_layers) {
    // k15-fn (LLAMA_MOE_LOWRAM_MAP=0 off): an explicit --load-mode none copies every host expert into pinned memory, which is not
    // reclaimable; where the host experts and 8 GiB do not fit in the available memory that load is killed for memory (G4 fndl at
    // 48 GiB: Flash-Next UD-IQ4_XS, 57.4 GB of host experts, rc -9). Load with mmap there instead: the weight store streams the experts
    // and the MoE cache RAM tier serves them. Never where they fit, so every such host loads as before
    uint64_t expert_bytes = 0, avail = 0;
    if (load_mode == LLAMA_LOAD_MODE_NONE && common_moe_lowram_maps(params, n_gpu_layers, &expert_bytes, &avail)) {
        COM_INF("MoE cache: --load-mode none would pin %.0f MiB of host experts with %.0f MiB of memory available: loading with mmap, "
            "the experts are streamed through the RAM tier (LLAMA_MOE_LOWRAM_MAP=0 keeps --load-mode none)\n", expert_bytes/1048576.0, avail/1048576.0);
        return LLAMA_LOAD_MODE_AUTO;
    }
    if (params.moe_cache_size <= 0 || load_mode != LLAMA_LOAD_MODE_AUTO || !(n_gpu_layers < 0 || n_gpu_layers > 1) || common_weight_stream_forced(params)) {
        return load_mode;
    }
    // the device lookup of the MoE cache needs pinned host experts and mmap'ed files usually cannot be pinned: load without mmap if the model fits in system memory and a device that will be used can do the lookup (the other backends keep the host lookup, mmap costs them nothing)
    // every device that gets layers must do the lookup: which device serves the host-expert layers is only known after placement, and a device without it keeps the host lookup that mmap serves as well
    std::vector<ggml_backend_dev_t> devs;
    for (ggml_backend_dev_t dev : params.devices) {
        if (dev != nullptr) {
            devs.push_back(dev);
        }
    }
    if (devs.empty()) {
        devs = common_default_model_devices();
    }
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE && !devs.empty()) {
        devs = params.main_gpu >= 0 && params.main_gpu < (int) devs.size() ? std::vector<ggml_backend_dev_t>{ devs[params.main_gpu] } : std::vector<ggml_backend_dev_t>{};
    }
    bool dev_lookup = !devs.empty();
    for (ggml_backend_dev_t dev : devs) {
        dev_lookup = dev_lookup && llama_moe_cache_device_lookup_supported(dev);
    }
    const char * env = getenv("LLAMA_MOE_CACHE_DEVICE");
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev_lookup && (env == nullptr || atoi(env) != 0) && cpu != nullptr) {
        size_t mem_free  = 0;
        size_t mem_total = 0;
        ggml_backend_dev_memory(cpu, &mem_free, &mem_total);
        uint64_t mem_avail = std::min(mem_free, mem_total);
        const uint64_t mem_sys = common_available_memory();
        if (mem_sys > 0) {
            mem_avail = std::min(mem_avail, mem_sys);
        }
        const uint64_t model_bytes = common_model_file_bytes(params.model.path);
        if (model_bytes > 0 && model_bytes <= 0.8*mem_avail) {
            // with host experts the load without mmap copies them into pinned memory, which is not reclaimable, while the page cache
            // still holds the file it reads them from: where both do not fit, the copy evicts the file it reads and the load reads it
            // from the disk again (L4, 53 GB RAM, Qwen3.6-35B-A3B UD-Q4_K_XL 21 GB at a 12 GB cap: 56 s vs 10.6 s with mmap). With mmap the
            // device lookup registers the mapped pages in place (no second copy). So without mmap only if the host experts and the file fit
            // in LLAMA_MOE_LOAD_NONE_RAM_FRAC (0.5) of the available memory; with every expert on a device nothing is pinned, the rule above.
            // G4 with a host RAM ballast, the same model at 12 GB (41 GB of host experts and file): MemAvailable 49 GB (84%) loaded without mmap
            // in 48.8 s vs 5.9 s with mmap, 59 GB (69%) still in 19-54 s, 177 GB (23%) in 8.2 s; the registered mapping reads slower once
            // loaded: decode 164.8 vs 179.2 tok/s, 2k prompt 5212 vs 7326 tok/s (--load-mode none keeps that steady state at the cost of the load)
            double none_frac = 0.5;
            if (const char * e = getenv("LLAMA_MOE_LOAD_NONE_RAM_FRAC")) {
                none_frac = std::max(0.0, atof(e));
            }
            const bool host_experts = common_experts_on_offloaded_host(params, params.tensor_buft_overrides.empty() ? nullptr :
                params.tensor_buft_overrides.data(), n_gpu_layers);
            uint64_t expert_bytes = 0;
            if (host_experts) {
                common_moe_slot_bytes sb;
                expert_bytes = common_moe_slot_bytes_of(params.model.path, sb) && sb.total > 0 ? (uint64_t) sb.total : model_bytes;
            }
            // a model file in tmpfs / ramfs is memory already, outside the available memory, and the copy cannot evict it: only the pinned
            // copy of the host experts counts there (G4, Flash-Next UD-IQ1_S in /dev/shm at 8 GB: 38000 + 69186 MiB against 50% of 108171 MiB
            // took the mmap load, 2k decode 60.4 vs 64.4 tok/s and prompt 1946 vs 2270 tok/s without mmap). LLAMA_MOE_LOAD_NONE_TMPFS=0 counts the file
            const char * env_tmpfs = getenv("LLAMA_MOE_LOAD_NONE_TMPFS");
            const bool file_in_ram = (env_tmpfs == nullptr || atoi(env_tmpfs) != 0) && common_model_file_in_ram(params.model.path);
            const uint64_t file_need = file_in_ram ? 0 : model_bytes;
            if (!host_experts || (double) (file_need + expert_bytes) <= none_frac*(double) mem_avail) {
                return LLAMA_LOAD_MODE_NONE;
            }
            if (file_in_ram) {
                COM_INF("MoE cache: load with mmap: the host experts (%.0f MiB, the model file is in tmpfs) do not fit in %.0f%% of the "
                    "available memory (%.0f MiB) for a load without mmap (--load-mode none forces it, LLAMA_MOE_LOAD_NONE_RAM_FRAC)\n",
                    expert_bytes/1048576.0, 100.0*none_frac, mem_avail/1048576.0);
            } else {
                COM_INF("MoE cache: load with mmap: the host experts (%.0f MiB) and the model file (%.0f MiB) do not both fit in %.0f%% of the "
                    "available memory (%.0f MiB) for a load without mmap (--load-mode none forces it, LLAMA_MOE_LOAD_NONE_RAM_FRAC)\n",
                    expert_bytes/1048576.0, model_bytes/1048576.0, 100.0*none_frac, mem_avail/1048576.0);
            }
        }
    }
    return load_mode;
}

// fit_deferred: the caller runs common_fit_params() and decides the MoE cache load mode afterwards with common_moe_cache_load_mode() on the fitted params
static struct llama_model_params common_model_params_to_llama_impl(common_params & params, bool fit_deferred) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.load_mode       = params.load_mode;
    // without a deferred fit decide from the params as they are now (a deferred fit decides after it has set n_gpu_layers and the MoE cache size)
    if (!fit_deferred) {
        mparams.load_mode = common_moe_cache_load_mode(params, mparams.load_mode, params.n_gpu_layers);
    }
    mparams.lazy_mode = params.lazy_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;
    mparams.load_mtp                    = params.load_mtp || std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    // k19-q38t: with a separate MTP drafter (-md + draft-mtp) the draft context runs on the drafter's own NextN block and draft head
    // (common_speculative_init_result), so the target's NextN layers and the restricted draft head built from them were loaded for
    // nothing: Qwen3.8-27B UD-Q3_K_XL on a T4, 569 MiB of device memory (12240 vs 11671 MiB of weights), enough to push 6 of 66 layers
    // to the host and drop the drafter there. The target is loaded without them (the drafter's file loads its own: its path is the
    // model path of the draft params). The target graph never reads them, so its text is unchanged.
    // LLAMA_MTP_MD_TARGET_NEXTN=1: load them as before
    if (mparams.load_mtp && !params.load_mtp && params.speculative.has_dft() && params.model.path != params.speculative.draft.mparams.path) {
        static const bool keep = getenv("LLAMA_MTP_MD_TARGET_NEXTN") != nullptr && atoi(getenv("LLAMA_MTP_MD_TARGET_NEXTN")) != 0;
        if (!keep) {
            mparams.load_mtp = false;
            static bool logged = false;
            if (!logged) {
                logged = true;
                COM_INF("k19-q38t: the target's own NextN layers are not loaded: the MTP drafter is '%s' (LLAMA_MTP_MD_TARGET_NEXTN=1 "
                    "loads them)\n", params.speculative.draft.mparams.path.c_str());
            }
        }
    }
    mparams.mtp_borrow_io               = params.mtp_borrow_io;

    return mparams;
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    return common_model_params_to_llama_impl(params, false);
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    cparams.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 0);
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;
    // speculative decoding with a model (a -md draft model, or a draft type that runs one: the built-in MTP heads, Eagle3, DFlash)
    cparams.spec_draft        = common_spec_draft_model(params);

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    cparams.kv_win       = params.kv_win;
    cparams.kv_win_sinks = params.kv_win_sinks;
    cparams.type_k_win   = params.cache_type_k_win;
    cparams.type_v_win   = params.cache_type_v_win;

    cparams.moe_cache_size = params.moe_cache_size;
    cparams.ple_prefetch   = params.ple_prefetch;

    cparams.weight_stream         = params.weight_stream;
    cparams.weight_stream_io      = params.weight_stream_io;
    cparams.weight_stream_threads = params.weight_stream_threads;
    cparams.weight_stream_ram_mib = params.weight_stream_ram_mib;
    cparams.moe_host_cache_mib    = params.moe_host_cache_mib;

    return cparams;
}

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

common_threadpools::~common_threadpools() {
    if (!free_fn) {
        return;
    }
    free_fn(threadpool);
    free_fn(threadpool_batch);
}

void common_threadpools::init(llama_context * ctx, const common_params & params) {
    GGML_ASSERT(!threadpool);
    GGML_ASSERT(!threadpool_batch);

    COM_INF("llama threadpool init, n_threads = %d\n", (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        COM_WRN("%s", "no CPU backend found\n");
        return;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    // each pool needs to match the respective n_threads exactly
    // see: https://github.com/ggml-org/llama.cpp/pull/27138#issuecomment-5332307332
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            COM_WRN("batch threadpool create failed : n_threads %d\n", tpp_batch.n_threads);
            return;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        COM_WRN("threadpool create failed : n_threads %d\n", tpp.n_threads);
        free_fn(threadpool_batch);
        threadpool_batch = nullptr;
        return;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        COM_ERR("failed to load control vector file from %s\n", load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        COM_WRN("no direction tensors found in %s\n", load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            COM_ERR("invalid/unparsable direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            COM_ERR("invalid (zero) direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            COM_ERR("invalid (non-F32) direction tensor type in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            COM_ERR("invalid (non-1D) direction tensor shape in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            COM_ERR("direction tensor in %s does not match previous dimensions\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        COM_WRN("skipping %s due to invalid direction tensors\n", load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            COM_ERR("control vectors in %s does not match previous dimensions\n", info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        COM_ERR("%s", "no valid control vector files passed\n");
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    common_batch batch(ctx);
    batch.add(last_token, pos, 0, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

static bool common_env_flag(const char * name, bool def) {
    const char * e = getenv(name);
    return e == nullptr || *e == '\0' ? def : atoi(e) != 0;
}

bool common_mtp_handoff_on(const char * lever) {
    static const bool all = common_env_flag("LLAMA_MTP_HANDOFF", true);
    return all && common_env_flag(("LLAMA_MTP_HANDOFF_" + std::string(lever)).c_str(), true);
}

bool common_mtp_handoff_trace() {
    static const bool on = common_env_flag("LLAMA_MTP_HANDOFF_TRACE", false);
    return on;
}

common_batch::common_batch(llama_context * ctx) : batch(llama_batch_ext_init(ctx)) {
    const auto rope_type = llama_model_rope_type(llama_get_model(ctx));
    n_pos = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? GGML_MROPE_SECTIONS : 1;
}

void common_batch::clear() {
    tokens.clear();
}

int32_t common_batch::add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output) {
    tokens.push_back({ id, { pos, 0, 0, 0 }, seq_id, output, { nullptr, 0, 0 }, { nullptr, 0, 0 }, {} });
    return size() - 1;
}

int32_t common_batch::add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output) {
    GGML_ASSERT(!seq_ids.empty());

    const int32_t idx = add(id, pos, seq_ids[0], output);
    for (size_t s = 1; s < seq_ids.size(); ++s) {
        add_seq(idx, seq_ids[s]);
    }
    return idx;
}

bool common_batch::add_seq(int32_t idx, llama_seq_id seq_id) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].seq_ids_extra.push_back(seq_id);
    return true;
}

bool common_batch::set_output(int32_t idx, bool value) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].output = value;
    return true;
}

bool common_batch::set_embd(int32_t idx, llama_embd embd) {
    if (idx < 0 || idx >= size() || tokens[idx].embd.data != nullptr) {
        return false;
    }
    tokens[idx].embd = embd;
    return true;
}

bool common_batch::set_embd_state(int32_t idx, llama_embd state) {
    if (idx < 0 || idx >= size() || tokens[idx].state.data != nullptr) {
        return false;
    }
    tokens[idx].state = state;
    return true;
}

int32_t common_batch::add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output) {
    token t = { LLAMA_TOKEN_NULL, { 0, 0, 0, 0 }, seq_id, output, embd, { nullptr, 0, 0 }, {} };
    for (int32_t j = 0; j < n_pos; ++j) {
        t.pos[j] = pos[j];
    }
    tokens.push_back(t);
    return size() - 1;
}

llama_batch_ext * common_batch::get_sub_batch(int32_t off, int32_t n) {
    GGML_ASSERT(batch && "common_batch was not initialized with a context");
    GGML_ASSERT(off >= 0 && n >= 0 && off + n <= size());

    llama_batch_ext * res = batch.get();
    llama_batch_ext_clear(res);
    // k19-host (LLAMA_BATCH_STATE_REF=0: copied as before): a batch of 64 entries or more (an MTP drafter's prompt pass) hands its state
    // rows to the context by reference: the views given to set_embd_state() stay valid until the decode that takes this batch returns
    // (the caller decodes it at once), so the rows are read in place instead of copied into the batch first (G4 Flash-Next + Q8_0 MTP
    // drafter: 240 MiB, ~13 ms on the server thread for the 6144-row segment of an 8k prompt)
    static const bool state_ref_env = getenv("LLAMA_BATCH_STATE_REF") == nullptr || atoi(getenv("LLAMA_BATCH_STATE_REF")) != 0;
    const bool state_ref = state_ref_env && n >= 64;
    {
        int32_t n_state = 0;
        for (int32_t i = off; i < off + n; ++i) {
            n_state += tokens[i].state.data != nullptr;
        }
        llama_batch_ext_reserve(res, n, state_ref ? 0 : n_state);
    }

    for (int32_t i = off; i < off + n; ++i) {
        const token & t = tokens[i];

        int32_t idx;
        if (t.id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(res, t.seq_id, t.id);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add token %d at index %d (error %d, n = %d)\n", __func__, t.id, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
            if (t.embd.data && !llama_batch_ext_set_embd_token(res, idx, t.embd)) {
                GGML_ABORT("%s: failed to set the embedding of token %d at index %d\n", __func__, t.id, i);
            }
        } else {
            idx = llama_batch_ext_add_embd(res, t.seq_id, t.embd);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add embedding at index %d (error %d, n = %d)\n", __func__, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
        }
        GGML_ASSERT(idx == i - off);

        for (const llama_seq_id seq_id : t.seq_ids_extra) {
            if (!llama_batch_ext_add_seq(res, idx, seq_id)) {
                GGML_ABORT("%s: failed to add seq %d to the entry at index %d\n", __func__, seq_id, i);
            }
        }
        if (t.output) {
            llama_batch_ext_set_output_logits(res, idx, true);
        }
        if (t.state.data) {
            if (state_ref) {
                llama_batch_ext_set_embd_state_ref(res, idx, t.state); // contexts without a state input ignore it
            } else {
                llama_batch_ext_set_embd_state(res, idx, t.state); // contexts without a state input ignore it
            }
        }
        if (t.decision_order != 0) {
            llama_batch_ext_set_decision_order(res, idx, (llama_decision_order) t.decision_order);
        }
    }

    return res;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_token * tokens, int32_t n_tokens) {
    common_batch batch(ctx);

    auto mem = llama_get_memory(ctx);
    llama_pos pos = llama_memory_seq_pos_max(mem, 0) + 1; // -1 + 1 == 0 when the memory is empty

    for (int32_t i = 0; i < n_tokens; ++i) {
        const bool output = i == n_tokens - 1;
        batch.add(tokens[i], pos, 0, output);
        pos++;
    }

    return batch;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_tokens & tokens) {
    return common_batch_get_one(ctx, tokens.data(), (int32_t) tokens.size());
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        llama_tokens prefix_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_tokens_before_last);
        common_batch batch_prefix = common_batch_get_one(ctx, prefix_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_prefix.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        COM_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        common_batch batch_last(ctx);
        batch_last.add(all_tokens.back(), n_past, 0, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_last.get())) {
            COM_ERR("%s", "failed to eval last token\n");
            return false;
        }
        n_past++;
    } else {
        llama_tokens new_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_new);
        common_batch batch = common_batch_get_one(ctx, new_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

static bool ckpt_pool_on() {
    static const bool on = getenv("LLAMA_CKPT_POOL") == nullptr || atoi(getenv("LLAMA_CKPT_POOL")) != 0;
    return on;
}

static std::mutex                    g_ckpt_pool_mutex;
static std::vector<common_state_buf> g_ckpt_pool;
// k17: capacities of the buffers a background copy returns to the pool once it is done (common_state_buf_pending_add)
static std::vector<size_t>           g_ckpt_pool_pending;

// k11: the pool keeps at most this many bytes (LLAMA_STATE_POOL_MIB, default 1024) and 8 buffers
// largest buffer asked of the pool so far (the lock is held by the callers that write it)
static size_t g_ckpt_pool_req_max = 0;

// ... and at most twice the largest request (one state being saved, one ready for the next save): a pool kept at its byte cap held up to
// 1 GiB idle at the end of a run, 6-7x the state of LFM2.5. k17: default on (LLAMA_STATE_POOL_REL=0: the byte cap alone), with the
// request save's checkpoint copies on a thread (LLAMA_PCS_CKPT_COPY_ASYNC) the T4 multi-turn run ends below the copies made in place:
// LFM2.5 2553 vs 2740 MiB, Qwen3.6 MTP 23379 vs 23975 MiB (with LLAMA_STATE_POOL_PENDING and _SMALL_FRESH), same text, same VmHWM,
// G4 q4x 12 GB chat TTFT 101 vs 125 ms
static size_t ckpt_pool_limit() {
    static const size_t lim = (size_t) (getenv("LLAMA_STATE_POOL_MIB") ? std::max(0, atoi(getenv("LLAMA_STATE_POOL_MIB"))) : 1024) << 20;
    static const bool rel = getenv("LLAMA_STATE_POOL_REL") == nullptr || atoi(getenv("LLAMA_STATE_POOL_REL")) != 0;
    return rel ? std::min(lim, 2*g_ckpt_pool_req_max + (g_ckpt_pool_req_max/8)) : lim;
}

// LLAMA_STATE_POOL_FIT=N (opt-in): a buffer is handed out for n only up to max(2, N) times n
static size_t ckpt_pool_fit() {
    static const size_t fit = getenv("LLAMA_STATE_POOL_FIT") != nullptr && atoi(getenv("LLAMA_STATE_POOL_FIT")) > 0 ?
        (size_t) std::max(2, atoi(getenv("LLAMA_STATE_POOL_FIT"))) : 0;
    return fit;
}

// keep the pool within its limits: drop the smallest buffers first (the lock is held by the caller)
static void ckpt_pool_trim() {
    auto total = [] { size_t t = 0; for (const auto & b : g_ckpt_pool) { t += b.capacity(); } return t; };
    while (!g_ckpt_pool.empty() && (g_ckpt_pool.size() > 8 || total() > ckpt_pool_limit())) {
        auto it = std::min_element(g_ckpt_pool.begin(), g_ckpt_pool.end(),
            [](const common_state_buf & a, const common_state_buf & b) { return a.capacity() < b.capacity(); });
        g_ckpt_pool.erase(it);
    }
}

void common_state_buf_recycle(common_state_buf && buf) {
    if (!ckpt_pool_on() || buf.capacity() < (1u << 20)) {
        return; // small buffers are not worth keeping (buf is freed by its owner)
    }
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    g_ckpt_pool.push_back(std::move(buf));
    ckpt_pool_trim();
}

// k17 (LLAMA_STATE_POOL_PENDING=0: off): a buffer on its way back to the pool (the originals of the checkpoint copies a request save runs
// on a thread, LLAMA_PCS_CKPT_COPY_ASYNC) counts as pooled for the prefault worker. Without it the worker saw a pool short of the
// buffers the copy still held, faulted in fresh ones, and the originals then landed on top: the pool ended fuller than with the copies
// made in place, where the slot's erase returns the originals before the worker runs (T4 LFM2.5 multi-turn end RSS +577 MiB)
static bool ckpt_pool_pending_on() {
    static const bool on = getenv("LLAMA_STATE_POOL_PENDING") == nullptr || atoi(getenv("LLAMA_STATE_POOL_PENDING")) != 0;
    return on;
}

void common_state_buf_pending_add(size_t capacity) {
    if (!ckpt_pool_on() || !ckpt_pool_pending_on() || capacity < (1u << 20)) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    g_ckpt_pool_pending.push_back(capacity);
}

void common_state_buf_pending_drop(size_t capacity) {
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    auto it = std::find(g_ckpt_pool_pending.begin(), g_ckpt_pool_pending.end(), capacity);
    if (it != g_ckpt_pool_pending.end()) {
        g_ckpt_pool_pending.erase(it);
    }
}

void common_state_buf_recycle_pending(common_state_buf && buf) {
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    auto it = std::find(g_ckpt_pool_pending.begin(), g_ckpt_pool_pending.end(), buf.capacity());
    if (it != g_ckpt_pool_pending.end()) {
        g_ckpt_pool_pending.erase(it);
    }
    if (!ckpt_pool_on() || buf.capacity() < (1u << 20)) {
        return;
    }
    g_ckpt_pool.push_back(std::move(buf));
    ckpt_pool_trim();
}

bool common_state_thp_on() {
#if defined(__linux__)
    static const bool on = getenv("LLAMA_STATE_THP") != nullptr && atoi(getenv("LLAMA_STATE_THP")) != 0;
    return on;
#else
    return false;
#endif
}

void * common_state_thp_alloc(size_t bytes) {
#if defined(__linux__)
    void * p = nullptr;
    if (posix_memalign(&p, 2u << 20, bytes) != 0 || p == nullptr) {
        throw std::bad_alloc();
    }
    // whole huge pages only: the tail stays on 4 KiB pages, so the resident size is the bytes used, not rounded up to 2 MiB
    const size_t huge = bytes & ~((size_t) (2u << 20) - 1);
    if (huge > 0) {
        madvise(p, huge, MADV_HUGEPAGE);
    }
    return p;
#else
    return ::operator new(bytes);
#endif
}

void common_state_thp_free(void * p) {
#if defined(__linux__)
    free(p);
#else
    ::operator delete(p);
#endif
}

// k11 (LLAMA_STATE_PREFAULT, default on with the pool, =0 off): a background thread keeps buffers whose pages are already resident
// in the pool, so the prompt-cache save and the checkpoints of the next request do not fault in fresh memory on the TTFT path
static bool state_prefault_on() {
    static const bool on = ckpt_pool_on() && (getenv("LLAMA_STATE_PREFAULT") == nullptr || atoi(getenv("LLAMA_STATE_PREFAULT")) != 0);
    return on;
}

namespace {
struct state_prefault_worker {
    std::mutex              m;
    std::condition_variable cv;
    struct want_t { size_t n; int count; int fit; };
    std::vector<want_t> want; // (bytes, count, fit: a ready buffer is at most fit times n, 0 = the pool's rule)
    bool        stop = false;
    std::thread th;

    void request(size_t n, int count, int fit = 0) {
        std::lock_guard<std::mutex> lock(m);
        if (stop) {
            return;
        }
        want.push_back({ n, count, fit });
        if (!th.joinable()) {
            th = std::thread([this] { run(); });
        }
        cv.notify_one();
    }

    void run() {
        std::unique_lock<std::mutex> lock(m);
        while (true) {
            cv.wait(lock, [this] { return stop || !want.empty(); });
            if (stop) {
                return;
            }
            auto w = std::move(want);
            want.clear();
            lock.unlock();
            for (const auto & r : w) {
                fill(r.n, r.count, r.fit);
            }
            lock.lock();
        }
    }

    void fill(size_t n, int count, int fit = 0) {
        const size_t n_alloc = n + n/16; // room for a slightly longer prompt
        while (true) {
            {
                std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
                g_ckpt_pool_req_max = std::max(g_ckpt_pool_req_max, n);
                int    have  = 0;
                size_t total = 0;
                for (const auto & b : g_ckpt_pool) {
                    // with the fit, a buffer over twice the size is not handed out for n: it does not count as ready (else the fill
                    // stopped and the next checkpoint of this size faulted in fresh pages on the request path)
                    have  += b.capacity() >= n && (fit > 0 ? b.capacity() / fit <= n : (!ckpt_pool_fit() || b.capacity() / ckpt_pool_fit() <= n));
                    total += b.capacity();
                }
                for (const size_t cap : g_ckpt_pool_pending) { // k17: the buffers a background copy returns shortly
                    have  += cap >= n && (fit > 0 ? cap / fit <= n : (!ckpt_pool_fit() || cap / ckpt_pool_fit() <= n));
                    total += cap;
                }
                if (have >= count || g_ckpt_pool.size() + g_ckpt_pool_pending.size() >= 8 || total + n_alloc > ckpt_pool_limit()) {
                    return;
                }
            }
            common_state_buf b;
            try {
                b.resize(n_alloc);
            } catch (const std::bad_alloc &) {
                return;
            }
            memset(b.data(), 0, b.size()); // fault the pages in here, not on the save path
            std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
            g_ckpt_pool.push_back(std::move(b));
            ckpt_pool_trim();
        }
    }

    ~state_prefault_worker() {
        {
            std::lock_guard<std::mutex> lock(m);
            stop = true;
        }
        cv.notify_one();
        if (th.joinable()) {
            th.join();
        }
    }
};
}

// constructed after the pool, so destroyed (joined) before it
static state_prefault_worker g_state_prefault;

void common_state_buf_prefault(size_t n, int count) {
    if (state_prefault_on() && n >= (1u << 20) && count > 0) {
        g_state_prefault.request(n, count);
    }
}

std::string common_state_buf_pool_desc() {
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    std::string r;
    for (const auto & b : g_ckpt_pool) {
        r += string_format("%s%.1f", r.empty() ? "" : ",", b.capacity() / 1048576.0);
    }
    if (!g_ckpt_pool_pending.empty()) {
        r += " pending";
        for (const size_t cap : g_ckpt_pool_pending) {
            r += string_format(" %.1f", cap / 1048576.0);
        }
    }
    return r;
}

// LLAMA_STATE_CKPT_FIT: unset = auto (off until common_state_ckpt_fit_enable), 1 = always, 0 = never
static std::atomic<bool> g_ckpt_fit_auto_on { false };
bool common_state_ckpt_fit() {
    static const int mode = getenv("LLAMA_STATE_CKPT_FIT") != nullptr ? (atoi(getenv("LLAMA_STATE_CKPT_FIT")) != 0 ? 1 : 0) : -1;
    return mode == 1 || (mode < 0 && g_ckpt_fit_auto_on.load(std::memory_order_relaxed));
}
void common_state_ckpt_fit_enable() {
    g_ckpt_fit_auto_on.store(true, std::memory_order_relaxed);
}

void common_state_buf_prefault_fit(size_t n, int count, int fit) {
    if (state_prefault_on() && n >= (1u << 20) && count > 0) {
        g_state_prefault.request(n, count, fit);
    }
}

size_t common_state_buf_pool_bytes() {
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    size_t t = 0;
    for (const auto & b : g_ckpt_pool) { t += b.capacity(); }
    return t;
}

// k17 (LLAMA_STATE_POOL_SMALL_FRESH=0: off, any request takes the smallest pooled buffer that fits): a request below the pool's minimum
// (1 MiB, the size it keeps) gets fresh memory, not a pooled buffer. Every pooled buffer is a state or a large checkpoint, so a small
// checkpoint took one whole and kept it resident for as long as it lived: T4 LFM2.5 multi-turn, 6 checkpoints of 2.8 MiB held 308 MiB
// in the slot and 1.4 MiB held 389 MiB in the cache entry. Faulting in a fresh buffer this small costs microseconds
static bool ckpt_pool_small_fresh() {
    static const bool on = getenv("LLAMA_STATE_POOL_SMALL_FRESH") == nullptr || atoi(getenv("LLAMA_STATE_POOL_SMALL_FRESH")) != 0;
    return on;
}

bool common_state_buf_take(common_state_buf & buf, size_t n) {
    if (!ckpt_pool_on() || buf.capacity() >= n) {
        return false;
    }
    if (n < (1u << 20) && ckpt_pool_small_fresh()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    g_ckpt_pool_req_max = std::max(g_ckpt_pool_req_max, n);
    // the smallest that fits
    auto best = g_ckpt_pool.end();
    for (auto it = g_ckpt_pool.begin(); it != g_ckpt_pool.end(); ++it) {
        if (it->capacity() >= n && (best == g_ckpt_pool.end() || it->capacity() < best->capacity())) {
            best = it;
        }
    }
    // a buffer more than twice the request is left in the pool: a small checkpoint that took a state buffer kept all of it resident for as
    // long as the checkpoint lived (its capacity is not in the prompt cache's size), e.g. 2.8 MiB of checkpoints held 149 MiB on LFM2.5
    // (opt-in LLAMA_STATE_POOL_FIT=1: G4 q4x reuse turns +1.4-2.5 ms with it and REL; default: any buffer that fits)
    if (best == g_ckpt_pool.end() || (ckpt_pool_fit() && best->capacity() / ckpt_pool_fit() > n)) {
        return false;
    }
    buf = std::move(*best);
    g_ckpt_pool.erase(best);
    return true;
}

bool common_state_buf_take_fit(common_state_buf & buf, size_t n, int fit) {
    if (!ckpt_pool_on() || buf.capacity() >= n || fit <= 0) {
        return fit <= 0 ? common_state_buf_take(buf, n) : false;
    }
    if (n < (1u << 20) && ckpt_pool_small_fresh()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_ckpt_pool_mutex);
    g_ckpt_pool_req_max = std::max(g_ckpt_pool_req_max, n);
    auto best = g_ckpt_pool.end();
    for (auto it = g_ckpt_pool.begin(); it != g_ckpt_pool.end(); ++it) {
        if (it->capacity() >= n && it->capacity() / fit <= n && (best == g_ckpt_pool.end() || it->capacity() < best->capacity())) {
            best = it;
        }
    }
    if (best == g_ckpt_pool.end()) {
        return false;
    }
    buf = std::move(*best);
    g_ckpt_pool.erase(best);
    return true;
}

static bool ckpt_no_zero_fill() {
    static const bool on = common_mtp_handoff_on("CKPT");
    return on;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    // k17 (LLAMA_STATE_CKPT_FIT=0: any pooled buffer that fits): a new checkpoint takes a pooled buffer of at most twice its size, and the
    // prefault worker keeps two of those ready. Every pooled buffer is a state or a checkpoint, and a checkpoint that took a state buffer kept
    // it whole for its life, in the slot and in the cache entry it was lent or moved to: G4 resident Qwen3.6 MTP multi-turn, 8 checkpoints
    // of 502 MiB held 860-980 MiB, the entry's 691 MiB held 1387 MiB with the idle lend (1173 MiB without). Copies (clones) keep the
    // pool's rule: they come several at once on a reuse turn, where a fitted pool faulted fresh pages on the request path (round 26)
    // default auto: only once the server lends checkpoints at idle (common_state_ckpt_fit_enable, resident placements): T4 Qwen3.6 MTP
    // (spilled, no lend) ended +152 MiB with the fit always on (fresh fitted buffers beside the pooled state buffers) and reuse turns
    // 2-5% slower; G4 resident -339 MiB with it (3649 vs 3988), where the lent checkpoints otherwise kept the state buffers whole
    if (common_state_ckpt_fit()) {
        common_state_buf_take_fit(data_tgt, ckpt_size, 2);
        common_state_buf_prefault_fit(ckpt_size, 2, 2);
    } else {
        common_state_buf_take(data_tgt, ckpt_size);
        common_state_buf_prefault(ckpt_size, 2);
    }
    data_tgt.resize(ckpt_size);
    if (!ckpt_no_zero_fill()) {
        std::fill(data_tgt.begin(), data_tgt.end(), 0); // LLAMA_MTP_HANDOFF_CKPT=0: zero-filled as before
    }

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    if (common_state_ckpt_fit()) {
        common_state_buf_take_fit(data_dft, ckpt_size, 2); // as update_tgt
    } else {
        common_state_buf_take(data_dft, ckpt_size);
    }
    data_dft.resize(ckpt_size);
    if (!ckpt_no_zero_fill()) {
        std::fill(data_dft.begin(), data_dft.end(), 0); // LLAMA_MTP_HANDOFF_CKPT=0: zero-filled as before
    }

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

bool common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_tgt.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    return n == data_tgt.size();
}

bool common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_dft.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    return n == data_dft.size();
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
