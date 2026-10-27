// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmid.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvf-splitk.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/shortconv-fused.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/moe-cache.cuh"
#include "ggml-moe-mailbox.h"
#include "ggml-cuda/copy-regions.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

// debug (GGML_CUDA_FAKE_FREE_MIB): the device memory this process allocated, per device (cudaMalloc through ggml_cuda_device_malloc, the
// virtual memory pool, the cuBLAS workspaces), so a test on a shared card sees N MiB minus its own allocations as free
static std::mutex                     ggml_cuda_own_mutex;
static std::map<void *, std::pair<int, size_t>> ggml_cuda_own_ptrs;
static int64_t                        ggml_cuda_own_bytes[GGML_CUDA_MAX_DEVICES] = {};

void ggml_cuda_own_add(int device, int64_t bytes) {
    std::lock_guard<std::mutex> lock(ggml_cuda_own_mutex);
    ggml_cuda_own_bytes[device] += bytes;
}

static void ggml_cuda_device_free(void * ptr) {
    if (ptr == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_own_mutex);
        auto it = ggml_cuda_own_ptrs.find(ptr);
        if (it != ggml_cuda_own_ptrs.end()) {
            ggml_cuda_own_bytes[it->second.first] -= (int64_t) it->second.second;
            ggml_cuda_own_ptrs.erase(it);
        }
    }
    CUDA_CHECK(cudaFree(ptr));
}

static cudaError_t ggml_cuda_device_malloc_impl(void ** ptr, size_t size, int device);

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    // debug: GGML_CUDA_FAKE_FREE_STRICT=1 with GGML_CUDA_FAKE_FREE_MIB fails an allocation past the fake budget, as cudaMalloc does on a
    // card of that size (the fake free memory alone lets it succeed on a larger card)
    static const bool fake_strict = getenv("GGML_CUDA_FAKE_FREE_STRICT") != nullptr && atoi(getenv("GGML_CUDA_FAKE_FREE_STRICT")) != 0 &&
                                    getenv("GGML_CUDA_FAKE_FREE_MIB") != nullptr && atoll(getenv("GGML_CUDA_FAKE_FREE_MIB")) > 0;
    if (fake_strict) {
        const int64_t cap = (int64_t) atoll(getenv("GGML_CUDA_FAKE_FREE_MIB")) << 20;
        std::lock_guard<std::mutex> lock(ggml_cuda_own_mutex);
        if (ggml_cuda_own_bytes[device] + (int64_t) size > cap) {
            *ptr = nullptr;
            return cudaErrorMemoryAllocation;
        }
    }
    const cudaError_t err = ggml_cuda_device_malloc_impl(ptr, size, device);
    if (err == cudaSuccess && *ptr != nullptr) {
        std::lock_guard<std::mutex> lock(ggml_cuda_own_mutex);
        ggml_cuda_own_ptrs[*ptr] = { device, size };
        ggml_cuda_own_bytes[device] += (int64_t) size;
    }
    return err;
}

static cudaError_t ggml_cuda_device_malloc_impl(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                ggml_cuda_device_free(b.ptr);
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        ggml_cuda_device_free(ptr);
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
            ggml_cuda_own_add(device, -(int64_t) pool_size);
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));
            ggml_cuda_own_add(device, (int64_t) reserve_size);

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (src1_memo.n_reused > 0 || src1_memo.n_produced > 0) {
        GGML_LOG_INFO("%s: src1 memo: %lld conversions reused (%.1f MiB of src1 not converted again), %lld written by the norm "
            "(GGML_CUDA_SRC1_MEMO=0 / GGML_CUDA_NORM_CONV=0 turn it off)\n",
            __func__, (long long) src1_memo.n_reused, src1_memo.bytes_skipped / 1048576.0, (long long) src1_memo.n_produced);
    }

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (mmvf_splitk_part != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(mmvf_splitk_part));
        CUDA_CHECK(cudaFree(mmvf_splitk_cnt));
    }
    if (hc_pre_cnt != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(hc_pre_cnt));
    }
    for (cudaEvent_t ev : copy_peer_events) {
        if (ev != nullptr) {
            CUDA_CHECK(cudaEventDestroy(ev));
        }
    }
    if (moe_fork != nullptr) {
        CUDA_CHECK(cudaEventDestroy(moe_fork));
        CUDA_CHECK(cudaEventDestroy(moe_join));
    }
    if (side_mv_fork != nullptr) {
        CUDA_CHECK(cudaEventDestroy(side_mv_fork));
    }
    for (cudaEvent_t ev : side_mv_join) {
        CUDA_CHECK(cudaEventDestroy(ev));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        ggml_cuda_device_free(dev_ptr);
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // k11: an F16 / BF16 copy of src1 shared with the next MUL_MAT of the graph (see ggml_cuda_src1_memo); only when src0 needs
    // no conversion, so that the shared copy is the first pool allocation of this product
    ggml_cuda_src1_memo & memo = ctx.src1_memo;
    const bool memo_ok = memo.top(dst) && compute_type != GGML_TYPE_F32 && src0->type == compute_type && src1->type == GGML_TYPE_F32 &&
        ggml_is_contiguously_allocated(src1);
    const int64_t memo_key[ggml_cuda_src1_memo::n_key] = { (int64_t) compute_type, ne10, ne11, ne12, ne13,
        (int64_t) nb11, (int64_t) nb12, (int64_t) nb13, 0, 0 };
    const bool memo_hit  = memo_ok && memo.match(2, src1, memo_key);
    const bool memo_hold = memo_ok && !memo_hit && memo.share_next;
    // a held copy this product does not read stays for its own reader, unless this product keeps one itself
    if (memo_hold) {
        memo.release();
    }

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else if (memo_hit || memo_hold) {
        if (memo_hit) {
            memo.note_reuse(src1);
        } else {
            memo.hold(ctx.pool(), 2, src1, memo_key, ggml_nelements(src1)*sizeof(cuda_t));
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, (cuda_t *) memo.ptr, ggml_nelements(src1), main_stream);
        }
        src1_ptr = (const cuda_t *) memo.ptr;
        const size_t src1_bs = ggml_blck_size(src1->type);
        s11 *= src1_bs;
        s12 *= src1_bs;
        s13 *= src1_bs;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// k15-lfm2: the kernels of the flagged verify products. One INFO line per load (the first graph with verify products, see
// ggml_cuda_verify_mmq_summary) counts them by kernel, so the engagement of GGML_CUDA_VERIFY_MMQ shows in any server log;
// GGML_CUDA_VERIFY_MMQ_LOG=1 also names each product once per kernel and width
struct ggml_cuda_verify_mmq_state {
    std::mutex                         mtx;
    std::map<std::string, std::string> last;  // weight name -> kernel, in the first verify graph
    std::map<std::string, std::string> seen;  // name + width -> kernel (GGML_CUDA_VERIFY_MMQ_LOG)
    int                                cc      = 0;
    int64_t                            cols    = 0;
    std::atomic<bool>                  printed { false };
};
static ggml_cuda_verify_mmq_state & ggml_cuda_verify_mmq_st() {
    static ggml_cuda_verify_mmq_state st;
    return st;
}

static void ggml_cuda_verify_mmq_log(const ggml_tensor * dst, int cc, const char * what) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_VERIFY_MMQ_LOG");
        return e != nullptr && atoi(e) != 0;
    }();
    auto & st = ggml_cuda_verify_mmq_st();
    const ggml_tensor * src0 = dst->src[0];
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!st.printed) {
        st.last[src0->name] = what;
        st.cc   = cc;
        st.cols = std::max<int64_t>(st.cols, dst->src[1]->ne[1]);
    }
    if (!on || st.seen.size() > 1024) {
        return;
    }
    std::string & prev = st.seen[std::string(src0->name) + "/" + std::to_string(dst->src[1]->ne[1])];
    if (prev == what) {
        return;
    }
    prev = what;
    GGML_LOG_INFO("ggml_cuda: verify product %s (%s %lld x %lld, %lld columns, cc %d): %s\n", src0->name, ggml_type_name(src0->type),
        (long long) src0->ne[0], (long long) src0->ne[1], (long long) dst->src[1]->ne[1], cc, what);
}

// after a graph: the one line per load, once a graph had verify products
static void ggml_cuda_verify_mmq_summary() {
    auto & st = ggml_cuda_verify_mmq_st();
    if (st.printed.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lock(st.mtx);
    if (st.printed || st.last.empty()) {
        return;
    }
    st.printed = true;
    std::map<std::string, int> n;
    for (const auto & [name, what] : st.last) {
        n[what]++;
    }
    std::string list;
    for (const auto & [what, k] : n) {
        list += (list.empty() ? "" : ", ") + std::to_string(k) + " " + what;
    }
    const char * e = getenv("GGML_CUDA_VERIFY_MMQ");
    GGML_LOG_INFO("ggml_cuda: verify products of the first verify graph (cc %d, up to %lld columns, GGML_CUDA_VERIFY_MMQ=%s): %s "
        "(=0 off, GGML_CUDA_VERIFY_MMQ_LOG=1 names them)\n", st.cc, (long long) st.cols, e != nullptr ? e : "default", list.c_str());
    st.last.clear();
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    // [TAG_GGML_PREC] F32 weights at GGML_PREC_F32: avoid TF32 rounding (MMF tensor cores, cuBLAS tensor-op math)
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ggml_get_op_params_i32(dst, 3) == GGML_PREC_F32) {
        const int cc_f32        = ggml_cuda_info().devices[ctx.device].cc;
        const int warp_size_f32 = ggml_cuda_info().devices[ctx.device].warp_size;
        if (!bad_padding_clear && ne11 <= MMVF_MAX_BATCH_SIZE &&
                ggml_cuda_should_use_mmvf(src0->type, cc_f32, warp_size_f32, src0->ne, src0->nb, /*ne11 =*/ 1)) {
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            return;
        }
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_DEFAULT_MATH));
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));
        return;
    }

    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // batch invariant (speculative verify batches): every column gets the bytes a single column (ne11 == 1) would get
    if ((dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && ne11 > 1) {
        // k15-lfm2, GGML_CUDA_VERIFY_MMQ: a quantized dense verify product of up to 16 columns runs MMQ (int8 tensor cores) instead
        // of the batched mmvq: the same q8_1 activations into the same int8 products with f32 accumulation, only the order of the f32
        // scale products and sums differs (plans/k11.md equivalence rule). T4 at 8 rows the dp4a mmvq is compute-bound (2x one row,
        // plans/k15_lfm2.md section 15). Default on Turing (cc 7.5) only, where it was timed and passed the KLD gate; =1 on any device with
        // the Turing MMA (Ampere and newer too), =0 off. Not here: the products of a multi-matrix mmvq launch (ggml_cuda_mmvq_multi_*),
        // which keep the batched mmvq. GGML_CUDA_VERIFY_MMQ_LOG=1 names each product once with the kernel it takes
        static const int verify_mmq_env = [] {
            const char * e = getenv("GGML_CUDA_VERIFY_MMQ");
            return e != nullptr ? atoi(e) : -1;
        }();
        const bool verify_mmq = verify_mmq_env > 0 ||
            (verify_mmq_env < 0 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING && cc < GGML_CUDA_CC_AMPERE);
        const bool take_mmq = verify_mmq && ne11 <= 16 && ne12*ne13 == 1 && ggml_is_quantized(src0->type) && turing_mma_available(cc) &&
                ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0);
        ggml_cuda_verify_mmq_log(dst, cc, take_mmq ? "MMQ" : verify_mmq ? "batched mmvq (not an MMQ product)" : "batched mmvq (VERIFY_MMQ off here)");
        if (take_mmq) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
            return;
        }
        // the vector kernels compute each column like their single column launch (mmvq: with the warps of that launch)
        if (ne11 <= MMVF_MAX_BATCH_SIZE && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, 1)) {
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            return;
        }
        if (ne11 <= MMVQ_MAX_BATCH_SIZE && !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, 1, false) &&
                ggml_cuda_should_use_mmvq(src0->type, cc, 1)) {
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            return;
        }
        // any other kernel: one column at a time
        if (ne11 <= 16 && ne12*ne13 == 1) {
            for (int64_t j = 0; j < ne11; ++j) {
                ggml_tensor src1_j = *src1;
                ggml_tensor dst_j  = *dst;
                src1_j.ne[1] = 1;
                dst_j.ne[1]  = 1;
                src1_j.data = (char *) src1->data + j*src1->nb[1];
                dst_j.data  = (char *)  dst->data + j* dst->nb[1];
                ggml_cuda_mul_mat(ctx, src0, &src1_j, &dst_j);
            }
            return;
        }
    }

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    // k13-pp: GGML_CUDA_DENSE_CUBLAS_MIN=N (opt-in, 0 = off): quantized dense products of at least N columns are dequantized to FP16 and
    // multiplied by cuBLAS instead of MMQ (Strata's prompt path for dense layers). Other rounding (KLD-gated)
    static const int dense_cublas_min = getenv("GGML_CUDA_DENSE_CUBLAS_MIN") ? atoi(getenv("GGML_CUDA_DENSE_CUBLAS_MIN")) : 0;
    if (dense_cublas_min > 0 && ggml_is_quantized(src0->type) && ne11 >= dense_cublas_min && GGML_CUDA_CC_IS_NVIDIA(cc)) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            // batch invariant: the single token mul_mat_vec_q per token (see ggml_cuda_mul_mat_id)
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc) || (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                // batch invariant (verify batches): mul_mat_vec_q runs the single token kernel per token
                if (ne2 <= mmvq_mmid_max || (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 3*ne_get_rows);

    int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    int32_t * ids_dst_compact = ids_buf_dev.ptr + 1*ne_get_rows;
    int32_t * ids_from_sorted = ids_buf_dev.ptr + 2*ne_get_rows;

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    // sort the ids by expert on the device with the MMQ/MMF helper, else on the host
    const bool sort_on_device = ids->type == GGML_TYPE_I32 && ids->nb[0] == ggml_element_size(ids) &&
        ggml_cuda_mm_ids_helper_supported(ne02, ne12, n_expert_used);

    if (sort_on_device) {
        // expert_bounds, then the number of skipped slots (id -1)
        ggml_cuda_pool_alloc<int32_t> expert_bounds_dev(ctx.pool(), ne02 + 2);

        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;
        GGML_ASSERT(sis1 > 0);

        ggml_cuda_launch_mm_ids_helper(ctx.pool(), (const int32_t *) ids->data, ids_to_sorted, ids_dst_compact, expert_bounds_dev.ptr,
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_launch_mm_ids_count_skipped((const int32_t *) ids->data, expert_bounds_dev.ptr + ne02 + 1, ne12, n_expert_used, si1, stream);
        CUDA_CHECK(cudaGetLastError());

        // the GEMMs below need the per-expert token counts on the host
        std::vector<int32_t> expert_bounds_host(ne02 + 2);
        CUDA_CHECK(cudaMemcpyAsync(expert_bounds_host.data(), expert_bounds_dev.ptr,
            (ne02 + 2)*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // fails for out of range ids other than -1 (skipped slot), and for repeated ids on the single-warp path
        const int32_t n_filled = expert_bounds_host[ne02];
        GGML_ASSERT(n_filled + expert_bounds_host[ne02 + 1] == ne_get_rows);
        GGML_ASSERT(expert_bounds_host[0] == 0);

        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            tokens_per_expert[i02] = expert_bounds_host[i02 + 1] - expert_bounds_host[i02];
        }

        // skipped slots have no compact row: point them at row 0, the final zero_dst overwrites them
        if (n_filled < ne_get_rows) {
            CUDA_CHECK(cudaMemsetAsync(ids_from_sorted, 0, ne_get_rows*sizeof(int32_t), stream));
        }
        ggml_cuda_launch_mm_ids_invert(ids_dst_compact, ids_from_sorted, n_filled, ne_get_rows, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {
        std::vector<int32_t> ids_to_sorted_host;
        ids_to_sorted_host.reserve(2*ne_get_rows);
        std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

        std::vector<char> ids_host(ggml_nbytes(ids));
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                    const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                    // GGML_ASSERT, not assert: plain assert compiles out in a release build
                    GGML_ASSERT(expert_to_use == -1 || (expert_to_use >= 0 && expert_to_use < ne02));
                    if (expert_to_use == i02) {
                        ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                        ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                        tokens_per_expert[i02]++;
                        break;
                    }
                }
            }
        }
        GGML_ASSERT(ids_to_sorted_host.size() <= size_t(ne_get_rows));

        // skipped slots shrink the compact set - pad it so that the get_rows below stay in bounds
        ids_to_sorted_host.resize(ne_get_rows, 0);

        ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

        CUDA_CHECK(cudaMemcpyAsync(ids_to_sorted, ids_to_sorted_host.data(),
            ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ids_from_sorted, ids_to_sorted_host.data() + ne_get_rows,
            ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }
    // the q8_1 copies of the slices above must not outlive src1_sorted: the pool hands its memory out again
    ctx.q8_1_cache.invalidate(src1_sorted.ptr, ne12*n_expert_used*ne10*ts_src1_sorted);

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);

    // the get_rows above copies a placeholder row into the skipped slots, overwrite them with zeros
    ggml_cuda_launch_mm_ids_zero_dst((const int32_t *) ids->data, (float *) dst->data,
        ne2, n_expert_used, ne0, ids->nb[1]/sizeof(int32_t), nb1/sizeof(float), nb2/sizeof(float), stream);
    CUDA_CHECK(cudaGetLastError());
}

// k15-fn debug (GGML_CUDA_GRAPH_NODE_STATS=3): the copy / gather nodes executed after the fusions, logged per captured graph
static std::map<const void *, std::map<std::string, std::pair<int, size_t>>> ggml_cuda_exec_copy_ops;

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    static const bool exec_stats = getenv("GGML_CUDA_GRAPH_NODE_STATS") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_NODE_STATS")) >= 3;
    static const bool exec_all = exec_stats && atoi(getenv("GGML_CUDA_GRAPH_NODE_STATS")) >= 4; // =4: every op run unfused
    if (exec_stats && (exec_all || dst->op == GGML_OP_CPY || dst->op == GGML_OP_CONT || dst->op == GGML_OP_DUP || dst->op == GGML_OP_CONCAT ||
                       dst->op == GGML_OP_GET_ROWS || dst->op == GGML_OP_REPEAT || dst->op == GGML_OP_SET_ROWS) && ggml_nelements(dst) > 0) {
        std::string nm;
        for (const char * c = dst->name; *c; ++c) {
            if (*c < '0' || *c > '9') { nm += *c; }
        }
        const ggml_tensor * s0 = dst->src[0];
        char key[256];
        snprintf(key, sizeof(key), "%s%s%s %s [%lld,%lld,%lld] <- %s %s", ggml_op_desc(dst), dst->op == GGML_OP_UNARY || dst->op == GGML_OP_GLU ? "" : "",
            "", nm.c_str(), (long long) dst->ne[0], (long long) dst->ne[1],
            (long long) dst->ne[2], s0 ? ggml_op_desc(s0) : "-", s0 ? s0->name : "");
        auto & e = ggml_cuda_exec_copy_ops[&ctx][key];
        e.first++;
        e.second += dst->op == GGML_OP_SET_ROWS || dst->op == GGML_OP_CPY ? ggml_nbytes(dst->op == GGML_OP_CPY ? dst->src[0] : dst->src[1]) : ggml_nbytes(dst);
    }
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_MOE_CACHE_LOOKUP:
            ggml_cuda_op_moe_cache_lookup(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

// k15-fn: the host -> device uploads of n tensors (offset 0, whole size given) as one cudaMemcpyBatchAsync on the backend's stream, in
// stream order like n cudaMemcpyAsync (the scheduler's graph inputs of a split: 21 per Flash-Next verify graph, issued back to back with
// the GPU idle between them). Returns false (nothing queued) where the batch API is missing or refuses them: the caller copies one by one
static bool ggml_backend_cuda_set_tensors_batch_async(ggml_backend_t backend, ggml_tensor * const * tensors, const void * const * data,
        const size_t * sizes, int n) {
#if CUDART_VERSION >= 12080 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (n <= 0) {
        return true;
    }
    std::vector<void *> dsts(n);
    std::vector<const void *> srcs(n);
    std::vector<size_t> sz(n);
    for (int i = 0; i < n; ++i) {
        ggml_backend_buffer_t buf = tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
        if (buf == nullptr || buf->buft != ggml_backend_cuda_buffer_type(cuda_ctx->device)) {
            return false;
        }
        dsts[i] = tensors[i]->data;
        srcs[i] = data[i];
        sz[i]   = sizes[i];
    }
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.srcLocHint.type = cudaMemLocationTypeHost;
    attr.dstLocHint.type = cudaMemLocationTypeDevice;
    attr.dstLocHint.id   = cuda_ctx->device;
    attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx = 0;
    const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sz.data(), (size_t) n, &attr, &attr_idx, 1, cuda_ctx->stream());
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
#else
    GGML_UNUSED(backend); GGML_UNUSED(tensors); GGML_UNUSED(data); GGML_UNUSED(sizes); GGML_UNUSED(n);
    return false;
#endif
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

// The device-to-device copies of a copy-only instance (the scheduler copy queue) run on the stream of its compute instance.
// With device work (a copy kernel or a copy engine device copy) on a second stream of the process, every later kernel launch
// of the process was 0.3 to 0.5 us slower on compute capability 12.0 (the next kernel no longer starts before the previous one
// ends, PDL included), so generation stayed ~7% slower after the first prompt that used the copy queue; uploads alone do not
// do it. The copies keep the order of the copy queue: they wait for its earlier work, and its later work waits for them.
static cudaStream_t ggml_cuda_copy_peer_begin(ggml_backend_cuda_context * ctx) {
    ggml_backend_cuda_context * peer = ctx->copy_peer;
    if (peer == nullptr || ctx->computes) {
        return ctx->stream();
    }
    cudaStream_t ps = peer->stream(peer->device, 0);
#if !defined(GGML_USE_HIP) // the copy peer is never set on HIP
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ps, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return ctx->stream();
    }
#endif // !defined(GGML_USE_HIP)
    ggml_cuda_set_device(ctx->device);
    for (cudaEvent_t & ev : ctx->copy_peer_events) {
        if (ev == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        }
    }
    CUDA_CHECK(cudaEventRecord(ctx->copy_peer_events[0], ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(ps, ctx->copy_peer_events[0], 0));
    return ps;
}

static void ggml_cuda_copy_peer_end(ggml_backend_cuda_context * ctx, cudaStream_t stream) {
    if (stream == ctx->stream()) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(ctx->copy_peer_events[1], stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx->stream(), ctx->copy_peer_events[1], 0));
}

// evicts the cached CUDA graphs last used at or after t_us (ggml_time_us), e.g. the graphs of a computation that will not run again
// (their keys are node addresses of freed graphs); returns how many. Without it they age out after 10 s of later graph use only
static size_t ggml_backend_cuda_graph_evict_since(ggml_backend_t backend, int64_t t_us) {
    size_t n = 0;
#ifdef USE_CUDA_GRAPH
    if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
        return 0;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    for (auto it = ctx->cuda_graphs.begin(); it != ctx->cuda_graphs.end(); ) {
        if (it->second->last_used_time >= t_us) {
            it = ctx->cuda_graphs.erase(it);
            n++;
        } else {
            ++it;
        }
    }
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(t_us);
#endif
    return n;
}

static void ggml_backend_cuda_set_copy_peer(ggml_backend_t copy_backend, ggml_backend_t compute_backend) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(copy_backend);
    GGML_UNUSED(compute_backend);
#else
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_COPY_PEER");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || !ggml_backend_is_cuda(copy_backend) || !ggml_backend_is_cuda(compute_backend) || copy_backend == compute_backend) {
        return;
    }
    ggml_backend_cuda_context * ctx  = (ggml_backend_cuda_context *) copy_backend->context;
    ggml_backend_cuda_context * peer = (ggml_backend_cuda_context *) compute_backend->context;
    if (ctx->device == peer->device) {
        ctx->copy_peer = peer;
    }
#endif // defined(GGML_USE_HIP)
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        cudaStream_t stream = ggml_cuda_copy_peer_begin(cuda_ctx_src);
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
        ggml_cuda_copy_peer_end(cuda_ctx_src, stream);
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            // k15-fn: GGML_CUDA_GRAPH_WHY=N logs the first changed node of up to N re-captures (diagnostic)
            static int why_left = getenv("GGML_CUDA_GRAPH_WHY") ? atoi(getenv("GGML_CUDA_GRAPH_WHY")) : 0;
            if (!res && why_left > 0) {
                why_left--;
                const ggml_cuda_graph::node_properties & o = graph->node_props[i];
                const char * what = "other tensor field";
                if (o.node.data != prop.node.data) {
                    what = "data";
                } else if (memcmp(o.node.ne, prop.node.ne, sizeof(o.node.ne)) != 0) {
                    what = "ne";
                } else if (memcmp(o.node.op_params, prop.node.op_params, sizeof(o.node.op_params)) != 0) {
                    what = "op_params";
                } else if (memcmp(o.node_src_data_ptrs, prop.node_src_data_ptrs, sizeof(o.node_src_data_ptrs)) != 0) {
                    what = "src data";
                } else if (memcmp(o.node_src_ne, prop.node_src_ne, sizeof(o.node_src_ne)) != 0) {
                    what = "src ne";
                } else if (memcmp(o.node.src, prop.node.src, sizeof(o.node.src)) != 0) {
                    what = "src pointer";
                } else if (o.node.extra != prop.node.extra) {
                    what = "extra";
                } else if (o.node.buffer != prop.node.buffer) {
                    what = "buffer";
                }
                GGML_LOG_INFO("%s: CUDA graph key %p (%d nodes) changed at node %d '%s' (%s): %s\n", __func__, graph_key, cgraph->n_nodes, i,
                    cgraph->nodes[i]->name, ggml_op_desc(cgraph->nodes[i]), what);
            }
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    // k11: GGML_CUDA_GRAPH_UPDATE_FAULT=N (debug) makes every Nth update fail as if the device were out of memory
    static const int update_fault = getenv("GGML_CUDA_GRAPH_UPDATE_FAULT") ? atoi(getenv("GGML_CUDA_GRAPH_UPDATE_FAULT")) : 0;
    static int64_t n_update = 0;
    if (update_fault > 0 && stat == cudaSuccess && ++n_update % update_fault == 0) {
        stat = cudaErrorMemoryAllocation;
    }

    // k11 (GGML_CUDA_GRAPH_UPDATE_RETRY=0 restores the assert): an update can also fail with another error than an update failure, e.g.
    // out of memory on a card other processes filled (seen on shared B200s with master, k10 and k11 builds alike). Such an error is not
    // sticky: it is cleared and the graph is instantiated again, as for an update failure. A sticky error (an earlier kernel fault)
    // fails the instantiation below with its own message
    static const bool update_retry = getenv("GGML_CUDA_GRAPH_UPDATE_RETRY") == nullptr || atoi(getenv("GGML_CUDA_GRAPH_UPDATE_RETRY")) != 0;
    const bool update_error = stat != cudaSuccess && stat != cudaErrorGraphExecUpdateFailure && update_retry;
    if (update_error) {
        static int n_logged = 0;
        if (n_logged++ < 4) {
            GGML_LOG_WARN("%s: CUDA graph update failed with %s (%d): instantiating the graph again\n", __func__, cudaGetErrorString(stat), (int) stat);
        }
        stat = cudaErrorGraphExecUpdateFailure;
    }

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        if (update_error) {
            // k16-race: after another error than an update failure (seen after a launch stalled for minutes on a crowded GPU: the
            // update returned invalid argument and so did this destroy, which aborted) the exec is destroyed at most once and a
            // failed destroy is not fatal: the handle is dropped and a new exec instantiated (GGML_CUDA_GRAPH_UPDATE_RETRY=0: assert)
            const cudaError_t de = graph->instance != nullptr ? cudaGraphExecDestroy(graph->instance) : cudaSuccess;
            if (de != cudaSuccess) {
                GGML_LOG_WARN("%s: destroying the exec after the failed update gave %s: dropped, instantiating a new one\n", __func__, cudaGetErrorString(de));
                (void)cudaGetLastError();
            }
        } else {
            CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        }
        graph->instance = nullptr;
        cudaError_t inst = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
        // k11: out of memory may pass once other processes free some; a few tries before the error is fatal
        for (int k = 0; k < 20 && inst == cudaErrorMemoryAllocation && update_retry; ++k) {
            (void)cudaGetLastError();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            inst = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
        }
        CUDA_CHECK(inst);
    } else {
        if (stat != cudaSuccess) {
            GGML_LOG_ERROR("%s: CUDA graph update failed with %s (%d)\n", __func__, cudaGetErrorString(stat), (int) stat);
        }
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// k15-fn: RMS_NORM -> MUL -> ROPE with a multi-section, non-vision rope (ggml_cuda_op_rms_norm_mul_mrope_fused): f32, the rope
// reads the mul output, the norm is the full-shape mul operand, the mul output is contiguous (the norm kernel writes it there)
static bool ggml_cuda_should_fuse_rms_norm_mul_mrope(const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }
    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 || mul->src[0]->type != GGML_TYPE_F32 ||
        mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }
    if (rope->src[0] != mul || (mul->src[0] != rms_norm && mul->src[1] != rms_norm)) {
        return false;
    }
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    if (!ggml_are_same_shape(rms_norm, mul) || !ggml_can_repeat(mul_src, mul) || !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous_rows(rms_norm->src[0]) || !ggml_is_contiguous_rows(mul_src)) {
        return false;
    }
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (!(mode & GGML_ROPE_TYPE_MROPE) || mode == GGML_ROPE_TYPE_VISION) {
        return false;
    }
    const int n_dims = ((const int32_t *) rope->op_params)[1];
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    const int64_t ne0 = mul->ne[0];
    if (n_dims <= 0 || n_dims % 2 != 0 || n_offs < 0 || n_offs % 2 != 0 || ne0 % 2 != 0 || n_offs + n_dims > ne0) {
        return false;
    }
    // the norm grid is (rows, channels, samples): the rope row is (head, token, sequence), pos is indexed by the token
    if (!ggml_are_same_shape(rope, mul) || rope->src[1]->type != GGML_TYPE_I32 || rope->src[1]->ne[0] < 4*mul->ne[2] ||
        (rope->src[2] != nullptr && rope->src[2]->type != GGML_TYPE_F32)) {
        return false;
    }
    // the rope output rows must not overlap the rows the kernel still reads (only the in-place rope writes back to the mul output)
    const char * r0 = (const char *) rope->data;
    const char * m0 = (const char *) mul->data;
    return r0 == m0 || r0 + ggml_nbytes(rope) <= m0 || m0 + ggml_nbytes(mul) <= r0;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_UNUSED_VARS(cgraph, node_idx, selection, args);
    return false;
#else
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
#endif // defined(GGML_USE_HIP) && HIP_VERSION < 60300000
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// Size of one ggml_cuda_q8_1_cache slot for this graph, 0 if no activation is read by more than one
// mul_mat_vec_q candidate and none is an RMS_NORM + MUL output (then the cache stays off and nothing changes).
// norm_targets: the producer tensors (RMS_NORM + MUL outputs, elementwise chain ends, DSV4_HC_PRE) whose q8_1 copy their
// kernel writes. They are matched to their readers by tensor identity, through reshapes, not by address: the allocator
// reuses the memory of dead tensors, so one address holds different tensors over a graph.
static size_t ggml_cuda_q8_1_cache_slot_size(const ggml_cgraph * cgraph, std::unordered_set<const void *> & norm_targets,
        std::unordered_map<const void *, int64_t> & norm_target_ne10) {
    norm_targets.clear();
    norm_target_ne10.clear();
    // GGML_CUDA_DISABLE_FUSION=1 also turns it off, so that it restores the unfused path as a whole
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_Q8_1_CACHE") || ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    if (disabled) {
        return 0;
    }

    auto is_candidate = [](const ggml_tensor * node) {
        if ((node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src1 = node->src[1];
        if (!ggml_is_quantized(src0->type) || src1->type != GGML_TYPE_F32 || src1->data == nullptr) {
            return false;
        }
        // mul_mat_vec_q batch sizes
        const int64_t ncols = node->op == GGML_OP_MUL_MAT ? src1->ne[1] : src1->ne[2];
        return ncols <= MMVQ_MAX_BATCH_SIZE && ggml_nrows(src1) <= 8*MMVQ_MAX_BATCH_SIZE;
    };

    std::unordered_map<const void *, int> n_readers;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (is_candidate(cgraph->nodes[i])) {
            n_readers[cgraph->nodes[i]->src[1]->data]++;
        }
    }

    // RMS_NORM + MUL outputs that a candidate reads as a whole contiguous activation: the fused norm kernel writes their
    // q8_1 copy directly (GGML_CUDA_DISABLE_NORM_Q8_1_FUSION=1 turns this off)
    static const bool norm_q8_1_disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_NORM_Q8_1_FUSION");
    // the same for the outputs of elementwise chains and of DSV4_HC_PRE (GGML_CUDA_DISABLE_PRODUCER_Q8_1=1 turns this off)
    static const bool producer_q8_1_disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_PRODUCER_Q8_1");
    std::unordered_set<const ggml_tensor *> norm_outputs;
    if (!norm_q8_1_disabled) {
        for (int i = 0; i + 1 < cgraph->n_nodes; ++i) {
            const ggml_tensor * norm = cgraph->nodes[i];
            const ggml_tensor * mul  = cgraph->nodes[i + 1];
            if (norm->op == GGML_OP_RMS_NORM && mul->op == GGML_OP_MUL && (mul->src[0] == norm || mul->src[1] == norm) &&
                mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && mul->data != nullptr) {
                norm_outputs.insert(mul);
            }
        }
    }
    // the x / scale output of the fused amax rescale (ggml_cuda_try_fuse_amax_rescale), GGML_CUDA_DISABLE_AMAX_Q8_1=1 turns this off
    static const bool amax_q8_1_disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_AMAX_Q8_1");
    if (!producer_q8_1_disabled && !amax_q8_1_disabled) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_DIV && node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && node->data != nullptr &&
                (node->flags & GGML_TENSOR_FLAG_COMPUTE) && node->src[1] != nullptr && node->src[1]->op == GGML_OP_UNARY &&
                ggml_get_unary_op(node->src[1]) == GGML_UNARY_OP_EXP) {
                norm_outputs.insert(node);
            }
        }
    }
    if (!producer_q8_1_disabled) {
        // k15: the LFM2 short conv * c product (shortconv-fused.cu writes its q8_1 copy)
        for (int i = 0; i + 1 < cgraph->n_nodes; ++i) {
            const ggml_tensor * conv = cgraph->nodes[i];
            const ggml_tensor * mul  = cgraph->nodes[i + 1];
            if (conv->op == GGML_OP_SSM_CONV && mul->op == GGML_OP_MUL && (mul->src[0] == conv || mul->src[1] == conv) &&
                mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && mul->data != nullptr) {
                norm_outputs.insert(mul);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if ((ggml_cuda_ew_chain_op_code(node) >= 0 || node->op == GGML_OP_DSV4_HC_PRE) && node->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(node) && node->data != nullptr && (node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                norm_outputs.insert(node);
            }
        }
    }
    // k15-mm2: gated outputs (GLU, and the MUL of a fused SILU / SIGMOID / SOFTPLUS + MUL pair), unary.cu writes their q8_1 copy, also
    // for a reader whose rows are a multiple of MATRIX_ROW_PADDING other than theirs (GGML_CUDA_GATED_Q8_1=0 turns this off)
    // (GGML_CUDA_DISABLE_GLU_Q8_1=1, the switch of k15-fn 414e7c2d4a5 that this replaces, is an alias)
    static const bool gated_q8_1_disabled = [] {
        const char * e = getenv("GGML_CUDA_GATED_Q8_1");
        const char * d = getenv("GGML_CUDA_DISABLE_GLU_Q8_1");
        return (e != nullptr && atoi(e) == 0) || (d != nullptr && atoi(d) != 0);
    }();
    std::unordered_set<const ggml_tensor *> flat_outputs;
    if (!producer_q8_1_disabled && !gated_q8_1_disabled) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || node->data == nullptr || !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                continue;
            }
            bool gated = node->op == GGML_OP_GLU;
            if (node->op == GGML_OP_MUL && i > 0) {
                const ggml_tensor * u = cgraph->nodes[i - 1];
                gated = u->op == GGML_OP_UNARY && (node->src[0] == u || node->src[1] == u) &&
                    (ggml_get_unary_op(u) == GGML_UNARY_OP_SILU || ggml_get_unary_op(u) == GGML_UNARY_OP_SIGMOID ||
                     ggml_get_unary_op(u) == GGML_UNARY_OP_SOFTPLUS);
            }
            if (gated) {
                norm_outputs.insert(node);
                flat_outputs.insert(node);
            }
        }
    }

    size_t slot_size = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!is_candidate(node)) {
            continue;
        }
        const ggml_tensor * src1 = node->src[1];
        // the producer behind whole-tensor views (reshapes) of it
        const ggml_tensor * prod = src1;
        while (prod->view_src != nullptr && prod->view_offs == 0 && ggml_nbytes(prod) == ggml_nbytes(prod->view_src)) {
            prod = prod->view_src;
        }
        const bool same_rows = ggml_cuda_q8_1_cache::same_copy_rows(src1->ne[0], prod->ne[0]);
        const bool flat_rows = !same_rows && flat_outputs.count(prod) && src1->ne[0] % MATRIX_ROW_PADDING == 0 &&
            !ggml_cuda_q8_1_cache::reshape_disabled();
        const bool norm_target = norm_outputs.count(prod) && ggml_is_contiguous(src1) && ggml_is_contiguous(prod) && (same_rows || flat_rows);
        if (norm_target) {
            norm_targets.insert(prod);
            if (flat_rows) {
                norm_target_ne10[prod] = src1->ne[0];
            }
        } else if (n_readers[src1->data] < 2) {
            continue;
        }
        const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
        slot_size = std::max(slot_size, (size_t) (ggml_nrows(src1)*ne10_padded/QK8_1*sizeof(block_q8_1)));
    }
    return GGML_PAD(slot_size, 256);
}

// the nodes [i, i + n) are done: drop the cached q8_1 activations they may have overwritten
static void ggml_cuda_q8_1_cache_node_done(ggml_cuda_q8_1_cache & cache, const ggml_cgraph * cgraph, int i, int n) {
    if (!cache.active()) {
        return;
    }
    for (int j = i; j < i + n; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_OPT_STEP_ADAMW || node->op == GGML_OP_OPT_STEP_SGD) {
            cache.invalidate_all(); // update src[0] in place
        }
        if (node->op == GGML_OP_MOE_CACHE_LOOKUP) {
            // updates its sources in place: the slot maps, the eviction state and the expert banks
            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                cache.invalidate(node->src[s]->data, ggml_nbytes(node->src[s]));
            }
        }
        if (node->data != nullptr) {
            cache.invalidate(node->data, ggml_nbytes(node));
        }
    }
}

// elementwise fusions of small decode-time op chains; each one has its own kill switch (ggml_cuda_env_flag)

// true when [a, a + size(a)) and [b, b + size(b)) overlap
static bool ggml_cuda_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    const char * a1 = a0 + ggml_nbytes(a);
    const char * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// an elementwise kernel reading src[i] and writing dst[i] in the same thread may run in place, but not on a shifted alias
static bool ggml_cuda_elementwise_alias_ok(const ggml_tensor * dst, const ggml_tensor * src) {
    if (!ggml_cuda_tensors_overlap(dst, src)) {
        return true;
    }
    return dst->data == src->data && ggml_is_contiguous(dst) && ggml_is_contiguous(src) && ggml_nbytes(dst) == ggml_nbytes(src);
}

// SCALE / CLAMP / SQR / SQRT / SIN / COS / LOG / UNARY chain, each node f32, contiguous, same shape, and read only by the next one.
// Returns the number of extra nodes covered (0 = no chain).
static int ggml_cuda_try_fuse_ew_chain(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_EW_CHAIN");
    if (disabled) {
        return 0;
    }

    const auto chain_node_ok = [&](const ggml_tensor * t) {
        return ggml_cuda_ew_chain_op_code(t) >= 0 && t->type == GGML_TYPE_F32 && t->src[0] != nullptr &&
               t->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_is_contiguous(t->src[0]) &&
               ggml_are_same_shape(t, t->src[0]) && (t->flags & GGML_TENSOR_FLAG_COMPUTE);
    };

    ggml_tensor * first = cgraph->nodes[i];
    if (!chain_node_ok(first)) {
        return 0;
    }

    int n = 1;
    while (n < GGML_CUDA_EW_CHAIN_MAX && i + n < cgraph->n_nodes) {
        ggml_tensor * prev = cgraph->nodes[i + n - 1];
        ggml_tensor * next = cgraph->nodes[i + n];
        if (!chain_node_ok(next) || next->src[0] != prev || !ggml_node_has_n_uses(cgraph, i + n - 1, 1)) {
            break;
        }
        // leave a gating activation that starts a top-k MoE fusion to that fusion
        if (next->op == GGML_OP_UNARY) {
            ggml_cuda_topk_moe_args args;
            if (ggml_cuda_topk_moe_fusion(cgraph, i + n, args)) {
                break;
            }
        }
        n++;
    }
    if (n < 2) {
        return 0;
    }

    ggml_tensor * last = cgraph->nodes[i + n - 1];
    if (!ggml_cuda_elementwise_alias_ok(last, first->src[0])) {
        return 0;
    }

    ggml_cuda_ew_chain chain{};
    chain.n_ops = n;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        chain.op[j] = ggml_cuda_ew_chain_op_code(t);
        chain.p0[j] = 0.0f;
        chain.p1[j] = 0.0f;
        if (t->op == GGML_OP_SCALE || t->op == GGML_OP_CLAMP) {
            memcpy(&chain.p0[j], (const float *) t->op_params + 0, sizeof(float));
            memcpy(&chain.p1[j], (const float *) t->op_params + 1, sizeof(float));
        }
    }
    // a mul_mat_vec_q reads the chain output: write its q8_1 copy in the same kernel
    void * q8_1_out = nullptr;
    ggml_cuda_q8_1_cache & cache = cuda_ctx->q8_1_cache;
    // k15-mm2: also several rows whose length is a multiple of 32 but not of the q8_1 row padding (the hyper-connection up input at
    // an MTP verify, 320 x 3), each q8_1 row padded with zeros as quantize_q8_1 pads it (GGML_CUDA_EW_Q8_1_ROWS=0: those rows not)
    static const bool rows_off = [] { const char * e = getenv("GGML_CUDA_EW_Q8_1_ROWS"); return e != nullptr && atoi(e) == 0; }();
    if (cache.active() && cuda_ctx->curr_stream_no == 0 && cache.norm_targets.count(last) &&
        (ggml_nrows(last) == 1 || last->ne[0] % MATRIX_ROW_PADDING == 0 || (!rows_off && last->ne[0] % QK8_1 == 0))) {
        const int64_t ne10_padded = GGML_PAD(last->ne[0], MATRIX_ROW_PADDING);
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(last->ne[0], ggml_nrows(last), key);
        q8_1_out = cache.reserve(last->data, ggml_nbytes(last), key, ggml_nrows(last)*ne10_padded/QK8_1*sizeof(block_q8_1));
    }
    ggml_cuda_op_ew_chain(*cuda_ctx, first->src[0], last, chain, q8_1_out);
    return n - 1;
}

// ABS -> POOL_1D(MAX, kernel = stride = ne0, no padding) -> elementwise chain -> DIV(x, chain): build_amax_src1_f16 (the Flint
// MoE down-projection rescale) as one kernel that writes the chain output (the scale, read again after the matmul) and x / scale.
// Returns the number of extra nodes covered (0 = no fusion). GGML_CUDA_DISABLE_AMAX_RESCALE_FUSION=1 turns it off.
static int ggml_cuda_try_fuse_amax_rescale(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_AMAX_RESCALE_FUSION");
    if (disabled) {
        return 0;
    }
    const auto f32c = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && (t->flags & GGML_TENSOR_FLAG_COMPUTE) &&
               !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    ggml_tensor * ab = cgraph->nodes[i];
    if (ab->op != GGML_OP_UNARY || ggml_get_unary_op(ab) != GGML_UNARY_OP_ABS || !f32c(ab) || ab->src[0] == nullptr ||
        ab->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(ab->src[0]) || i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * x  = ab->src[0];
    ggml_tensor * pl = cgraph->nodes[i + 1];
    if (pl->op != GGML_OP_POOL_1D || pl->src[0] != ab || !f32c(pl) || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return 0;
    }
    const int32_t * pp = (const int32_t *) pl->op_params; // op, k0, s0, p0
    if (pp[0] != GGML_OP_POOL_MAX || pp[1] != x->ne[0] || pp[2] != x->ne[0] || pp[3] != 0 || pl->ne[0] != 1 ||
        ggml_nelements(pl) != ggml_nrows(x)) {
        return 0;
    }
    int n = 0;
    ggml_tensor * prev = pl;
    while (n < GGML_CUDA_EW_CHAIN_MAX && i + 2 + n < cgraph->n_nodes) {
        ggml_tensor * t = cgraph->nodes[i + 2 + n];
        if (ggml_cuda_ew_chain_op_code(t) < 0 || t->src[0] != prev || !f32c(t) || !ggml_are_same_shape(t, prev) ||
            !ggml_node_has_n_uses(cgraph, i + 1 + n, 1)) {
            break;
        }
        prev = t;
        n++;
    }
    const int id = i + 2 + n;
    if (n == 0 || id >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * dv = cgraph->nodes[id];
    if (dv->op != GGML_OP_DIV || dv->src[0] != x || dv->src[1] != prev || dv->type != GGML_TYPE_F32 || !ggml_is_contiguous(dv) ||
        !ggml_are_same_shape(dv, x) || !(dv->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_cuda_elementwise_alias_ok(dv, x)) {
        return 0;
    }
    ggml_cuda_ew_chain chain{};
    chain.n_ops = n;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + 2 + j];
        chain.op[j] = ggml_cuda_ew_chain_op_code(t);
        chain.p0[j] = 0.0f;
        chain.p1[j] = 0.0f;
        if (t->op == GGML_OP_SCALE || t->op == GGML_OP_CLAMP) {
            memcpy(&chain.p0[j], (const float *) t->op_params + 0, sizeof(float));
            memcpy(&chain.p1[j], (const float *) t->op_params + 1, sizeof(float));
        }
    }
    // a mul_mat_vec_q reads x / scale (the down projection): write its q8_1 copy in the same kernel (see the q8_1 cache planner)
    void * q8_1_out = nullptr;
    ggml_cuda_q8_1_cache & cache = cuda_ctx->q8_1_cache;
    if (cache.active() && cuda_ctx->curr_stream_no == 0 && cache.norm_targets.count(dv) &&
        (ggml_nrows(dv) == 1 || dv->ne[0] % MATRIX_ROW_PADDING == 0)) {
        const int64_t ne10_padded = GGML_PAD(dv->ne[0], MATRIX_ROW_PADDING);
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(dv->ne[0], ggml_nrows(dv), key);
        q8_1_out = cache.reserve(dv->data, ggml_nbytes(dv), key, ggml_nrows(dv)*ne10_padded/QK8_1*sizeof(block_q8_1));
    }
    ggml_cuda_op_amax_rescale(*cuda_ctx, x, prev, dv, chain, q8_1_out);
    return id - i;
}

// k15-mm2: the next hyper-connection mix's grouped norm (RMS_NORM -> MUL, a q8_1 producer) reads the DSV4_HC_POST output of the
// chain nodes [i, i + n_chain) + nodes[ip]: computed in one norm kernel (see below). Returns the number of extra nodes covered.
    // k15-mm2: the next hyper-connection mix's grouped norm (RMS_NORM -> MUL, a q8_1 producer) reads this DSV4_HC_POST output: the
    // norm kernel computes the post itself (stored to the post output as before) and writes the norm, its q8_1 copy, one launch
    // (GGML_CUDA_HC_POST_NORM=0: separate launches). Same expressions in the same order, so every output keeps its bytes.
static int ggml_cuda_try_fuse_hc_post_norm(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i, int n_chain, int ip) {
    ggml_tensor * hc_post = cgraph->nodes[ip];
    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * post_src = cgraph->nodes[i]->src[0];
    const int64_t hc       = residual->ne[1];
    const int64_t n_tokens = x->ne[1];
    if (hc_post->src[3] != nullptr || post_src == nullptr || post_src->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
            residual->type != GGML_TYPE_F32 || hc_post->type != GGML_TYPE_F32 || ggml_nelements(post_src) != hc*n_tokens ||
            ggml_cuda_tensors_overlap(hc_post, post_src)) {
        return 0;
    }
    {
        // opt-in (GGML_CUDA_HC_POST_NORM=1), as on k15 1923a00666c: it raised the rate of the pinned Flash-Next graphs race
        // (integration gate, 6/35 vs 0/14) until that is fixed; decode cost of leaving it off +0.4 to 0.5% (inside the noise)
        static const bool off = [] { const char * e = getenv("GGML_CUDA_HC_POST_NORM"); return e == nullptr || atoi(e) == 0; }();
        int irn = ip + 1;
        while (irn < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[irn])) {
            irn++;
        }
        bool ok = !off && irn + 1 < cgraph->n_nodes && n_chain <= GGML_CUDA_HC_POST_NORM_MAX_OPS && cuda_ctx->curr_stream_no == 0 &&
            cuda_ctx->q8_1_cache.active();
        ggml_tensor * rn  = ok ? cgraph->nodes[irn]     : nullptr;
        ggml_tensor * mul = ok ? cgraph->nodes[irn + 1] : nullptr;
        ok = ok && rn->op == GGML_OP_RMS_NORM && rn->src[0] == hc_post && (rn->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            mul->op == GGML_OP_MUL && (mul->src[0] == rn || mul->src[1] == rn) && (mul->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            ggml_node_has_n_uses(cgraph, irn, 1) && !(rn->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_is_contiguous(hc_post) &&
            ggml_is_contiguous(mul) && rn->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && cuda_ctx->q8_1_cache.norm_targets.count(mul) &&
            hc_post->ne[0] == x->ne[0] && hc_post->ne[1] == hc && hc_post->ne[2] == n_tokens && hc_post->ne[3] == 1 &&
            x->nb[0] == sizeof(float) && residual->nb[0] == sizeof(float) && ggml_is_contiguous(post_src);
        for (int j = 0; ok && j < n_chain; ++j) {
            const ggml_tensor * t = cgraph->nodes[i + j];
            ok = t->op == GGML_OP_SCALE || (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID);
        }
        // every block reads the whole block output row of its token (x) while others write the post output: no overlap; the residual
        // only element by element (in place is fine when it is the same layout)
        ok = ok && !ggml_cuda_tensors_overlap(hc_post, x) &&
            (!ggml_cuda_tensors_overlap(hc_post, residual) || (hc_post->data == residual->data && ggml_are_same_stride(hc_post, residual)));
        if (ok) {
            const int64_t ne10_padded = GGML_PAD(mul->ne[0], MATRIX_ROW_PADDING);
            int64_t key[ggml_cuda_q8_1_cache::n_key];
            ggml_cuda_q8_1_cache::key_contiguous(mul->ne[0], ggml_nrows(mul), key);
            void * q8_1_out = cuda_ctx->q8_1_cache.reserve(mul->data, ggml_nbytes(mul), key, ggml_nrows(mul)*ne10_padded/QK8_1*sizeof(block_q8_1));
            if (q8_1_out != nullptr) {
                ggml_cuda_hc_post_args hcp;
                hcp.bo       = (const float *) x->data;
                hcp.residual = (const float *) residual->data;
                hcp.post     = (const float *) post_src->data;
                hcp.sbo1     = x->nb[1] / sizeof(float);
                hcp.sr1      = residual->nb[1] / sizeof(float);
                hcp.sr2      = residual->nb[2] / sizeof(float);
                hcp.hc       = (int) hc;
                hcp.n_ops    = n_chain;
                for (int j = 0; j < n_chain; ++j) {
                    hcp.op[j] = cgraph->nodes[i + j]->op == GGML_OP_SCALE ? 0 : 1;
                    const ggml_tensor * t = cgraph->nodes[i + j];
                    hcp.p0[j] = 0.0f;
                    hcp.p1[j] = 0.0f;
                    if (t->op == GGML_OP_SCALE) {
                        memcpy(&hcp.p0[j], (const float *) t->op_params + 0, sizeof(float));
                        memcpy(&hcp.p1[j], (const float *) t->op_params + 1, sizeof(float));
                    }
                }
                ggml_cuda_op_rms_norm_fused(*cuda_ctx, rn, mul, q8_1_out, nullptr, &hcp);
                return irn + 1 - i;
            }
        }
    }
    return 0;
}

// elementwise chain -> DSV4_HC_POST (no comb) where the chain computes the post weights: one kernel, the chain applied to the
// weight each thread reads. Returns the number of extra nodes covered (0 = no fusion).
// GGML_CUDA_DISABLE_HC_POST_CHAIN_FUSION=1 turns it off.
static int ggml_cuda_try_fuse_hc_post_chain(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_HC_POST_CHAIN_FUSION");
    if (disabled) {
        return 0;
    }

    const auto chain_node_ok = [&](const ggml_tensor * t) {
        return ggml_cuda_ew_chain_op_code(t) >= 0 && t->type == GGML_TYPE_F32 && t->src[0] != nullptr &&
               t->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_is_contiguous(t->src[0]) &&
               ggml_are_same_shape(t, t->src[0]) && (t->flags & GGML_TENSOR_FLAG_COMPUTE);
    };

    if (!chain_node_ok(cgraph->nodes[i])) {
        return 0;
    }
    int n_chain = 1;
    while (n_chain < GGML_CUDA_EW_CHAIN_MAX && i + n_chain < cgraph->n_nodes && chain_node_ok(cgraph->nodes[i + n_chain]) &&
           cgraph->nodes[i + n_chain]->src[0] == cgraph->nodes[i + n_chain - 1]) {
        n_chain++;
    }
    const int ip = i + n_chain;
    if (ip >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * hc_post = cgraph->nodes[ip];
    if (hc_post->op != GGML_OP_DSV4_HC_POST || hc_post->src[3] != nullptr || hc_post->src[2] != cgraph->nodes[ip - 1] ||
        !(hc_post->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return 0;
    }
    for (int j = i; j < ip; ++j) {
        if (!ggml_node_has_n_uses(cgraph, j, 1) || (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }

    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * post_src = cgraph->nodes[i]->src[0];
    for (const ggml_tensor * t : { x, residual, post_src, (const ggml_tensor *) hc_post }) {
        if (t->type != GGML_TYPE_F32 || t->data == nullptr) {
            return 0;
        }
    }
    const int64_t hc       = residual->ne[1];
    const int64_t n_tokens = x->ne[1];
    if (post_src->ne[0] != hc || ggml_nelements(post_src) != hc*n_tokens) {
        return 0;
    }
    // the chain source is read by every thread while others write the output: they must not overlap
    {
        const char * a0 = (const char *) hc_post->data;
        const char * b0 = (const char *) post_src->data;
        if (a0 < b0 + ggml_nbytes(post_src) && b0 < a0 + ggml_nbytes(hc_post)) {
            return 0;
        }
    }

    ggml_cuda_ew_chain chain{};
    chain.n_ops = n_chain;
    for (int j = 0; j < n_chain; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        chain.op[j] = ggml_cuda_ew_chain_op_code(t);
        chain.p0[j] = 0.0f;
        chain.p1[j] = 0.0f;
        if (t->op == GGML_OP_SCALE || t->op == GGML_OP_CLAMP) {
            memcpy(&chain.p0[j], (const float *) t->op_params + 0, sizeof(float));
            memcpy(&chain.p1[j], (const float *) t->op_params + 1, sizeof(float));
        }
    }

    if (const int n = ggml_cuda_try_fuse_hc_post_norm(cuda_ctx, cgraph, i, n_chain, ip)) {
        return n;
    }

    ggml_cuda_op_hc_post_chain(*cuda_ctx, chain, post_src, hc_post);
    return n_chain;
}

// UNARY(gate) -> MUL(x, gate) where gate has one element per row of x: shared expert gates
// returns the number of extra nodes covered (0 = no fusion): 1, or 2 with a following ADD of the product (k15-fn,
// GGML_CUDA_DISABLE_GATE_MUL_ADD_FUSION=1 turns that part off)
static int ggml_cuda_try_fuse_unary_mul_gate(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_GATE_MUL_FUSION");
    if (disabled || i + 1 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * unary = cgraph->nodes[i];
    ggml_tensor * mul   = cgraph->nodes[i + 1];
    if (!ggml_cuda_unary_mul_gate_supported(unary) || mul->op != GGML_OP_MUL || !(mul->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (mul->src[0] == unary) == (mul->src[1] == unary) || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return false;
    }
    const ggml_tensor * g = unary->src[0];
    const ggml_tensor * x = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    if (g->type != GGML_TYPE_F32 || unary->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    if (g->ne[0] != 1 || x->ne[0] == 1 || !ggml_are_same_shape(unary, g) || !ggml_are_same_shape(x, mul) ||
            g->ne[1] != x->ne[1] || g->ne[2] != x->ne[2] || g->ne[3] != x->ne[3]) {
        return false;
    }
    if (!ggml_is_contiguous_1(g) || !ggml_is_contiguous_1(x) || x->nb[0] != sizeof(float) || !ggml_is_contiguous(mul)) {
        return false;
    }
    if (ggml_cuda_tensors_overlap(mul, g) || !ggml_cuda_elementwise_alias_ok(mul, x)) {
        return false;
    }
    static const bool add_disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_GATE_MUL_ADD_FUSION");
    if (!add_disabled && i + 2 < cgraph->n_nodes) {
        ggml_tensor * add = cgraph->nodes[i + 2];
        const ggml_tensor * a = add->op == GGML_OP_ADD ? (add->src[0] == mul ? add->src[1] : add->src[1] == mul ? add->src[0] : nullptr) : nullptr;
        if (a != nullptr && a != mul && (add->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_node_has_n_uses(cgraph, i + 1, 1) &&
                a->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && ggml_are_same_shape(a, add) && ggml_are_same_shape(mul, add) &&
                ggml_is_contiguous(a) && ggml_is_contiguous(add) && !ggml_cuda_tensors_overlap(add, g) &&
                ggml_cuda_elementwise_alias_ok(add, x) && ggml_cuda_elementwise_alias_ok(add, a) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 2 })) {
            ggml_cuda_op_unary_mul_gate_add(*cuda_ctx, unary, mul, add);
            return 2;
        }
    }
    ggml_cuda_op_unary_mul_gate(*cuda_ctx, unary, mul);
    return 1;
}

// ADD(a, bias) -> UNARY -> MUL(., scale) with bias and scale one row each: the delta-net alpha gate
static bool ggml_cuda_try_fuse_add_unary_mul(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_ADD_UNARY_MUL_FUSION");
    if (disabled || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * add   = cgraph->nodes[i];
    ggml_tensor * unary = cgraph->nodes[i + 1];
    ggml_tensor * mul   = cgraph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || !ggml_cuda_unary_mul_gate_supported(unary) || mul->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_op ops[3] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };
    if (!ggml_can_fuse(cgraph, i, ops, 3) || unary->src[0] != add || (mul->src[0] == unary) == (mul->src[1] == unary)) {
        return false;
    }
    const ggml_tensor * a     = ggml_are_same_shape(add->src[0], add) ? add->src[0] : add->src[1];
    const ggml_tensor * bias  = a == add->src[0] ? add->src[1] : add->src[0];
    const ggml_tensor * scale = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    if (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32 || unary->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(a, add) || !ggml_is_contiguous(a) || !ggml_is_contiguous(mul) ||
            bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] || ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
        return false;
    }
    if (!ggml_cuda_elementwise_alias_ok(mul, a) || ggml_cuda_tensors_overlap(mul, bias) || ggml_cuda_tensors_overlap(mul, scale)) {
        return false;
    }
    ggml_cuda_op_add_unary_mul(*cuda_ctx, add, unary, mul);
    return true;
}

// try and fuse nodes and return the number of nodes to skip
// k11-decode: the experts of the MoE cache are read through a view of its expert bank, and that VIEW node sits between the gate and the up
// product ({MUL_MAT_ID, VIEW, MUL_MAT_ID, GLU}), so the gate + up + GLU fusion below never matched them (k10: three launches per expert
// FFN). The fused launch is the one the {MUL_MAT_ID, MUL_MAT_ID, GLU} pattern makes: mul_mat_vec_q with the gate in its fusion
// arguments, whose up and gate sums are the sums of the unfused products and whose GLU is the GLU op's expression, so the output is
// the bytes of the three launches (batch invariant verify included: the GLU node carries the flag). GGML_CUDA_MOE_CACHE_GLU_FUSE=1: on.
static int ggml_cuda_try_fuse_moe_cache_glu(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    // off by default: G4 Flash-Next (Studio argv / no drafter) and Qwen3.6 MTP / LFM2.5 at 8 GB moved within the A/A spread
    static const bool on = getenv("GGML_CUDA_MOE_CACHE_GLU_FUSE") != nullptr && atoi(getenv("GGML_CUDA_MOE_CACHE_GLU_FUSE")) != 0;
    if (!on || i + 3 >= cgraph->n_nodes || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return 0;
    }
    ggml_tensor * a   = cgraph->nodes[i];
    ggml_tensor * v   = cgraph->nodes[i + 1];
    ggml_tensor * b   = cgraph->nodes[i + 2];
    ggml_tensor * glu = cgraph->nodes[i + 3];
    if (a->op != GGML_OP_MUL_MAT_ID || v->op != GGML_OP_VIEW || b->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU || b->src[0] != v) {
        return 0;
    }
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    if (glu->src[0] == a && glu->src[1] == b) {
        gate = a;
        up   = b;
    } else if (glu->src[0] == b && glu->src[1] == a) {
        gate = b;
        up   = a;
    } else {
        return 0;
    }
    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu) || !ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
        return 0;
    }
    const int     idx[3]  = { i, i + 2, i + 3 };
    const ggml_op ops[3]  = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    const int     outs[1] = { i + 3 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx, 3, ops, outs, 1) || !ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, outs, 1)) {
        return 0;
    }
    ggml_cuda_mm_fusion_args_host fusion_data{};
    fusion_data.gate      = gate->src[0];
    fusion_data.glu_op    = ggml_get_glu_op(glu);
    fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);
    ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], glu, &fusion_data);
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: MoE cache experts: gate + up + GLU in one launch (%lld tokens; GGML_CUDA_MOE_CACHE_GLU_FUSE=1)\n", __func__,
                (long long) up->ne[2]);
    }
    return 3;
}

// k15: ADD -> RMS_NORM -> MUL where a mul_mat_vec_q reads the MUL output (the q8_1 copy of ggml_cuda_q8_1_cache): the residual add
// is computed by the norm kernel (rms_norm_mul_q8_1_f32 with ra / rb), its output still written for its other users. LFM2.5 decode:
// two residual adds per layer, 80 launches per token. Bit-identical to the unfused ops (same float add, same reduction order).
// GGML_CUDA_ADD_RMS_FUSE=0: off
static int ggml_cuda_try_fuse_add_rms_norm(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_ADD_RMS_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!on || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * add = cgraph->nodes[i];
    ggml_tensor * rms = cgraph->nodes[i + 1];
    ggml_tensor * mul = cgraph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rms->src[0] != add ||
            (mul->src[0] != rms && mul->src[1] != rms)) {
        return 0;
    }
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(a, add) || !ggml_are_same_shape(b, add) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(add) ||
            !ggml_is_contiguous(mul) || !ggml_are_same_shape(rms, mul) ||
            !ggml_cuda_is_aligned(a, 16) || !ggml_cuda_is_aligned(b, 16) || !ggml_cuda_is_aligned(add, 16)) {
        return 0;
    }
    // in place (a or b at add's address) is fine: each element is read and written by the same thread and the norm pass reads after
    // a barrier; any other overlap is not
    const auto overlaps = [add](const ggml_tensor * t) {
        const char * lo = (const char *) t->data;
        const char * al = (const char *) add->data;
        return t->data != add->data && lo < al + ggml_nbytes(add) && al < lo + ggml_nbytes(t);
    };
    if (overlaps(a) || overlaps(b)) {
        return 0;
    }
    const ggml_tensor * mul_other = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (mul_other->type != GGML_TYPE_F32 || !ggml_can_repeat(mul_other, mul)) {
        return 0;
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 })) {
        return 0;
    }
    ggml_cuda_q8_1_cache & cache = cuda_ctx->q8_1_cache;
    if (!cache.active() || cuda_ctx->curr_stream_no != 0 || !cache.norm_targets.count(mul)) {
        return 0;
    }
    const int64_t ne10_padded = GGML_PAD(mul->ne[0], MATRIX_ROW_PADDING);
    int64_t key[ggml_cuda_q8_1_cache::n_key];
    ggml_cuda_q8_1_cache::key_contiguous(mul->ne[0], ggml_nrows(mul), key);
    void * q8_1_out = cache.reserve(mul->data, ggml_nbytes(mul), key, ggml_nrows(mul)*ne10_padded/QK8_1*sizeof(block_q8_1));
    if (q8_1_out == nullptr) {
        return 0;
    }
    ggml_cuda_op_rms_norm_fused(*cuda_ctx, rms, mul, q8_1_out, add);
    return 2;
}

// gate + up + GLU (MUL_MAT, single column) whose GLU output the allocator placed over the shared activation src1, which the
// unfused GLU may overwrite because both products have read it by then. ggml_cuda_check_fusion_memory_ranges refuses that, but
// the mul_mat_vec_q fusion reads only the q8_1 copy of src1 (quantized, or taken from the q8_1 cache, before the kernel that
// writes the output), so the alias is harmless there. Any other overlap still refuses. Shared experts and dense FFNs of MoE
// models hit this (e.g. Flint's MoE layers). GGML_CUDA_DISABLE_GLU_SRC1_ALIAS_FUSION=1 turns it off.
static bool ggml_cuda_glu_mmvq_src1_alias_ok(const ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_GLU_SRC1_ALIAS_FUSION");
    if (disabled || i + 2 >= cgraph->n_nodes ||
        !ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { i + 2 })) {
        return false;
    }
    const ggml_tensor * glu  = cgraph->nodes[i + 2];
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!((gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]))) {
        return false;
    }
    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu) || !ggml_is_quantized(up->src[0]->type) || !ggml_cuda_should_fuse_mul_mat_vec_q(up) ||
        ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
        return false;
    }
    const ggml_tensor * src1 = up->src[1];
    for (int j = i; j < i + 3; ++j) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = cgraph->nodes[j]->src[s];
            if (src == nullptr || src == src1 || src == gate || src == up || !ggml_cuda_tensors_overlap(glu, src)) {
                continue;
            }
            return false;
        }
    }
    return true;
}

// k15-mm: multi-matrix mul_mat_vec_q. Dense quantized products of one type and K over the same activation (Q/K/V of attention, the
// qkv + gate projections of a gated delta net layer, the shared expert's gate + up at several columns, ...) run as one launch that
// quantizes the activation once and gives every matrix its own blocks (ggml_cuda_mul_mat_vec_q_multi): each output gets the bytes of
// its own launch. The node order and the allocation are not changed: a later product of the group is computed early, at the first
// one, only when no node in between touches the memory of its output or writes the activation; it is then skipped at its own
// position (ggml_backend_cuda_context::mmvq_done_early). GGML_CUDA_DISABLE_MMVQ_MULTI=1 turns it off.
static bool ggml_cuda_mmvq_multi_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_MMVQ_MULTI");
    return disabled;
}

static constexpr int GGML_CUDA_MMVQ_MULTI_LOOKAHEAD = 64; // nodes searched after the first product of a group

// a dense product of a quantized weight (a leaf) and a contiguous f32 activation
static bool ggml_cuda_mmvq_multi_candidate(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || node->view_src != nullptr || node->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    return ggml_is_quantized(src0->type) && src0->op == GGML_OP_NONE && src0->view_src == nullptr && ggml_is_contiguous(src0) &&
        ggml_is_matrix(src0) && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(node);
}

// a and b can share one multi-matrix launch
static bool ggml_cuda_mmvq_multi_pair(const ggml_tensor * a, const ggml_tensor * b) {
    return a->src[1] == b->src[1] && a->src[0]->type == b->src[0]->type && a->src[0]->ne[0] == b->src[0]->ne[0] &&
        a->src[0]->nb[1] == b->src[0]->nb[1] && (a->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) == (b->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT);
}

// ops that fuse with the single column product they read directly (bias, scale, GLU)
static bool ggml_cuda_mmvq_multi_partner_op(const ggml_op op) {
    return op == GGML_OP_ADD || op == GGML_OP_ADD_ID || op == GGML_OP_MUL || op == GGML_OP_GLU;
}

// node j has a fusion of its own with one of the next two nodes (single column only, see ggml_cuda_should_fuse_mul_mat_vec_q)
static bool ggml_cuda_mmvq_multi_has_partner(const ggml_cgraph * cgraph, const int j) {
    const ggml_tensor * t = cgraph->nodes[j];
    if (t->ne[1] != 1) {
        return false;
    }
    for (int k = j + 1; k <= j + 2 && k < cgraph->n_nodes; ++k) {
        const ggml_tensor * p = cgraph->nodes[k];
        if (!ggml_cuda_mmvq_multi_partner_op(p->op)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (p->src[s] == t) {
                return true;
            }
        }
    }
    return false;
}

// node i is a product that ggml_cuda_mul_mat sends to ggml_cuda_mul_mat_vec_q (same dispatch, see there) and that has a multi kernel
static bool ggml_cuda_mmvq_multi_runtime_ok(const ggml_backend_cuda_context * ctx, const ggml_tensor * node) {
    if (!ggml_cuda_mmvq_multi_candidate(node) || ggml_cuda_op_mul_mat_use_fwht(node)) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src0->buffer == nullptr || !ggml_backend_buft_is_cuda(src0->buffer->buft) ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE || src1->data == nullptr || node->data == nullptr) {
        return false;
    }
    const int     cc        = ggml_cuda_info().devices[ctx->device].cc;
    const int     warp_size = ggml_cuda_info().devices[ctx->device].warp_size;
    const int64_t ne11      = src1->ne[1];
    const bool    exact     = (node->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) != 0;
    if (ne11 > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }
    if (exact && ne11 > 1) {
        if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, 1) ||
                ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, 1, false) || !ggml_cuda_should_use_mmvq(src0->type, cc, 1)) {
            return false;
        }
    } else if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11) ||
            ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, false) || !ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        return false;
    }
    const bool ok = ggml_cuda_mul_mat_vec_q_multi_ok(src0->type, src0->ne[0], ne11, exact, cc);
    if (ok && exact && ne11 > 1) {
        ggml_cuda_verify_mmq_log(node, cc, "multi-matrix batched mmvq (not VERIFY_MMQ)");
    }
    return ok;
}

static bool ggml_cuda_mmvq_multi_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->data == nullptr || b->data == nullptr) {
        return false;
    }
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// node i and the products after it that share its activation: one launch. Returns -1 if not taken, else the number of nodes right
// after i that the launch covered (the others are in mmvq_done_early).
// k15-mm2 QSA maskless. qwen4exp builds a dense selection mask per QSA layer (llama_model_qwen4exp::graph::build_qsa_sel):
//   sel = add(view(set_rows(repeat(fill(-inf)), repeat(fill(0)), idx)), kq_mask)
// and the sparse flash attention gather compacts its finite entries into index lists. The lists are the selected cells
// (idx < n_kv, dead slots point at dump rows >= n_kv) where kq_mask is finite, so launch_fattn builds them from idx and kq_mask
// directly and the mask nodes are not run; a kernel that does not gather rebuilds the dense mask in one launch instead
static int32_t ggml_cuda_graph_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos) ||
        cgraph->visited_hash_set.keys[pos] != t) {
        return -1;
    }
    return cgraph->use_counts[pos];
}

static bool ggml_cuda_qsa_private(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    return t != nullptr && (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_cuda_graph_use_count(cgraph, t) == 1;
}

// t is (reshapes of) repeat(fill(c)), F16, every node used once: collect the nodes
static bool ggml_cuda_qsa_fill_chain(const ggml_cgraph * cgraph, const ggml_tensor * t, const bool neg_inf,
        std::vector<const ggml_tensor *> & nodes) {
    while (t != nullptr && t->op == GGML_OP_RESHAPE) {
        if (!ggml_cuda_qsa_private(cgraph, t)) {
            return false;
        }
        nodes.push_back(t);
        t = t->src[0];
    }
    if (t == nullptr || t->op != GGML_OP_REPEAT || t->type != GGML_TYPE_F16 || !ggml_cuda_qsa_private(cgraph, t)) {
        return false;
    }
    nodes.push_back(t);
    const ggml_tensor * f = t->src[0];
    if (f == nullptr || f->op != GGML_OP_FILL || f->type != GGML_TYPE_F16 || !ggml_cuda_qsa_private(cgraph, f)) {
        return false;
    }
    const float c = ggml_get_op_params_f32(f, 0);
    if (neg_inf ? !(std::isinf(c) && c < 0.0f) : c != 0.0f) {
        return false;
    }
    nodes.push_back(f);
    return true;
}

#define GGML_CUDA_QSA_MASKLESS_MAX_TOKENS 16

static void ggml_cuda_qsa_maskless_plan(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    cuda_ctx->qsa_maskless.clear();
    cuda_ctx->qsa_skip.clear();
    if (ggml_cuda_qsa_maskless_mode() == 0) {
        return;
    }
    std::vector<const ggml_tensor *> nodes;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * fa = cgraph->nodes[i];
        if (fa->op != GGML_OP_FLASH_ATTN_EXT || (fa->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) || ggml_get_op_params_i32(fa, 4) <= 0) {
            continue;
        }
        const ggml_tensor * add = fa->src[3];
        if (add == nullptr || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F16 || !ggml_cuda_qsa_private(cgraph, add)) {
            continue;
        }
        const ggml_tensor * view = add->src[0];
        const ggml_tensor * kq   = add->src[1];
        if (view == nullptr || kq == nullptr || view->op != GGML_OP_VIEW || kq->type != GGML_TYPE_F16 ||
            !ggml_cuda_qsa_private(cgraph, view) || !ggml_is_contiguous(add) || kq->ne[2] != 1 || kq->nb[0] != sizeof(half)) {
            continue;
        }
        bool same = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            same = same && view->ne[d] == kq->ne[d] && add->ne[d] == kq->ne[d] && add->nb[d] == kq->nb[d];
        }
        const ggml_tensor * sr = view->src[0];
        if (!same || sr == nullptr || sr->op != GGML_OP_SET_ROWS || sr->type != GGML_TYPE_F16 || view->view_offs != sr->view_offs ||
            !ggml_cuda_qsa_private(cgraph, sr)) {
            continue;
        }
        // set_rows dst [1, n_kv + n_sel, n_tokens]: the plane of token t = q + ne1*seq is mask row (q, seq)
        const int64_t n_kv   = kq->ne[0];
        const int64_t n_tok  = kq->ne[1]*kq->ne[3];
        // decode and verify windows: a dense kernel's pool copy of the mask stays small, and no mask scan (one stream)
        if (n_tok > GGML_CUDA_QSA_MASKLESS_MAX_TOKENS || kq->ne[3] != 1) {
            continue;
        }
        const ggml_tensor * idx = sr->src[1]; // set_rows: src[0] rows, src[1] indices, src[2] dst
        if (sr->ne[0] != 1 || sr->ne[1] < n_kv || sr->ne[2] != n_tok || sr->ne[3] != 1 || !ggml_is_contiguous(sr) ||
            view->nb[1] != sr->nb[2] || view->nb[3] != sr->nb[2]*kq->ne[1] ||
            idx == nullptr || idx->type != GGML_TYPE_I32 || idx->nb[0] != sizeof(int32_t) || idx->ne[1] != n_tok || idx->ne[2] != 1 ||
            idx->ne[3] != 1) {
            continue;
        }
        nodes.clear();
        if (!ggml_cuda_qsa_fill_chain(cgraph, sr->src[2], true, nodes) || !ggml_cuda_qsa_fill_chain(cgraph, sr->src[0], false, nodes)) {
            continue;
        }
        const ggml_tensor * zeros = sr->src[0];
        if (zeros->ne[0] != 1 || zeros->ne[1] != idx->ne[0] || zeros->ne[2] != n_tok) {
            continue;
        }
        nodes.push_back(sr);
        nodes.push_back(view);
        nodes.push_back(add);
        // idx is read after its last graph reader (set_rows) and kq maybe after its last one (the add): the nodes up to the
        // attention, and its output for kq, must not have been given their memory
        int i_sr = -1;
        for (int j = i - 1; j >= 0 && i_sr < 0; --j) {
            i_sr = cgraph->nodes[j] == sr ? j : -1;
        }
        if (i_sr < 0) {
            continue;
        }
        const auto hits = [](const char * r0, const char * r1, const ggml_tensor * t) {
            return r0 < (const char *) t->data + ggml_nbytes(t) && (const char *) t->data < r1;
        };
        const auto overlaps = [&](const ggml_tensor * a, const ggml_tensor * b) {
            return hits((const char *) a->data, (const char *) a->data + ggml_nbytes(a), b);
        };
        bool clobbered = false;
        for (int j = i_sr + 1; j < i && !clobbered; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(t) || std::find(nodes.begin(), nodes.end(), t) != nodes.end()) {
                continue;
            }
            clobbered = overlaps(t, idx) || overlaps(t, kq);
        }
        // the attention writes its F16 K / V copies behind its output before the list build reads idx, and its output after;
        // it reads kq throughout
        const char * fa0 = (const char *) fa->data;
        const char * fa1 = fa0 + ggml_cuda_flash_attn_ext_get_alloc_size(cuda_ctx->device, fa);
        const char * fo1 = fa0 + ggml_nbytes(fa);
        clobbered = clobbered || hits(fo1, fa1, idx) || hits(fa0, fa1, kq);
        if (clobbered) {
            continue;
        }
        cuda_ctx->qsa_maskless[fa] = { idx, kq, overlaps(add, idx) || overlaps(add, kq) };
        cuda_ctx->qsa_skip.insert(nodes.begin(), nodes.end());
    }
}

static int ggml_cuda_try_fuse_mmvq_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_mmvq_multi_disabled() || ggml_cuda_fusion_disabled() || !cuda_ctx->stream_context().concurrent_events.empty() ||
            cuda_ctx->curr_stream_no != 0) {
        return -1;
    }
    ggml_tensor * node = cgraph->nodes[i];
    // cheap checks first: most products have no partner reading the same activation
    if (!ggml_cuda_mmvq_multi_candidate(node) || node->src[1]->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return -1;
    }
    {
        bool any = false;
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + GGML_CUDA_MMVQ_MULTI_LOOKAHEAD && !any; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            any = t->op == GGML_OP_MUL_MAT && t->src[1] == node->src[1];
        }
        if (!any) {
            return -1;
        }
    }
    if (!ggml_cuda_mmvq_multi_runtime_ok(cuda_ctx, node) || ggml_cuda_mmvq_multi_has_partner(cgraph, i)) {
        return -1;
    }
    const ggml_tensor * act = node->src[1];
    ggml_tensor * dst[GGML_CUDA_MMVQ_MULTI_MAX] = { node };
    const ggml_tensor * src0[GGML_CUDA_MMVQ_MULTI_MAX] = { node->src[0] };
    int idx[GGML_CUDA_MMVQ_MULTI_MAX] = { i };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + GGML_CUDA_MMVQ_MULTI_LOOKAHEAD && n < GGML_CUDA_MMVQ_MULTI_MAX; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (cuda_ctx->mmvq_done_early.count(t)) {
            continue;
        }
        const bool member = ggml_cuda_mmvq_multi_pair(node, t) && ggml_cuda_mmvq_multi_runtime_ok(cuda_ctx, t) &&
            !ggml_cuda_mmvq_multi_has_partner(cgraph, j);
        if (member) {
            // computed at i instead of j: no node in between may touch its output's memory (the allocator may have placed short lived
            // tensors there), and no member's output may overlap another one
            bool ok = true;
            for (int k = i + 1; k < j && ok; ++k) {
                const ggml_tensor * u = cgraph->nodes[k];
                if (ggml_cuda_mmvq_multi_overlap(u, t)) {
                    ok = false;
                }
                for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
                    if (u->src[s] != nullptr && ggml_cuda_mmvq_multi_overlap(u->src[s], t)) {
                        ok = false;
                    }
                }
            }
            for (int m = 0; m < n && ok; ++m) {
                ok = !ggml_cuda_mmvq_multi_overlap(dst[m], t);
            }
            if (ok) {
                dst[n] = t;
                src0[n] = t->src[0];
                idx[n] = j;
                n++;
                continue;
            }
        }
        // a node that writes the activation (anything but a pure view over it) ends the search
        if (!ggml_cuda_is_view_or_noop(t) && ggml_cuda_mmvq_multi_overlap(t, act)) {
            break;
        }
    }
    if (n < 2) {
        return -1;
    }
    ggml_cuda_mul_mat_vec_q_multi(*cuda_ctx, n, src0, act, dst);
    // the members right after i are skipped by the caller, the others at their own position
    int run = 0;
    for (int m = 1; m < n; ++m) {
        if (idx[m] == i + run + 1) {
            run++;
        } else {
            cuda_ctx->mmvq_done_early.insert(dst[m]);
            cuda_ctx->q8_1_cache.invalidate(dst[m]->data, ggml_nbytes(dst[m]));
        }
    }
    return run;
}

// k15-fn (GGML_CUDA_GDN_STATE_IN_PLACE=0 off): the gather of one recurrent-state row (build_rs, a GET_ROWS of the cache rows by the
// rollback-group ids) whose only reader is a gated_delta_net of one sequence is skipped, and the kernel reads that row from the cache
// through the same ids: bitwise the same state, no 3 MiB copy per layer and step (Qwen3.8-Flash-Next: 36 GDN layers, 108 MiB per verify).
// Not for the mma kernel (it reads a contiguous state), nor when the gathered rows have another reader
static bool ggml_cuda_try_gdn_state_in_place(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = getenv("GGML_CUDA_GDN_STATE_IN_PLACE") == nullptr || atoi(getenv("GGML_CUDA_GDN_STATE_IN_PLACE")) != 0;
    if (!enabled || ggml_cuda_fusion_disabled()) {
        return false;
    }
    const ggml_tensor * gr   = cgraph->nodes[i];
    const ggml_tensor * rows = gr->src[0];
    const ggml_tensor * ids  = gr->src[1];
    if (gr->type != GGML_TYPE_F32 || rows->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        gr->ne[1] != 1 || gr->ne[2] != 1 || gr->ne[3] != 1 || ids->ne[0] != 1 || !ggml_is_contiguous(ids) ||
        rows->nb[0] != sizeof(float) || rows->ne[2] != 1 || rows->ne[3] != 1 || rows->nb[1] % sizeof(float) != 0 ||
        rows->data == nullptr || ids->data == nullptr) {
        return false;
    }
    const ggml_tensor * gdn = nullptr;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = n->src[k];
            if (s == nullptr || (s != gr && s->view_src != gr)) {
                continue;
            }
            if (ggml_cuda_is_view_or_noop(n) && n->view_src == gr) {
                continue; // a view of the rows: its readers read gr too and are checked themselves
            }
            if (n->op == GGML_OP_CPY && ggml_nelements(n->src[0]) == 0) {
                continue; // the copy of the extra states, empty with one state
            }
            if (n->op == GGML_OP_GATED_DELTA_NET && k == 5 && gdn == nullptr && s->view_offs == 0 && ggml_nelements(s) == gr->ne[0] &&
                ggml_is_contiguous(s) && n->src[2]->ne[3] == 1 && !ggml_cuda_gated_delta_net_may_use_mma(n)) {
                gdn = n;
                continue;
            }
            return false;
        }
    }
    if (gdn == nullptr) {
        return false;
    }
    // the kernel reads the row the gather would have copied: rows + ids[0] * nb[1]
    cuda_ctx->gdn_state_in_place[gdn] = { (const float *) rows->data, (int64_t) (rows->nb[1] / sizeof(float)), (const int32_t *) ids->data };
    return true;
}

// k15-fn (GGML_CUDA_DISABLE_CONV_STATE_FUSION=1 off): the conv-state copy of a recurrent layer (qwen4exp build_conv_state_at: a CPY of
// the last d_conv - 1 columns of the conv input into the recurrent cache) is deferred to the short convolution that reads the same input
// a few nodes later, whose short-token kernel writes the same floats (ssm_conv_f32, see ssm_conv_state_copies): one launch per layer
// fewer. Only when nothing between them touches the copy's destination or writes over the input, and the convolution's outputs do not
// overlap its input (an allocator could put the SILU output over the freed input: then the copy could read overwritten floats)
static bool ggml_cuda_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr || ggml_nbytes(a) == 0 || ggml_nbytes(b) == 0) {
        return false;
    }
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// a deferred conv-state copy whose convolution did not take it (not expected: the defer checks the convolution's path) runs at the end
// of the graph evaluation as the copy it was
static void ggml_cuda_conv_state_flush(ggml_backend_cuda_context * cuda_ctx) {
    if (cuda_ctx->conv_state_pending.empty()) {
        return;
    }
    static bool warned = false;
    if (!warned) {
        warned = true;
        GGML_LOG_WARN("%s: deferred conv-state copies ran at the end of the graph (GGML_CUDA_DISABLE_CONV_STATE_FUSION=1 off)\n", __func__);
    }
    for (auto & [conv, cpys] : cuda_ctx->conv_state_pending) {
        for (ggml_tensor * c : cpys) {
            ggml_cuda_cpy(*cuda_ctx, c->src[0], c->src[1]);
        }
    }
    cuda_ctx->conv_state_pending.clear();
}

// GGML_CUDA_FUSION_TRACE=1: why a conv-input concat was not elided (refusal point, concat name)
static bool ggml_cuda_conv_cat_refuse(const char * name, int where) {
    static const bool trace = ggml_cuda_env_flag("GGML_CUDA_FUSION_TRACE");
    if (trace) {
        GGML_LOG_INFO("conv cat elide: %s refused at %d\n", name, where);
    }
    return false;
}

static bool ggml_cuda_try_conv_state_defer(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_CONV_STATE_FUSION");
    if (disabled || ggml_cuda_fusion_disabled() || cuda_ctx->curr_stream_no != 0) {
        return false;
    }
    ggml_tensor * cpy = cgraph->nodes[i];
    const ggml_tensor * tail = cpy->src[0];
    const ggml_tensor * dst  = cpy->src[1];
    if (cpy->op != GGML_OP_CPY || tail == nullptr || dst == nullptr || tail->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        (cpy->flags & GGML_TENSOR_FLAG_OUTPUT) || tail->view_src == nullptr || tail->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        tail->ne[3] != 1 || dst->ne[3] != 1 || !ggml_are_same_shape(tail, dst) || dst->nb[1] % sizeof(float) != 0 ||
        dst->nb[2] % sizeof(float) != 0 || tail->data == nullptr || dst->data == nullptr) {
        return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 1);
    }
    const ggml_tensor * input = tail->view_src;
    // the tail is a column range of every row of the input: the same row and sequence strides, the offset inside the first row
    const size_t offs = (const char *) tail->data - (const char *) input->data;
    if (input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) || tail->nb[1] != input->nb[1] || tail->nb[2] != input->nb[2] ||
        tail->ne[1] != input->ne[1] || tail->ne[2] != input->ne[2] || offs % sizeof(float) != 0 ||
        offs / sizeof(float) + tail->ne[0] > (size_t) input->ne[0]) {
        return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 2);
    }
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 24; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_SSM_CONV && n->src[0] == input) {
            const ggml_tensor * w = n->src[1];
            const int64_t n_t = n->ne[1];
            if (w == nullptr || w->ne[0] != tail->ne[0] + 1 || n->ne[0] != input->ne[1] || n->ne[2] != input->ne[2] || n_t > 32 ||
                input->nb[1] != input->ne[0] * (int64_t) sizeof(float) || input->ne[1] % 128 != 0) {
                return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 3);
            }
            // the LFM2 short conv * c fusion does not take the copies; the outputs the kernel may write must not overlap the input
            if (j + 1 < cgraph->n_nodes && cgraph->nodes[j + 1]->op == GGML_OP_MUL) {
                return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 4);
            }
            for (int k = j; k < cgraph->n_nodes && k <= j + 2; ++k) {
                if (ggml_cuda_ranges_overlap(cgraph->nodes[k], input) || ggml_cuda_ranges_overlap(cgraph->nodes[k], dst)) {
                    return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 5);
                }
            }
            auto & v = cuda_ctx->conv_state_pending[n];
            if (v.size() >= 4) {
                return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 6);
            }
            v.push_back(cpy);
            return true;
        }
        // views do not run
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        // a node between them: it must not read or write the copy's destination, nor write over the input
        if (ggml_cuda_ranges_overlap(n, dst) || ggml_cuda_ranges_overlap(n, input)) {
            if (!(n->op == GGML_OP_CPY && n->src[0] != nullptr && n->src[0]->view_src == input && !ggml_cuda_ranges_overlap(n, dst))) {
                return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 7);
            }
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (ggml_cuda_ranges_overlap(n->src[k], dst)) {
                return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 8);
            }
        }
    }
    return ggml_cuda_conv_cat_refuse(cpy->name, 100 + 9);
}

// k15-fn (GGML_CUDA_DISABLE_CONV_CAT_FUSION=1 off): the CONCAT (dim 0) of a short convolution's state columns and its transposed new
// tokens (qwen4exp build_conv_state_at) is not run in a decode / verify graph: the convolution reads both parts (ssm_conv_cat_f32), and
// the conv-state copies deferred to it read the same registers. Only when every reader of the concat is that convolution or a copy of a
// view of it (a copy that is not deferred makes the concat run first, ggml_cuda_conv_cat_materialize), nothing that runs before the
// convolution writes over either part, and the convolution's outputs do not overlap them
static bool ggml_cuda_try_conv_cat_elide(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_CONV_CAT_FUSION");
    if (disabled || ggml_cuda_fusion_disabled() || cuda_ctx->curr_stream_no != 0) {
        return false;
    }
    const ggml_tensor * cat = cgraph->nodes[i];
    const char * cat_dbg_name = cat->name;
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || cat->type != GGML_TYPE_F32 || a == nullptr || b == nullptr ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || (cat->flags & GGML_TENSOR_FLAG_OUTPUT) || cat->ne[3] != 1 ||
        a->ne[3] != 1 || b->ne[3] != 1 || cat->data == nullptr || a->data == nullptr || b->data == nullptr || a->nb[0] != sizeof(float) ||
        a->nb[1] % sizeof(float) != 0 || a->nb[2] % sizeof(float) != 0 || b->nb[0] % sizeof(float) != 0 || b->nb[1] % sizeof(float) != 0 ||
        b->nb[2] % sizeof(float) != 0 || b->ne[0] > 8 || a->ne[1] != b->ne[1] || a->ne[2] != b->ne[2] || cat->ne[1] % 128 != 0 ||
        cat->nb[1] != (size_t) cat->ne[0] * sizeof(float)) {
        return ggml_cuda_conv_cat_refuse(cat_dbg_name, 2);
    }
    int jc = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 24; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_SSM_CONV && n->src[0] == cat) {
            const ggml_tensor * w = n->src[1];
            if (w == nullptr || w->ne[0] != a->ne[0] + 1 || n->ne[1] != b->ne[0] || n->ne[2] != a->ne[2]) {
                return ggml_cuda_conv_cat_refuse(cat_dbg_name, 3);
            }
            // the LFM2 short conv * c product reads the input in its own kernel
            if (j + 1 < cgraph->n_nodes && cgraph->nodes[j + 1]->op == GGML_OP_MUL) {
                return ggml_cuda_conv_cat_refuse(cat_dbg_name, 4);
            }
            for (int k = j; k < cgraph->n_nodes && k <= j + 2; ++k) {
                if (ggml_cuda_ranges_overlap(cgraph->nodes[k], a) || ggml_cuda_ranges_overlap(cgraph->nodes[k], b)) {
                    return ggml_cuda_conv_cat_refuse(cat_dbg_name, 5);
                }
            }
            jc = j;
            break;
        }
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        // a copy of a view of the concat is deferred to the convolution, or makes the concat run before it: it writes only after that
        if (n->op == GGML_OP_CPY && n->src[0] != nullptr && n->src[0]->view_src == cat) {
            continue;
        }
        if (ggml_cuda_ranges_overlap(n, a) || ggml_cuda_ranges_overlap(n, b)) {
            return ggml_cuda_conv_cat_refuse(cat_dbg_name, 6);
        }
    }
    if (jc < 0) {
        return ggml_cuda_conv_cat_refuse(cat_dbg_name, 7);
    }
    // every reader of the concat: the convolution, or a copy of a view of it that comes before the convolution (a later one would
    // read the concat after the convolution consumed it, unwritten)
    for (int k = i + 1; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (k == jc || ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = n->src[s];
            if (t != nullptr && (t == cat || t->view_src == cat) && !(n->op == GGML_OP_CPY && s == 0 && t->view_src == cat && k < jc)) {
                return ggml_cuda_conv_cat_refuse(cat_dbg_name, 8);
            }
        }
    }
    cuda_ctx->conv_cat_elided.insert(cat);
    return true;
}

// an elided conv-input concat that a node about to run reads (other than its convolution), or an elided CONT (other than by the fused
// UNARY + MUL that took it): it runs now
static void ggml_cuda_conv_cat_materialize(ggml_backend_cuda_context * cuda_ctx, const ggml_tensor * node) {
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * t = node->src[s];
        if (t == nullptr) {
            continue;
        }
        const ggml_tensor * root = t->view_src != nullptr ? t->view_src : t;
        if (node->op != GGML_OP_SSM_CONV) {
            auto it = cuda_ctx->conv_cat_elided.find(root);
            if (it != cuda_ctx->conv_cat_elided.end()) {
                cuda_ctx->conv_cat_elided.erase(it);
                ggml_cuda_op_concat(*cuda_ctx, const_cast<ggml_tensor *>(root));
            }
        }
        auto ic = cuda_ctx->unary_cont_elided.find(root);
        if (ic != cuda_ctx->unary_cont_elided.end()) {
            cuda_ctx->unary_cont_elided.erase(ic);
            static int n_warn = 0;
            if (n_warn++ < 4) {
                GGML_LOG_WARN("%s: elided CONT %s read by %s (%s): it runs (not expected)\n", __func__, root->name, node->name, ggml_op_desc(node));
            }
            ggml_cuda_dup(*cuda_ctx, const_cast<ggml_tensor *>(root));
        }
    }
}

// k15-fn (GGML_CUDA_DISABLE_UNARY_SRC_CONT_FUSION=1 off): the CONT of a strided 3D f32 view (the attention output gate of qwen35 /
// qwen35moe / qwen3next / qwen4exp: a view of every head's second half of Qcur_full, made contiguous) whose only reader is a UNARY
// fused with the MUL after it (ggml_cuda_op_unary_mul): the fused kernel reads the view, the CONT does not run. Decode / verify rows
// only (at most 8); nothing that runs up to the MUL may write over the view. Returns the UNARY index, -1: no match
static int ggml_cuda_unary_src_cont_match(const ggml_cgraph * cgraph, int i, bool runtime) {
    const ggml_tensor * c = cgraph->nodes[i];
    const ggml_tensor * s = c->src[0];
    if (c->op != GGML_OP_CONT || c->type != GGML_TYPE_F32 || s == nullptr || s->type != GGML_TYPE_F32 || s->nb[0] != sizeof(float) ||
        s->ne[3] != 1 || s->ne[2] > 8 || c->ne[0] != s->ne[0]*s->ne[1] || ggml_nrows(c) != s->ne[2] || !ggml_is_contiguous(c) ||
        s->nb[1] % sizeof(float) != 0 || s->nb[2] % sizeof(float) != 0 || (c->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return -1;
    }
    int u = -1;
    for (int k = i + 1; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        bool reads = false;
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            const ggml_tensor * t = n->src[q];
            if (t != nullptr && (t == c || t->view_src == c)) {
                reads = true;
            }
        }
        if (!reads) {
            continue;
        }
        if (u >= 0 || n->op != GGML_OP_UNARY || n->src[0] != c) {
            return -1;
        }
        u = k;
        if (!runtime) {
            break;
        }
    }
    if (u < 0 || u + 1 >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * un  = cgraph->nodes[u];
    const ggml_tensor * mul = cgraph->nodes[u + 1];
    const ggml_unary_op op = ggml_get_unary_op(un);
    if ((op != GGML_UNARY_OP_SILU && op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) || mul->op != GGML_OP_MUL ||
        (mul->src[0] != un && mul->src[1] != un) || !ggml_cuda_can_fuse(cgraph, u, { GGML_OP_UNARY, GGML_OP_MUL }, { op })) {
        return -1;
    }
    const ggml_tensor * x = mul->src[0] == un ? mul->src[1] : mul->src[0];
    if (x->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous_1(x) || x->nb[0] != sizeof(float) ||
        !ggml_are_same_shape(c, x) || c->ne[0] == 1) {
        return -1;
    }
    if (runtime) {
        if (s->data == nullptr) {
            return -1;
        }
        for (int k = i + 1; k <= u + 1; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (!ggml_cuda_is_view_or_noop(n) && n != un && ggml_cuda_ranges_overlap(n, s)) {
                return -1;
            }
        }
    }
    return u;
}

static bool ggml_cuda_try_unary_src_cont_elide(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_UNARY_SRC_CONT_FUSION");
    if (disabled || ggml_cuda_fusion_disabled() || cuda_ctx->curr_stream_no != 0) {
        return false;
    }
    if (ggml_cuda_unary_src_cont_match(cgraph, i, /*runtime=*/true) < 0) {
        return false;
    }
    cuda_ctx->unary_cont_elided.insert(cgraph->nodes[i]);
    return true;
}

// k15-fn (GGML_CUDA_DISABLE_QSA_POOL_FUSION=1 off): a QSA layer's new pooled keys (qwen4exp build_qsa_sel): GET_ROWS of the kpool
// member keys -> CONT (member 0) -> kpool - 1 ADDs (members 1..) -> SCALE -> RMS_NORM -> MUL (weight row) -> RESHAPE -> ROPE (multi-section,
// not vision) -> RESHAPE -> SET_ROWS (the pooled half of the cache rows), only views between them. Fills m with the chain (nullptr: no
// match) and returns the SET_ROWS index. runtime: also the conditions on the data (alignment the unfused norm launch takes the float4
// path with, the gather and the scatter on disjoint columns, no reader of an intermediate outside the chain)
struct ggml_cuda_qsa_pool_chain {
    const ggml_tensor * get_rows = nullptr;
    const ggml_tensor * scale    = nullptr;
    const ggml_tensor * rms_norm = nullptr;
    const ggml_tensor * mul      = nullptr;
    const ggml_tensor * rope     = nullptr;
    const ggml_tensor * set_rows = nullptr;
    int kpool = 0;
};

static int ggml_cuda_qsa_pool_match(const ggml_cgraph * cgraph, int i, ggml_cuda_qsa_pool_chain & m, bool runtime) {
    m = {};
    const ggml_tensor * g = cgraph->nodes[i];
    if (g->op != GGML_OP_GET_ROWS || g->type != GGML_TYPE_F32 || g->src[0] == nullptr || g->src[1] == nullptr ||
        (g->src[0]->type != GGML_TYPE_F16 && g->src[0]->type != GGML_TYPE_F32) || g->src[0]->nb[0] != ggml_type_size(g->src[0]->type) ||
        g->src[1]->type != GGML_TYPE_I32 || g->ne[2] != 1 || g->ne[3] != 1 || g->src[0]->ne[2] != 1 || g->src[0]->ne[3] != 1 ||
        ggml_nelements(g->src[1]) != g->ne[1] || !ggml_is_contiguous(g)) {
        return -1;
    }
    const int64_t ncols = g->ne[0];
    if (ncols % 4 != 0 || ncols / 4 > 32 || ncols > GGML_CUDA_QSA_POOL_MAX_COLS) {
        return -1;
    }
    auto next = [&](int j) {
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            j++;
        }
        return j;
    };
    std::vector<const ggml_tensor *> chain = { g };
    int j = next(i + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * c = cgraph->nodes[j];
    if (c->op != GGML_OP_CONT || c->type != GGML_TYPE_F32 || c->src[0] == nullptr || c->src[0]->view_src != g) {
        return -1;
    }
    const size_t row_b = ncols * sizeof(float);
    const int64_t kpool = c->src[0]->nb[1] / row_b;
    if (c->src[0]->nb[1] % row_b != 0 || kpool < 2 || kpool > 8 || g->ne[1] % kpool != 0) {
        return -1;
    }
    const int64_t n_new = g->ne[1] / kpool;
    auto slice_ok = [&](const ggml_tensor * t, int64_t k) {
        return t != nullptr && t->view_src == g && t->type == GGML_TYPE_F32 && t->ne[0] == ncols && t->ne[1] == n_new && t->ne[2] == 1 &&
               t->ne[3] == 1 && t->nb[0] == sizeof(float) && t->nb[1] == (size_t) kpool * row_b && t->view_offs == (size_t) k * row_b;
    };
    if (!slice_ok(c->src[0], 0) || !ggml_is_contiguous(c) || c->ne[0] != ncols || c->ne[1] != n_new) {
        return -1;
    }
    chain.push_back(c);
    const ggml_tensor * prev = c;
    for (int64_t k = 1; k < kpool; ++k) {
        j = next(j + 1);
        if (j >= cgraph->n_nodes) {
            return -1;
        }
        const ggml_tensor * a = cgraph->nodes[j];
        if (a->op != GGML_OP_ADD || a->type != GGML_TYPE_F32 || a->src[0] != prev || !slice_ok(a->src[1], k) || !ggml_are_same_shape(a, c) ||
            !ggml_is_contiguous(a)) {
            return -1;
        }
        chain.push_back(a);
        prev = a;
    }
    j = next(j + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * sc = cgraph->nodes[j];
    if (sc->op != GGML_OP_SCALE || sc->type != GGML_TYPE_F32 || sc->src[0] != prev || !ggml_is_contiguous(sc)) {
        return -1;
    }
    chain.push_back(sc);
    j = next(j + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * rn = cgraph->nodes[j];
    if (rn->op != GGML_OP_RMS_NORM || rn->type != GGML_TYPE_F32 || rn->src[0] != sc) {
        return -1;
    }
    chain.push_back(rn);
    j = next(j + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * mul = cgraph->nodes[j];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || (mul->src[0] != rn && mul->src[1] != rn) || !ggml_are_same_shape(mul, rn) ||
        !ggml_is_contiguous(mul)) {
        return -1;
    }
    const ggml_tensor * w = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    if (w == nullptr || w == rn || w->type != GGML_TYPE_F32 || w->ne[0] != ncols || ggml_nrows(w) != 1 || !ggml_is_contiguous(w)) {
        return -1;
    }
    chain.push_back(mul);
    j = next(j + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * rope = cgraph->nodes[j];
    const ggml_tensor * rs   = rope->src[0];
    if (rope->op != GGML_OP_ROPE || rope->type != GGML_TYPE_F32 || rs == nullptr || rs->view_src != mul || rs->view_offs != 0 ||
        !ggml_is_contiguous(rs) || rs->ne[0] != ncols || rs->ne[1] != 1 || rs->ne[2] != n_new || rs->ne[3] != 1 ||
        !ggml_are_same_shape(rope, rs) || !ggml_is_contiguous(rope) || rope->src[1] == nullptr || rope->src[1]->type != GGML_TYPE_I32 ||
        rope->src[1]->ne[0] < 4*n_new || (rope->src[2] != nullptr && rope->src[2]->type != GGML_TYPE_F32)) {
        return -1;
    }
    const int mode   = ((const int32_t *) rope->op_params)[2];
    const int n_dims = ((const int32_t *) rope->op_params)[1];
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (!(mode & GGML_ROPE_TYPE_MROPE) || mode == GGML_ROPE_TYPE_VISION || n_dims <= 0 || n_dims % 2 != 0 || n_offs < 0 || n_offs % 2 != 0 ||
        n_offs + n_dims > ncols) {
        return -1;
    }
    chain.push_back(rope);
    j = next(j + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * sr = cgraph->nodes[j];
    const ggml_tensor * sv = sr->src[0];
    if (sr->op != GGML_OP_SET_ROWS || (sr->type != GGML_TYPE_F16 && sr->type != GGML_TYPE_F32) || sv == nullptr || sv->view_src != rope ||
        sv->view_offs != 0 || !ggml_is_contiguous(sv) || sv->ne[0] != ncols || sv->ne[1] != n_new || sv->ne[2] != 1 || sv->ne[3] != 1 ||
        sr->src[1] == nullptr || (sr->src[1]->type != GGML_TYPE_I64 && sr->src[1]->type != GGML_TYPE_I32) || sr->src[1]->ne[0] != n_new ||
        ggml_nelements(sr->src[1]) != n_new || sr->ne[0] != ncols || sr->ne[2] != 1 || sr->ne[3] != 1 || sr->nb[0] != ggml_type_size(sr->type)) {
        return -1;
    }
    if (runtime) {
        for (const ggml_tensor * t : chain) {
            if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
                return -1;
            }
        }
        // the float4 path of the unfused norm (rms_norm_f32_dispatch): its input, output and weight 16-byte aligned
        if (!ggml_cuda_is_aligned(sc, 16) || ((uintptr_t) mul->data & 15) != 0 || !ggml_cuda_is_aligned(w, 16) || g->src[0]->data == nullptr ||
            sr->data == nullptr) {
            return -1;
        }
        // the gather reads the first ncols of each key row, the scatter writes the pooled columns of a row of the same cache: disjoint
        // columns of rows of one stride, or no overlap at all
        const ggml_tensor * kv = g->src[0];
        if (ggml_cuda_ranges_overlap(kv, sr)) {
            const ggml_tensor * kr = kv->view_src != nullptr ? kv->view_src : kv;
            const ggml_tensor * sroot = sr->view_src != nullptr ? sr->view_src : sr;
            if (kr != sroot || kv->nb[1] != sr->nb[1]) {
                return -1;
            }
            const size_t nb1 = kv->nb[1];
            const size_t ck  = ((const char *) kv->data - (const char *) kr->data) % nb1;
            const size_t cs  = ((const char *) sr->data - (const char *) kr->data) % nb1;
            const size_t ek  = ck + ncols * ggml_type_size(kv->type);
            const size_t es  = cs + ncols * ggml_type_size(sr->type);
            if (ek > nb1 || es > nb1 || !(ek <= cs || es <= ck)) {
                return -1;
            }
        }
        // no reader of an intermediate outside the chain
        for (int k = i + 1; k < cgraph->n_nodes; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (n == sr || ggml_cuda_is_view_or_noop(n) || std::find(chain.begin(), chain.end(), n) != chain.end()) {
                continue;
            }
            for (int q = 0; q < GGML_MAX_SRC; ++q) {
                const ggml_tensor * t = n->src[q];
                if (t == nullptr) {
                    continue;
                }
                const ggml_tensor * root = t->view_src != nullptr ? t->view_src : t;
                if (std::find(chain.begin(), chain.end(), root) != chain.end()) {
                    return -1;
                }
            }
        }
    }
    m.get_rows = g;
    m.scale    = sc;
    m.rms_norm = rn;
    m.mul      = mul;
    m.rope     = rope;
    m.set_rows = sr;
    m.kpool    = (int) kpool;
    return j;
}

// k15-fn (GGML_CUDA_DISABLE_QSA_KEY_PACK_FUSION=1 off): a QSA layer's packed indexer key row (qwen4exp build_qsa_sel: FILL of the
// pooled half -> CONCAT (dim 0) of the raw key and it -> views -> SET_ROWS into the cache) as one kernel at the FILL that writes the
// raw key and the fill value with set_rows' conversion (the concat only copies floats). Returns the SET_ROWS index, -1: no match
struct ggml_cuda_qsa_key_pack {
    const ggml_tensor * fill     = nullptr;
    const ggml_tensor * concat   = nullptr;
    const ggml_tensor * set_rows = nullptr;
};

static int ggml_cuda_qsa_key_pack_match(const ggml_cgraph * cgraph, int i, ggml_cuda_qsa_key_pack & m, bool runtime) {
    m = {};
    const ggml_tensor * f = cgraph->nodes[i];
    if (f->op != GGML_OP_FILL || f->type != GGML_TYPE_F32 || !ggml_is_contiguous(f) || f->ne[2] != 1 || f->ne[3] != 1) {
        return -1;
    }
    auto next = [&](int j) {
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            j++;
        }
        return j;
    };
    int j = next(i + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * cat = cgraph->nodes[j];
    const ggml_tensor * kr  = cat->src[0];
    const int64_t nt = f->ne[1];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || cat->type != GGML_TYPE_F32 || cat->src[1] != f || kr == nullptr ||
        kr->type != GGML_TYPE_F32 || kr->nb[0] != sizeof(float) || kr->ne[1] != nt || kr->ne[2] != 1 || kr->ne[3] != 1 ||
        kr->nb[1] % sizeof(float) != 0 || cat->ne[0] != kr->ne[0] + f->ne[0] || cat->ne[1] != nt) {
        return -1;
    }
    j = next(j + 1);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * sr = cgraph->nodes[j];
    const ggml_tensor * sv = sr->src[0];
    const ggml_tensor * ix = sr->src[1];
    if (sr->op != GGML_OP_SET_ROWS || (sr->type != GGML_TYPE_F16 && sr->type != GGML_TYPE_F32) || sr->nb[0] != ggml_type_size(sr->type) ||
        sv == nullptr || sv->view_src != cat || sv->view_offs != 0 || !ggml_is_contiguous(sv) || sv->ne[0] != cat->ne[0] || sv->ne[1] != nt ||
        sv->ne[2] != 1 || sv->ne[3] != 1 || ix == nullptr || (ix->type != GGML_TYPE_I64 && ix->type != GGML_TYPE_I32) ||
        ggml_nelements(ix) != nt || ix->ne[0] != nt || sr->ne[0] != cat->ne[0] || sr->ne[2] != 1 || sr->ne[3] != 1) {
        return -1;
    }
    if (runtime) {
        if ((f->flags & GGML_TENSOR_FLAG_OUTPUT) || (cat->flags & GGML_TENSOR_FLAG_OUTPUT) || kr->data == nullptr || sr->data == nullptr ||
            ix->data == nullptr || ggml_cuda_ranges_overlap(kr, sr)) {
            return -1;
        }
        for (int k = i + 1; k < cgraph->n_nodes; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (n == cat || n == sr || ggml_cuda_is_view_or_noop(n)) {
                continue;
            }
            for (int q = 0; q < GGML_MAX_SRC; ++q) {
                const ggml_tensor * t = n->src[q];
                if (t == nullptr) {
                    continue;
                }
                const ggml_tensor * root = t->view_src != nullptr ? t->view_src : t;
                if (root == f || root == cat) {
                    return -1;
                }
            }
        }
    }
    m.fill     = f;
    m.concat   = cat;
    m.set_rows = sr;
    return j;
}

template <typename dst_t, typename idx_t>
static __global__ void k_qsa_key_pack(const float * kr, const int64_t kr_s1, const int n0, const int n, const float c, const idx_t * idx,
                                      const int64_t idx_s0, char * dst, const int64_t dst_nb1) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int t = blockIdx.x;
    const int64_t row = (int64_t) idx[(int64_t) t*idx_s0];
    dst_t * d = (dst_t *) (dst + row*dst_nb1);
    const float * k = kr + (int64_t) t*kr_s1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        d[j] = ggml_cuda_cast<dst_t>(j < n0 ? k[j] : c);
    }
}

static int ggml_cuda_try_fuse_qsa_key_pack(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_QSA_KEY_PACK_FUSION");
    if (disabled || ggml_cuda_fusion_disabled() || cuda_ctx->curr_stream_no != 0) {
        return 0;
    }
    ggml_cuda_qsa_key_pack m;
    const int j = ggml_cuda_qsa_key_pack_match(cgraph, i, m, /*runtime=*/true);
    if (j < 0) {
        return 0;
    }
    const ggml_tensor * kr = m.concat->src[0];
    const ggml_tensor * ix = m.set_rows->src[1];
    float c;
    memcpy(&c, m.fill->op_params, sizeof(float));
    const int nt = (int) m.fill->ne[1];
    const int n0 = (int) kr->ne[0];
    const int n  = (int) m.concat->ne[0];
    const int64_t kr_s1  = kr->nb[1] / sizeof(float);
    const int64_t idx_s0 = ix->nb[0] / ggml_type_size(ix->type);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(std::min(256, GGML_PAD(n, 32)), 1, 1), 0,
                                                                             cuda_ctx->stream());
    char * dst = (char *) m.set_rows->data;
    const size_t nb1 = m.set_rows->nb[1];
    const float * krd = (const float *) kr->data;
    if (m.set_rows->type == GGML_TYPE_F16) {
        if (ix->type == GGML_TYPE_I64) {
            ggml_cuda_kernel_launch(k_qsa_key_pack<half, int64_t>, lp, krd, kr_s1, n0, n, c, (const int64_t *) ix->data, idx_s0, dst, (int64_t) nb1);
        } else {
            ggml_cuda_kernel_launch(k_qsa_key_pack<half, int32_t>, lp, krd, kr_s1, n0, n, c, (const int32_t *) ix->data, idx_s0, dst, (int64_t) nb1);
        }
    } else {
        if (ix->type == GGML_TYPE_I64) {
            ggml_cuda_kernel_launch(k_qsa_key_pack<float, int64_t>, lp, krd, kr_s1, n0, n, c, (const int64_t *) ix->data, idx_s0, dst, (int64_t) nb1);
        } else {
            ggml_cuda_kernel_launch(k_qsa_key_pack<float, int32_t>, lp, krd, kr_s1, n0, n, c, (const int32_t *) ix->data, idx_s0, dst, (int64_t) nb1);
        }
    }
    return j - i + 1;
}

static int ggml_cuda_try_fuse_qsa_pool(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_QSA_POOL_FUSION");
    if (disabled || ggml_cuda_fusion_disabled() || cuda_ctx->curr_stream_no != 0) {
        return 0;
    }
    ggml_cuda_qsa_pool_chain m;
    const int j = ggml_cuda_qsa_pool_match(cgraph, i, m, /*runtime=*/true);
    if (j < 0) {
        return 0;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: QSA pooled keys in one launch (kpool %d, %d nodes; GGML_CUDA_DISABLE_QSA_POOL_FUSION=1 off)\n", __func__, m.kpool,
            j - i + 1);
    }
    ggml_cuda_op_qsa_pool_fused(*cuda_ctx, m.get_rows, m.scale, m.rms_norm, m.mul, m.rope, m.set_rows, m.kpool);
    return j - i + 1;
}

// k15-fn: the gated_delta_net (non-KDA, one sequence, not the mma kernel) that is the only reader of t (through views of t; its src[k] a
// contiguous whole view of t), or -1
static int ggml_cuda_sole_gdn_reader(const ggml_cgraph * cgraph, int i, const ggml_tensor * t, int k_expected) {
    int gdn = -1;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = n->src[k];
            if (s == nullptr || (s != t && s->view_src != t)) {
                continue;
            }
            if (ggml_cuda_is_view_or_noop(n) && n->view_src == t) {
                continue;
            }
            if (n->op == GGML_OP_GATED_DELTA_NET && k == k_expected && gdn < 0 && s->view_offs == 0 && ggml_nelements(s) == ggml_nelements(t) &&
                ggml_is_contiguous(s) && n->src[3]->ne[0] == 1 && n->src[2]->ne[3] == 1 && !ggml_cuda_gated_delta_net_may_use_mma(n)) {
                gdn = j;
                continue;
            }
            return -1;
        }
    }
    return gdn;
}

static bool ggml_cuda_gdn_sigmoid_deferred(const ggml_cgraph * cgraph, int i);

// k15-fn: no node between i and j writes over the bytes of t (the allocator may reuse them once their last planned reader ran); a beta
// sigmoid that its gated_delta_net applies itself (ggml_cuda_gdn_sigmoid_deferred) writes nothing
static bool ggml_cuda_kept_until(const ggml_cgraph * cgraph, int i, int j, const ggml_tensor * t, bool skip_deferred = false) {
    for (int m = i + 1; m < j; ++m) {
        const ggml_tensor * n = cgraph->nodes[m];
        if (ggml_cuda_is_view_or_noop(n) || n->data == nullptr) {
            continue;
        }
        if (skip_deferred && ggml_cuda_gdn_sigmoid_deferred(cgraph, m)) {
            continue;
        }
        if (ggml_cuda_tensors_overlap(n, t)) {
            static const bool dbg = getenv("GGML_CUDA_GDN_GATES_DEBUG") != nullptr;
            if (dbg) {
                GGML_LOG_INFO("%s: %s overwritten by node %d %s (%s)\n", __func__, t->name, m - i, n->name, ggml_op_desc(n));
            }
            return false;
        }
    }
    return true;
}

// k15-fn: the beta sigmoid at node i is applied by the gated_delta_net that reads it (ggml_cuda_try_gdn_gates), so it is not run
static bool ggml_cuda_gdn_sigmoid_deferred(const ggml_cgraph * cgraph, int i) {
    static const bool enabled = getenv("GGML_CUDA_GDN_GATES") == nullptr || atoi(getenv("GGML_CUDA_GDN_GATES")) != 0;
    const ggml_tensor * node = cgraph->nodes[i];
    if (!enabled || ggml_cuda_fusion_disabled() || node->op != GGML_OP_UNARY || ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * raw = node->src[0];
    if (node->type != GGML_TYPE_F32 || raw->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || !ggml_is_contiguous(raw) ||
        (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int j = ggml_cuda_sole_gdn_reader(cgraph, i, node, 4);
    return j >= 0 && ggml_cuda_kept_until(cgraph, i, j, raw);
}

// k15-fn (GGML_CUDA_GDN_GATES=0 off): the delta-net gates beta = sigmoid(b) and g = softplus(a + dt_bias) * A, each read only by a
// gated_delta_net, are applied by that kernel to its raw inputs: two launches fewer per layer (Qwen3.8-Flash-Next: 72 per verify), the
// same float operations, so bitwise. Returns the nodes covered (1 for the sigmoid, 3 for ADD -> SOFTPLUS -> MUL), 0 if not applicable
static int ggml_cuda_try_gdn_gates(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = getenv("GGML_CUDA_GDN_GATES") == nullptr || atoi(getenv("GGML_CUDA_GDN_GATES")) != 0;
    if (!enabled || ggml_cuda_fusion_disabled()) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID) {
        if (!ggml_cuda_gdn_sigmoid_deferred(cgraph, i)) {
            return 0;
        }
        const int j = ggml_cuda_sole_gdn_reader(cgraph, i, node, 4);
        cuda_ctx->gdn_gates_in[cgraph->nodes[j]].beta_raw = (const float *) node->src[0]->data;
        return 1;
    }
    if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
        const ggml_tensor * add   = node;
        const ggml_tensor * unary = cgraph->nodes[i + 1];
        const ggml_tensor * mul   = cgraph->nodes[i + 2];
        if (unary->op != GGML_OP_UNARY || ggml_get_unary_op(unary) != GGML_UNARY_OP_SOFTPLUS || unary->src[0] != add || mul->op != GGML_OP_MUL ||
            (mul->src[0] == unary) == (mul->src[1] == unary)) {
            return 0;
        }
        const ggml_op ops[3] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };
        if (!ggml_can_fuse(cgraph, i, ops, 3) || (mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        const ggml_tensor * a     = ggml_are_same_shape(add->src[0], add) ? add->src[0] : add->src[1];
        const ggml_tensor * bias  = a == add->src[0] ? add->src[1] : add->src[0];
        const ggml_tensor * scale = mul->src[0] == unary ? mul->src[1] : mul->src[0];
        if (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(a, add) || !ggml_are_same_shape(a, mul) || !ggml_is_contiguous(a) || !ggml_is_contiguous(mul) ||
            bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] || ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            return 0;
        }
        const int j = ggml_cuda_sole_gdn_reader(cgraph, i + 2, mul, 3);
        static const bool dbg = getenv("GGML_CUDA_GDN_GATES_DEBUG") != nullptr;
        if (dbg) {
            GGML_LOG_INFO("%s: %s: reader %d, heads %lld / %lld, a kept %d\n", __func__, mul->name, j, j >= 0 ? (long long) cgraph->nodes[j]->src[2]->ne[1] : -1LL,
                (long long) a->ne[0], j >= 0 ? (int) ggml_cuda_kept_until(cgraph, i + 2, j, a, true) : -1);
        }
        // the gate row index of the kernel is its head: a's rows are the heads ([H, T, S], as the kernel's g [1, H, T, S])
        // the chain's own nodes are not run, so an in-place ADD / SOFTPLUS / MUL over a does not count
        if (j < 0 || cgraph->nodes[j]->src[2]->ne[1] != a->ne[0] || !ggml_cuda_kept_until(cgraph, i + 2, j, a, true) ||
            !ggml_cuda_kept_until(cgraph, i + 2, j, bias, true) || !ggml_cuda_kept_until(cgraph, i + 2, j, scale, true)) {
            return 0;
        }
        auto & e = cuda_ctx->gdn_gates_in[cgraph->nodes[j]];
        e.a     = (const float *) a->data;
        e.bias  = (const float *) bias->data;
        e.scale = (const float *) scale->data;
        return 3;
    }
    return 0;
}

static bool ggml_cuda_side_mv_run(const ggml_tensor * node);

// k15-mm2: the same for the F32 / BF16 vector kernel (ggml_cuda_mul_mat_vec_f_multi): products of one weight type and K over one
// activation, e.g. a gated delta net layer's alpha + beta, the router + the shared expert gate, the indexer q + k. Each output gets
// the bytes of its own launch; the lookahead and the overlap rules are those of ggml_cuda_try_fuse_mmvq_multi. Products the side
// stream runs (single column, ggml_cuda_side_mv_run) and K >= 4096 (the split-K launch) are left alone.
// GGML_CUDA_DISABLE_MMVF_MULTI=1 turns it off.
static bool ggml_cuda_mmvf_multi_disabled() {
    // on again: the text that was not repeatable with it at a spilled tier (Qwen3.6-35B-A3B MTP, B200) came from the fused residual
    // add + RMS norm writing over its inputs, see the allocation dependencies in ggml_backend_cuda_graph_optimize
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_MMVF_MULTI");
    return disabled;
}

static bool ggml_cuda_mmvf_multi_candidate(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || node->view_src != nullptr || node->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16) && src0->op == GGML_OP_NONE && src0->view_src == nullptr &&
        ggml_is_contiguous(src0) && ggml_is_matrix(src0) && src0->ne[0] < 4096 && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) &&
        src1->ne[2] == 1 && src1->ne[3] == 1 && src1->ne[1] <= 4 && ggml_is_contiguous(node) && !ggml_cuda_side_mv_run(node);
}

// node goes to ggml_cuda_mul_mat_vec_f in ggml_cuda_mul_mat (same dispatch, see there)
static bool ggml_cuda_mmvf_multi_runtime_ok(const ggml_backend_cuda_context * ctx, const ggml_tensor * node) {
    if (!ggml_cuda_mmvf_multi_candidate(node) || ggml_cuda_op_mul_mat_use_fwht(node)) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src0->buffer == nullptr || !ggml_backend_buft_is_cuda(src0->buffer->buft) ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE || src1->data == nullptr || node->data == nullptr) {
        return false;
    }
    const int     cc        = ggml_cuda_info().devices[ctx->device].cc;
    const int     warp_size = ggml_cuda_info().devices[ctx->device].warp_size;
    const int64_t ne11      = src1->ne[1];
    bool ok;
    if (src0->type == GGML_TYPE_F32 && ggml_get_op_params_i32(node, 3) == GGML_PREC_F32) {
        ok = ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, 1);
    } else if ((node->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && ne11 > 1) {
        ok = ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, 1);
    } else {
        ok = ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11);
    }
    return ok && ggml_cuda_mul_mat_vec_f_multi_ok(src0->type, src0->ne[0], ne11, ctx->device);
}

static int ggml_cuda_try_fuse_mmvf_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_mmvf_multi_disabled() || ggml_cuda_fusion_disabled() || !cuda_ctx->stream_context().concurrent_events.empty() ||
            cuda_ctx->curr_stream_no != 0) {
        return -1;
    }
    ggml_tensor * node = cgraph->nodes[i];
    if (!ggml_cuda_mmvf_multi_candidate(node)) {
        return -1;
    }
    {
        bool any = false;
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + GGML_CUDA_MMVQ_MULTI_LOOKAHEAD && !any; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            any = t->op == GGML_OP_MUL_MAT && t->src[1] == node->src[1];
        }
        if (!any) {
            return -1;
        }
    }
    if (!ggml_cuda_mmvf_multi_runtime_ok(cuda_ctx, node) || ggml_cuda_mmvq_multi_has_partner(cgraph, i)) {
        return -1;
    }
    const ggml_tensor * act = node->src[1];
    ggml_tensor * dst[GGML_CUDA_MMVQ_MULTI_MAX] = { node };
    const ggml_tensor * src0[GGML_CUDA_MMVQ_MULTI_MAX] = { node->src[0] };
    int idx[GGML_CUDA_MMVQ_MULTI_MAX] = { i };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + GGML_CUDA_MMVQ_MULTI_LOOKAHEAD && n < GGML_CUDA_MMVQ_MULTI_MAX; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (cuda_ctx->mmvq_done_early.count(t)) {
            continue;
        }
        const bool member = ggml_cuda_mmvq_multi_pair(node, t) && ggml_get_op_params_i32(node, 3) == ggml_get_op_params_i32(t, 3) &&
            ggml_cuda_mmvf_multi_runtime_ok(cuda_ctx, t) && !ggml_cuda_mmvq_multi_has_partner(cgraph, j);
        if (member) {
            bool ok = true;
            for (int k = i + 1; k < j && ok; ++k) {
                const ggml_tensor * u = cgraph->nodes[k];
                if (ggml_cuda_mmvq_multi_overlap(u, t)) {
                    ok = false;
                }
                for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
                    if (u->src[s] != nullptr && ggml_cuda_mmvq_multi_overlap(u->src[s], t)) {
                        ok = false;
                    }
                }
            }
            for (int m = 0; m < n && ok; ++m) {
                ok = !ggml_cuda_mmvq_multi_overlap(dst[m], t);
            }
            if (ok) {
                dst[n] = t;
                src0[n] = t->src[0];
                idx[n] = j;
                n++;
                continue;
            }
        }
        if (!ggml_cuda_is_view_or_noop(t) && ggml_cuda_mmvq_multi_overlap(t, act)) {
            break;
        }
    }
    if (n < 2) {
        return -1;
    }
    ggml_cuda_mul_mat_vec_f_multi(*cuda_ctx, n, src0, act, dst);
    int run = 0;
    for (int m = 1; m < n; ++m) {
        if (idx[m] == i + run + 1) {
            run++;
        } else {
            cuda_ctx->mmvq_done_early.insert(dst[m]);
            cuda_ctx->q8_1_cache.invalidate(dst[m]->data, ggml_nbytes(dst[m]));
        }
    }
    return run;
}

// k15-fn (GGML_CUDA_DISABLE_QSA_IDX_FUSION=1 off): the QSA dead-slot index map of each layer, i32 -> f32 cast, (idx - dump) * live + dump,
// f32 -> i32 cast (5 launches), as one kernel. The same float operations in the same order, rounded as the separate kernels round
// them (explicit _rn intrinsics, no contraction into an FMA), and the casts as ggml_cuda_cast does them
static __global__ void k_qsa_idx_select(const int32_t * __restrict__ idx, const float * __restrict__ dump, const float * __restrict__ live,
        int32_t * __restrict__ dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float f = ggml_cuda_cast<float>(idx[i]);
    const float d = dump[i];
    const float r = __fadd_rn(__fmul_rn(__fsub_rn(f, d), live[i]), d);
    dst[i] = ggml_cuda_cast<int32_t>(r);
}

// the same map in one block, results staged in shared memory before any is written: safe when dst overlaps an input at an offset
static __global__ void k_qsa_idx_select_staged(const int32_t * __restrict__ idx, const float * __restrict__ dump, const float * __restrict__ live,
        int32_t * __restrict__ dst, const int n) {
    extern __shared__ int32_t staged[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float f = ggml_cuda_cast<float>(idx[i]);
        const float d = dump[i];
        const float r = __fadd_rn(__fmul_rn(__fsub_rn(f, d), live[i]), d);
        staged[i] = ggml_cuda_cast<int32_t>(r);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = staged[i];
    }
}

static int ggml_cuda_try_fuse_qsa_idx(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    // (before the overlap rule below the 8k / 16k md5 on G4 fn4 IQ4_XS 32 GB Studio varied from launch to launch: k15gd2_g4, k15ge_g4;
    // with it stable and equal to the kill switch, k15gv_g4)
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_QSA_IDX_FUSION");
    if (disabled || i + 4 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * c1  = cgraph->nodes[i];
    ggml_tensor * sub = cgraph->nodes[i + 1];
    ggml_tensor * mul = cgraph->nodes[i + 2];
    ggml_tensor * add = cgraph->nodes[i + 3];
    ggml_tensor * c2  = cgraph->nodes[i + 4];
    const auto f32c = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_are_same_shape(t, c1);
    };
    if (c1->op != GGML_OP_CPY || c1->src[0]->type != GGML_TYPE_I32 || !ggml_is_contiguous(c1->src[0]) || !f32c(c1) ||
            !ggml_are_same_shape(c1, c1->src[0]) ||
            sub->op != GGML_OP_SUB || sub->src[0] != c1 || !f32c(sub) || !f32c(sub->src[1]) ||
            mul->op != GGML_OP_MUL || mul->src[0] != sub || !f32c(mul) || !f32c(mul->src[1]) ||
            add->op != GGML_OP_ADD || add->src[0] != mul || add->src[1] != sub->src[1] || !f32c(add) ||
            c2->op != GGML_OP_CPY || c2->src[0] != add || c2->type != GGML_TYPE_I32 || !ggml_is_contiguous(c2) || !ggml_are_same_shape(c2, c1)) {
        return 0;
    }
    // read only by the next node of the chain (a cast is also its own src[1], see ggml_cast)
    for (int k = 0; k < 4; ++k) {
        const ggml_tensor * t = cgraph->nodes[i + k];
        const int self = t->op == GGML_OP_CPY && t->src[1] == t ? 1 : 0;
        if (ggml_node_get_use_count(cgraph, i + k) != 1 + self || t->view_src != nullptr || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    // the cast of a CPY node writes its src[1] (dst view): the result tensor is c2 itself here (ggml_cast)
    if (c2->src[1] != nullptr && c2->src[1]->data != c2->data) {
        return 0;
    }
    // the allocator may place c2 in memory freed inside the chain (live after the mul, dump after the add): the separate kernels finish
    // reading it first, one grid would race with its other threads unless c2 starts exactly where the input does (element i of both
    // belongs to thread i). Then one block stages every result before writing any (decode / verify sizes), else no fusion.
    // G4 fn4 32 GB: 8k md5 changed from launch to launch without this (k15gd2_g4, k15ge_g4); 8 of 12 layers overlap in the fn1 verify graph
    bool overlap = false;
    for (const ggml_tensor * in : { c1->src[0], sub->src[1], mul->src[1] }) {
        overlap = overlap || (in->data != c2->data && ggml_cuda_tensors_overlap(in, c2));
    }
    const int64_t n_all = ggml_nelements(c1);
    if (overlap && n_all > 8192) {
        return 0;
    }
    const int64_t n = n_all;
    if (overlap) {
        k_qsa_idx_select_staged<<<1, 1024, n*sizeof(int32_t), cuda_ctx->stream()>>>((const int32_t *) c1->src[0]->data,
            (const float *) sub->src[1]->data, (const float *) mul->src[1]->data, (int32_t *) c2->data, (int) n);
    } else {
        const int block = 256;
        k_qsa_idx_select<<<(unsigned) ((n + block - 1)/block), block, 0, cuda_ctx->stream()>>>((const int32_t *) c1->src[0]->data,
            (const float *) sub->src[1]->data, (const float *) mul->src[1]->data, (int32_t *) c2->data, n);
    }
    CUDA_CHECK(cudaGetLastError());
    return 4;
}

// k15-fn (GGML_CUDA_DISABLE_QSA_SEL_FUSION=1 off): the rest of a QSA layer's selection chain in the launch of its index map. From the
// top-k pools to the i32 cell index of every slot (llama_model_qwen4exp::graph::build_qsa_sel):
//   sel_idx = concat(get_rows(pools, top_k), tail)                                   member cells of the top pools, then the tail
//   live    = concat(repeat(clamp(scale(get_rows(score, top_k)))), clamp(scale(cast(tail))))
//   dump    = scale(cumsum(fill(live, 1)))                                            n_kv + slot
//   idx     = cast_i32((cast_f32(sel_idx) - dump) * live + dump)
// about 14 launches per layer as one block. Every value is what the separate kernels produce: integer cells and dump rows below 2^24
// are exact in f32, the scale / clamp of the live flags use the nodes' own parameters in the kernels' expressions, the map the explicit
// _rn intrinsics of k_qsa_idx_select. The chain nodes are skipped (qsa_sel_skip) only when nothing outside the chain reads them and no
// node that still runs before the index node writes over an input of the chain (the allocator may give memory the skipped readers
// freed); the block stages its results, so its output may overlap its inputs
static __device__ __forceinline__ int32_t qsa_sel_elem(const ggml_backend_cuda_context::qsa_sel_args & a, const int i) {
    const int n_pool_sel = a.kpool*a.n_top;
    const int n_sel = n_pool_sel + a.n_tail;
    const int s = i % n_sel;
    const int t = i / n_sel;
    int32_t cell;
    float   live;
    if (s < n_pool_sel) {
        const int32_t p = a.top_k[(int64_t) t*a.top_k_s1 + s / a.kpool];
        cell = a.pools[(int64_t) p*a.pools_s1 + s % a.kpool];
        const float sc = a.score[(int64_t) t*a.score_s1 + p];
        live = fminf(fmaxf(a.lp_scale * sc + a.lp_bias, a.lp_min), a.lp_max);
    } else {
        cell = a.tail[(int64_t) t*a.tail_s1 + (s - n_pool_sel)];
        live = fminf(fmaxf(a.lt_scale * ggml_cuda_cast<float>(cell) + a.lt_bias, a.lt_min), a.lt_max);
    }
    const float d = a.d_scale * (float) (s + 1) + a.d_bias;
    const float f = ggml_cuda_cast<float>(cell);
    const float r = __fadd_rn(__fmul_rn(__fsub_rn(f, d), live), d);
    return ggml_cuda_cast<int32_t>(r);
}

static __global__ void k_qsa_sel_chain(const ggml_backend_cuda_context::qsa_sel_args a, int32_t * __restrict__ dst) {
    extern __shared__ int32_t staged[];
    const int n = (a.kpool*a.n_top + a.n_tail)*a.n_tok;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        staged[i] = qsa_sel_elem(a, i);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = staged[i];
    }
}

// k15-fn (GGML_CUDA_QSA_SEL_GRID=0: the staged single block always): the output overlaps no input (the keep-alive table keeps them
// apart), so each thread writes its own element: one thread per element over a grid instead of one block looping over all of them
static __global__ void k_qsa_sel_chain_grid(const ggml_backend_cuda_context::qsa_sel_args a, int32_t * __restrict__ dst, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = qsa_sel_elem(a, i);
    }
}

// t through reshapes (each read once): the first non-reshape node, the reshapes collected
static const ggml_tensor * ggml_cuda_qsa_sel_unreshape(const ggml_tensor * t, std::vector<const ggml_tensor *> & chain) {
    while (t != nullptr && t->op == GGML_OP_RESHAPE) {
        chain.push_back(t);
        t = t->src[0];
    }
    return t;
}

static bool ggml_cuda_qsa_sel_match(const ggml_cgraph * cgraph, const ggml_tensor * c2, ggml_backend_cuda_context::qsa_sel_args & a,
        std::vector<const ggml_tensor *> & chain, std::vector<const ggml_tensor *> & inputs) {
    const auto i32c = [](const ggml_tensor * t) { return t != nullptr && t->type == GGML_TYPE_I32 && t->nb[0] == sizeof(int32_t); };
    const auto f32c = [](const ggml_tensor * t) { return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); };
    if (c2->op != GGML_OP_CPY || c2->type != GGML_TYPE_I32 || !ggml_is_contiguous(c2) || (c2->src[1] != nullptr && c2->src[1]->data != c2->data)) {
        return false;
    }
    const ggml_tensor * add = c2->src[0];
    if (add == nullptr || add->op != GGML_OP_ADD || !f32c(add)) return false;
    const ggml_tensor * mul = add->src[0];
    const ggml_tensor * dump = add->src[1];
    if (mul == nullptr || mul->op != GGML_OP_MUL || !f32c(mul)) return false;
    const ggml_tensor * sub = mul->src[0];
    const ggml_tensor * live = mul->src[1];
    if (sub == nullptr || sub->op != GGML_OP_SUB || sub->src[1] != dump || !f32c(sub) || !f32c(live) || !f32c(dump)) return false;
    const ggml_tensor * c1 = sub->src[0];
    if (c1 == nullptr || c1->op != GGML_OP_CPY || !f32c(c1)) return false;
    const ggml_tensor * sel_idx = c1->src[0];
    if (!i32c(sel_idx) || !ggml_is_contiguous(sel_idx) || sel_idx->op != GGML_OP_CONCAT || ggml_get_op_params_i32(sel_idx, 0) != 0) return false;
    const int64_t n_sel = c2->ne[0];
    const int64_t n_tok = c2->ne[1];
    if (c2->ne[2] != 1 || c2->ne[3] != 1 || !ggml_are_same_shape(c1, c2) || !ggml_are_same_shape(sel_idx, c2) || !ggml_are_same_shape(live, c2) ||
            !ggml_are_same_shape(dump, c2) || !ggml_are_same_shape(add, c2) || !ggml_are_same_shape(mul, c2) || !ggml_are_same_shape(sub, c2) ||
            n_sel*n_tok > 8192) {
        return false;
    }
    chain.insert(chain.end(), { add, mul, sub, c1, sel_idx, live, dump });
    // sel_idx = concat(reshape(get_rows(pools, reshape(top_k))), tail)
    const ggml_tensor * gr = ggml_cuda_qsa_sel_unreshape(sel_idx->src[0], chain);
    const ggml_tensor * tail = sel_idx->src[1];
    if (gr == nullptr || gr->op != GGML_OP_GET_ROWS || !i32c(gr) || !ggml_is_contiguous(gr) || !i32c(tail) || tail->ne[1] != n_tok ||
            tail->ne[2] != 1 || tail->ne[3] != 1) {
        return false;
    }
    chain.push_back(gr);
    const ggml_tensor * pools = gr->src[0];
    const ggml_tensor * topk_flat = gr->src[1];
    if (!i32c(pools) || pools->ne[2] != 1 || pools->ne[3] != 1 || topk_flat == nullptr || topk_flat->type != GGML_TYPE_I32) return false;
    const ggml_tensor * top_k = topk_flat;
    while (top_k->op == GGML_OP_RESHAPE && top_k->src[0] != nullptr) {
        top_k = top_k->src[0];
    }
    if (!i32c(top_k) || top_k->ne[1] != n_tok || top_k->ne[2] != 1 || top_k->ne[3] != 1 || !ggml_is_contiguous(topk_flat)) return false;
    const int64_t kpool = pools->ne[0];
    const int64_t n_top = top_k->ne[0];
    if (gr->ne[0] != kpool || gr->ne[1] != n_top*n_tok || kpool*n_top + tail->ne[0] != n_sel) return false;
    // live = concat(reshape(repeat(clamp(scale(get_rows(reshape(score), top_k))))), clamp(scale(cast(tail))))
    if (live->op != GGML_OP_CONCAT || ggml_get_op_params_i32(live, 0) != 0) return false;
    const ggml_tensor * rp = ggml_cuda_qsa_sel_unreshape(live->src[0], chain);
    if (rp == nullptr || rp->op != GGML_OP_REPEAT || !f32c(rp) || rp->ne[0] != kpool || rp->ne[1] != n_top || rp->ne[2] != n_tok) return false;
    const ggml_tensor * cl1 = rp->src[0];
    if (cl1 == nullptr || cl1->op != GGML_OP_CLAMP || !f32c(cl1) || cl1->ne[0] != 1 || cl1->ne[1] != n_top || cl1->ne[2] != n_tok) return false;
    const ggml_tensor * sc1 = cl1->src[0];
    if (sc1 == nullptr || sc1->op != GGML_OP_SCALE || !f32c(sc1)) return false;
    const ggml_tensor * gs = sc1->src[0];
    if (gs == nullptr || gs->op != GGML_OP_GET_ROWS || !f32c(gs) || gs->src[1] != top_k || gs->ne[0] != 1) return false;
    const ggml_tensor * score_r = gs->src[0];
    if (score_r == nullptr || score_r->op != GGML_OP_RESHAPE || score_r->ne[0] != 1 || score_r->ne[2] != n_tok) return false;
    const ggml_tensor * score = score_r->src[0];
    if (!f32c(score) || score->ne[1] != n_tok || score->ne[0] != score_r->ne[1]) return false;
    chain.insert(chain.end(), { rp, cl1, sc1, gs, score_r });
    const ggml_tensor * cl2 = live->src[1];
    if (cl2 == nullptr || cl2->op != GGML_OP_CLAMP || !f32c(cl2) || !ggml_are_same_shape(cl2, tail)) return false;
    const ggml_tensor * sc2 = cl2->src[0];
    if (sc2 == nullptr || sc2->op != GGML_OP_SCALE || !f32c(sc2)) return false;
    const ggml_tensor * ct = sc2->src[0];
    if (ct == nullptr || ct->op != GGML_OP_CPY || !f32c(ct) || ct->src[0] != tail) return false;
    chain.insert(chain.end(), { cl2, sc2, ct });
    // dump = scale(cumsum(fill(live, 1)))
    if (dump->op != GGML_OP_SCALE) return false;
    const ggml_tensor * cs = dump->src[0];
    if (cs == nullptr || cs->op != GGML_OP_CUMSUM || !f32c(cs)) return false;
    const ggml_tensor * fl = cs->src[0];
    if (fl == nullptr || fl->op != GGML_OP_FILL || !f32c(fl) || fl->src[0] != live || ggml_get_op_params_f32(fl, 0) != 1.0f ||
            fl->view_src != nullptr) {
        return false;
    }
    chain.insert(chain.end(), { cs, fl });
    float p[2];
    memcpy(p, sc1->op_params, sizeof(p)); a.lp_scale = p[0]; a.lp_bias = p[1];
    memcpy(p, cl1->op_params, sizeof(p)); a.lp_min = p[0]; a.lp_max = p[1];
    memcpy(p, sc2->op_params, sizeof(p)); a.lt_scale = p[0]; a.lt_bias = p[1];
    memcpy(p, cl2->op_params, sizeof(p)); a.lt_min = p[0]; a.lt_max = p[1];
    memcpy(p, dump->op_params, sizeof(p)); a.d_scale = p[0]; a.d_bias = p[1];
    // the dump rows must be exact integers in f32
    if (!(std::fabs(a.d_scale*(float) n_sel) + std::fabs(a.d_bias) < 16777216.0f) || a.d_scale != 1.0f) return false;
    // a scale of +-1 makes scale * x + bias one rounding with or without contraction into an FMA, as in scale_f32
    if (std::fabs(a.lp_scale) != 1.0f || std::fabs(a.lt_scale) != 1.0f) return false;
    a.top_k = (const int32_t *) top_k->data; a.top_k_s1 = top_k->nb[1] / sizeof(int32_t);
    a.pools = (const int32_t *) pools->data; a.pools_s1 = pools->nb[1] / sizeof(int32_t);
    a.tail  = (const int32_t *) tail->data;  a.tail_s1  = tail->nb[1] / sizeof(int32_t);
    a.score = (const float *) score->data;   a.score_s1 = score->nb[1] / sizeof(float);
    a.kpool = (int) kpool; a.n_top = (int) n_top; a.n_tail = (int) tail->ne[0]; a.n_tok = (int) n_tok;
    if (top_k->nb[1] % sizeof(int32_t) || pools->nb[1] % sizeof(int32_t) || tail->nb[1] % sizeof(int32_t) || score->nb[1] % sizeof(float)) {
        return false;
    }
    inputs.insert(inputs.end(), { top_k, pools, tail, score });
    GGML_UNUSED(cgraph);
    return true;
}

static void ggml_cuda_qsa_sel_plan(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    cuda_ctx->qsa_sel.clear();
    cuda_ctx->qsa_sel_skip.clear();
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_QSA_SEL_FUSION");
    if (disabled || ggml_cuda_fusion_disabled()) {
        return;
    }
    std::unordered_map<const ggml_tensor *, int> pos;
    std::vector<const ggml_tensor *> chain, inputs;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * c2 = cgraph->nodes[i];
        if (c2->op != GGML_OP_CPY || c2->type != GGML_TYPE_I32) {
            continue;
        }
        ggml_backend_cuda_context::qsa_sel_args a = {};
        chain.clear();
        inputs.clear();
        if (!ggml_cuda_qsa_sel_match(cgraph, c2, a, chain, inputs)) {
            continue;
        }
        if (pos.empty()) {
            for (int j = 0; j < cgraph->n_nodes; ++j) {
                pos[cgraph->nodes[j]] = j;
            }
        }
        // every chain node: a graph node before c2, not an output, read only by chain nodes (and c2)
        std::unordered_set<const ggml_tensor *> in_chain(chain.begin(), chain.end());
        bool ok = true;
        int i_first = i;
        for (const ggml_tensor * t : chain) {
            auto it = pos.find(t);
            if (it == pos.end() || it->second >= i || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                ok = false;
                break;
            }
            i_first = std::min(i_first, it->second);
            int inside = 0;
            for (const ggml_tensor * u : chain) {
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    inside += u->src[k] == t;
                }
            }
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                inside += c2->src[k] == t;
            }
            if (ggml_cuda_graph_use_count(cgraph, t) != inside) {
                ok = false;
                break;
            }
        }
        // the inputs stay as they were until c2: no node that still runs from the first skipped reader on writes over them
        for (int j = i_first + 1; ok && j < i; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            if (in_chain.count(t) || ggml_cuda_is_view_or_noop(t) || cuda_ctx->qsa_skip.count(t) || t->data == nullptr) {
                continue;
            }
            for (const ggml_tensor * in : inputs) {
                if (ggml_cuda_tensors_overlap(t, in)) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            continue;
        }
        a.no_alias = c2->data != nullptr;
        for (const ggml_tensor * in : inputs) {
            if (in->data == nullptr || ggml_cuda_tensors_overlap(c2, in)) {
                a.no_alias = false;
            }
        }
        cuda_ctx->qsa_sel[c2] = a;
        cuda_ctx->qsa_sel_skip.insert(chain.begin(), chain.end());
    }
    static size_t n_logged = SIZE_MAX;
    if (!cuda_ctx->qsa_sel.empty() && cuda_ctx->qsa_sel.size() != n_logged) {
        n_logged = cuda_ctx->qsa_sel.size();
        GGML_LOG_INFO("%s: %zu QSA selection chains in one launch each, %zu nodes not run (GGML_CUDA_DISABLE_QSA_SEL_FUSION=1 off)\n", __func__,
            cuda_ctx->qsa_sel.size(), cuda_ctx->qsa_sel_skip.size());
    }
}

static void ggml_cuda_qsa_sel_launch(ggml_backend_cuda_context * cuda_ctx, const ggml_tensor * c2, const ggml_backend_cuda_context::qsa_sel_args & a) {
    const int n = (a.kpool*a.n_top + a.n_tail)*a.n_tok;
    static const bool grid = getenv("GGML_CUDA_QSA_SEL_GRID") == nullptr || atoi(getenv("GGML_CUDA_QSA_SEL_GRID")) != 0;
    if (grid && a.no_alias) {
        k_qsa_sel_chain_grid<<<(n + 127)/128, 128, 0, cuda_ctx->stream()>>>(a, (int32_t *) c2->data, n);
    } else {
        k_qsa_sel_chain<<<1, 256, n*sizeof(int32_t), cuda_ctx->stream()>>>(a, (int32_t *) c2->data);
    }
    CUDA_CHECK(cudaGetLastError());
}

// k15-fn (GGML_CUDA_DISABLE_SET_ROWS_PAIR=1 off): a SET_ROWS whose next compute node is another SET_ROWS of the same types (the K and
// V cache stores of an attention layer, the two index-cache stores of a QSA layer) runs with it in one launch. Neither reads or writes
// the tensor the other writes (different view roots), so the order between them does not matter. Returns the extra nodes covered.
static const ggml_tensor * ggml_cuda_view_root(const ggml_tensor * t) {
    while (t != nullptr && t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

static int ggml_cuda_try_fuse_set_rows_pair(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_SET_ROWS_PAIR") || getenv("GGML_CUDA_SET_ROWS_CHECK") != nullptr;
    const ggml_tensor * a = cgraph->nodes[i];
    if (disabled || a->op != GGML_OP_SET_ROWS) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && j <= i + 4 && (ggml_cuda_is_view_or_noop(cgraph->nodes[j]) || (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
        j++;
    }
    if (j >= cgraph->n_nodes || j > i + 4) {
        return 0;
    }
    const ggml_tensor * b = cgraph->nodes[j];
    if (b->op != GGML_OP_SET_ROWS || (b->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !ggml_cuda_set_rows_pair_supported(a, b)) {
        return 0;
    }
    if (!cuda_ctx->qsa_skip.empty() && (cuda_ctx->qsa_skip.count(a) || cuda_ctx->qsa_skip.count(b))) {
        return 0;
    }
    const ggml_tensor * ra = ggml_cuda_view_root(a);
    const ggml_tensor * rb = ggml_cuda_view_root(b);
    if (ra == rb || ggml_cuda_view_root(b->src[0]) == ra || ggml_cuda_view_root(b->src[1]) == ra ||
            ggml_cuda_view_root(a->src[0]) == rb || ggml_cuda_view_root(a->src[1]) == rb) {
        return 0;
    }
    // the skipped views between them are not read by b as a's output (checked above through the roots)
    if (!ggml_cuda_set_rows_pair(*cuda_ctx, a, b)) {
        return 0;
    }
    static int64_t n_pairs = 0;
    if (n_pairs++ == 0) {
        GGML_LOG_INFO("%s: SET_ROWS pairs in one launch (%s + %s; GGML_CUDA_DISABLE_SET_ROWS_PAIR=1 off)\n", __func__, a->name, b->name);
    }
    return j - i;
}

// k15-mm2: SCALE / SILU chain (each node read only by the next) -> MUL_MAT whose src1 is the chain end, a short-K dense Q8_0
// product of 1..4 columns (the hyper-connection up projection behind silu(scale(down))): the product's blocks apply the chain and
// quantize its output themselves (ggml_cuda_mul_mat_vec_q_ew), one launch in place of the chain kernel, quantize_q8_1 and
// mul_mat_vec_q, with the same bytes. Opt-in, GGML_CUDA_MMVQ_EW=1. Returns the number of extra nodes covered.
static int ggml_cuda_try_fuse_ew_mmvq(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    // opt-in (GGML_CUDA_MMVQ_EW=1): measured slower on B200 at the Flash-Next verify (+5% per forward), every block repeats the chain
    static const bool off = [] { const char * e = getenv("GGML_CUDA_MMVQ_EW"); return e == nullptr || atoi(e) == 0; }();
    if (off || cuda_ctx->curr_stream_no != 0) {
        return 0;
    }
    int   ops[GGML_CUDA_MMVQ_EW_MAX];
    float p0[GGML_CUDA_MMVQ_EW_MAX];
    float p1[GGML_CUDA_MMVQ_EW_MAX];
    int n = 0;
    const ggml_tensor * ew_src = cgraph->nodes[i]->src[0];
    while (i + n < cgraph->n_nodes && n < GGML_CUDA_MMVQ_EW_MAX) {
        const ggml_tensor * t = cgraph->nodes[i + n];
        const bool is_scale = t->op == GGML_OP_SCALE;
        const bool is_silu  = t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SILU;
        if (!(is_scale || is_silu) || t->type != GGML_TYPE_F32 || t->src[0] == nullptr || t->src[0]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(t) || !ggml_is_contiguous(t->src[0]) || !ggml_are_same_shape(t, t->src[0]) ||
                !(t->flags & GGML_TENSOR_FLAG_COMPUTE) || (n > 0 && t->src[0] != cgraph->nodes[i + n - 1]) ||
                !ggml_node_has_n_uses(cgraph, i + n, 1)) {
            break;
        }
        ops[n] = is_scale ? 0 : 1;
        p0[n]  = 0.0f;
        p1[n]  = 0.0f;
        if (is_scale) {
            memcpy(&p0[n], (const float *) t->op_params + 0, sizeof(float));
            memcpy(&p1[n], (const float *) t->op_params + 1, sizeof(float));
        }
        n++;
    }
    if (n == 0 || i + n >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i + n];
    const ggml_tensor * last = cgraph->nodes[i + n - 1];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != last || !(mm->flags & GGML_TENSOR_FLAG_COMPUTE) || mm->view_src != nullptr ||
            cuda_ctx->mmvq_done_early.count(mm)) {
        return 0;
    }
    const ggml_tensor * src0 = mm->src[0];
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const int64_t ne11 = mm->ne[1];
    const int64_t ne11_k = (mm->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) ? 1 : ne11; // batch invariant: as one column
    // the kernel the unfused product would take (ggml_cuda_mul_mat): mul_mat_vec_q
    if (src0->buffer == nullptr ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE || ggml_cuda_op_mul_mat_use_fwht(mm) ||
            !ggml_cuda_mul_mat_vec_q_ew_ok(src0, mm, cc) || ew_src->ne[0] != src0->ne[0] || ew_src->ne[1] != ne11 ||
            !ggml_is_contiguous(ew_src) || ggml_cuda_should_use_mmf(src0->type, cc, 32, src0->ne, src0->nb, ne11_k, false) ||
            !ggml_cuda_should_use_mmvq(src0->type, cc, ne11_k)) {
        return 0;
    }
    // every block reads the whole chain input while others write their rows: they must not share memory
    if (ggml_cuda_tensors_overlap(mm, ew_src)) {
        return 0;
    }
    ggml_cuda_mul_mat_vec_q_ew(*cuda_ctx, src0, ew_src, n, ops, p0, p1, mm);
    return n;
}

// node goes to ggml_cuda_mul_mat_vec_f in ggml_cuda_mul_mat (same dispatch, see there), whatever its K
static bool ggml_cuda_mmvf_dispatch_ok(const ggml_backend_cuda_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (node->op != GGML_OP_MUL_MAT || ggml_cuda_op_mul_mat_use_fwht(node) || src0->buffer == nullptr ||
            !ggml_backend_buft_is_cuda(src0->buffer->buft) || ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            src0->view_src != nullptr || src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || src1->ne[1] > MMVF_MAX_BATCH_SIZE) {
        return false;
    }
    const int     cc        = ggml_cuda_info().devices[ctx->device].cc;
    const int     warp_size = ggml_cuda_info().devices[ctx->device].warp_size;
    const int64_t ne11      = src1->ne[1];
    if ((src0->type == GGML_TYPE_F32 && ggml_get_op_params_i32(node, 3) == GGML_PREC_F32) ||
            ((node->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && ne11 > 1)) {
        return ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, 1);
    }
    return ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11);
}

// k15-mm2: the hyper-connection down projection (Q8_0) and inject product (F32, K = hc*n_embd -> hc rows, split-K) of one mix read the
// same activation and are adjacent once the side-stream planner moved the inject behind its activation: one launch for both
// (ggml_cuda_mul_mat_vec_q_inject, same bytes). Node i and the next compute node j; the inject then does not use the side stream.
// Opt-in, GGML_CUDA_HC_INJECT_DOWN=1: break-even on G4 (k15mm2g4k fn4 32 GB: 128.9 / 115.6 tok/s chat / 8k vs 129.1 / 115.4 off) and
// B200 (-0.4% / -0.1% per decode), inside the noise; it gives up the inject's side-stream overlap for one launch fewer per mix.
static bool ggml_cuda_hc_inject_down_match(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i, int & j_out,
        ggml_tensor ** q_out, ggml_tensor ** f_out, ggml_cuda_mmvf_splitk_args * inj) {
    static const bool off = [] { const char * e = getenv("GGML_CUDA_HC_INJECT_DOWN"); return e == nullptr || atoi(e) == 0; }();
    ggml_tensor * a = cgraph->nodes[i];
    if (off || ggml_cuda_fusion_disabled() || a->op != GGML_OP_MUL_MAT || !(a->flags & GGML_TENSOR_FLAG_COMPUTE) || a->view_src != nullptr) {
        return false;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * b = cgraph->nodes[j];
    if (b->op != GGML_OP_MUL_MAT || !(b->flags & GGML_TENSOR_FLAG_COMPUTE) || b->view_src != nullptr || b->src[1] != a->src[1] ||
            (a->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) != (b->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
        return false;
    }
    ggml_tensor * q = a->src[0]->type == GGML_TYPE_Q8_0 ? a : b;
    ggml_tensor * f = q == a ? b : a;
    const int iq = q == a ? i : j;
    const int iff = q == a ? j : i;
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (q->src[0]->type != GGML_TYPE_Q8_0 || f->src[0]->type != GGML_TYPE_F32 || f->src[0]->op != GGML_OP_NONE ||
            !ggml_cuda_mul_mat_vec_q_inject_ok(q->src[0], q->src[1], q, cc) || !ggml_cuda_mmvq_multi_runtime_ok(cuda_ctx, q) ||
            !ggml_cuda_mmvf_dispatch_ok(cuda_ctx, f) || ggml_cuda_mmvq_multi_has_partner(cgraph, iq) ||
            ggml_cuda_mmvq_multi_has_partner(cgraph, iff) || !ggml_is_contiguous(f) || !ggml_is_contiguous(f->src[0])) {
        return false;
    }
    if (ggml_cuda_tensors_overlap(q, f) || ggml_cuda_tensors_overlap(q, q->src[1]) || ggml_cuda_tensors_overlap(f, q->src[1])) {
        return false;
    }
    ggml_cuda_mmvf_splitk_args args;
    if (!ggml_cuda_mul_mat_vec_f_splitk_plan(*cuda_ctx, f->src[0], f->src[1], f, args)) {
        return false;
    }
    j_out = j;
    if (q_out) {
        *q_out = q;
    }
    if (f_out) {
        *f_out = f;
    }
    if (inj) {
        *inj = args;
    }
    return true;
}

static int ggml_cuda_try_fuse_hc_inject_down(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (cuda_ctx->curr_stream_no != 0 || !cuda_ctx->stream_context().concurrent_events.empty() || cuda_ctx->mmvq_done_early.count(cgraph->nodes[i])) {
        return 0;
    }
    int j = -1;
    ggml_tensor * q = nullptr;
    ggml_tensor * f = nullptr;
    ggml_cuda_mmvf_splitk_args inj;
    if (!ggml_cuda_hc_inject_down_match(cuda_ctx, cgraph, i, j, &q, &f, &inj) || cuda_ctx->mmvq_done_early.count(cgraph->nodes[j])) {
        return 0;
    }
    ggml_cuda_mul_mat_vec_q_inject(*cuda_ctx, q->src[0], q->src[1], q, inj);
    return j - i;
}

// k15-mm2: the hyper-connection up projection (Q8_0, gate) and the gated DSV4_HC_PRE reading it (the next compute node): one launch
// that writes the gate, the mixed stream and its q8_1 copy (ggml_cuda_mul_mat_vec_q_hc_pre, same bytes). Opt-in, GGML_CUDA_HC_UP_PRE=1:
// slower on B200 at the Flash-Next verify (10.1 us vs 5.4 + 2.3 us for the two launches; the group arrival and the epilogue after the
// last block of a group cost more than the launch saves).
static int ggml_cuda_try_fuse_hc_up_pre(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool off = [] { const char * e = getenv("GGML_CUDA_HC_UP_PRE"); return e == nullptr || atoi(e) == 0; }();
    ggml_tensor * up = cgraph->nodes[i];
    if (off || cuda_ctx->curr_stream_no != 0 || !cuda_ctx->stream_context().concurrent_events.empty() || up->op != GGML_OP_MUL_MAT ||
            !(up->flags & GGML_TENSOR_FLAG_COMPUTE) || up->view_src != nullptr || up->src[0]->type != GGML_TYPE_Q8_0 ||
            cuda_ctx->mmvq_done_early.count(up)) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * pre = cgraph->nodes[j];
    if (pre->op != GGML_OP_DSV4_HC_PRE || !(pre->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_get_op_params_i32(pre, 1) == 0 ||
            cuda_ctx->mmvq_done_early.count(pre)) {
        return 0;
    }
    const ggml_tensor * x = pre->src[0];
    const ggml_tensor * w = pre->src[1];
    const int64_t n_embd = x->ne[0];
    const int64_t hc     = x->ne[1];
    const int64_t nt     = x->ne[2];
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || pre->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[3] != 1 ||
            w->data != up->data || !ggml_is_contiguous(w) || ggml_nelements(w) != ggml_nelements(up) || w->ne[0] != n_embd || w->ne[1] != hc ||
            w->ne[2] != nt || !ggml_is_contiguous(pre) || pre->ne[0] != n_embd || pre->ne[1] != nt || up->ne[1] != nt ||
            !ggml_cuda_mul_mat_vec_q_hc_pre_ok(up->src[0], up->src[1], up, n_embd, hc, cc) || !ggml_cuda_mmvq_multi_runtime_ok(cuda_ctx, up) ||
            ggml_cuda_mmvq_multi_has_partner(cgraph, i)) {
        return 0;
    }
    // the kernel reads the q8_1 copy of src1, made before it starts (the mixed stream may take the memory of src1, which the up
    // projection read last)
    if (ggml_cuda_tensors_overlap(up, x) || ggml_cuda_tensors_overlap(up, pre) || ggml_cuda_tensors_overlap(pre, x) ||
            ggml_cuda_tensors_overlap(up, up->src[1])) {
        return 0;
    }
    // the group counters, allocated outside a capture (the first evaluations run without one)
    if (cuda_ctx->hc_pre_cnt == nullptr) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(cuda_ctx->stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return 0;
        }
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        CUDA_CHECK(cudaMalloc((void **) &cuda_ctx->hc_pre_cnt, GGML_CUDA_HC_PRE_MAX_GROUPS*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(cuda_ctx->hc_pre_cnt, 0, GGML_CUDA_HC_PRE_MAX_GROUPS*sizeof(unsigned int)));
    }
    // src1's q8_1 copy first, as the unfused up projection gets it before the DSV4_HC_PRE reserves its own
    ggml_cuda_pool_alloc<char> q8_1_alloc(cuda_ctx->pool());
    const char * vy = ggml_cuda_mmvq_src1_q8_1(*cuda_ctx, up->src[1], q8_1_alloc);
    // the q8_1 copy of the mixed stream, as ggml_cuda_op_dsv4_hc_pre reserves it
    void * q8_1_out = nullptr;
    ggml_cuda_q8_1_cache & cache = cuda_ctx->q8_1_cache;
    if (cache.active() && cache.norm_targets.count(pre)) {
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(n_embd, nt, key);
        q8_1_out = cache.reserve(pre->data, ggml_nbytes(pre), key, nt*n_embd/QK8_1*sizeof(block_q8_1));
    }
    ggml_cuda_mul_mat_vec_q_hc_pre(*cuda_ctx, up->src[0], up->src[1], vy, up, (const float *) x->data, x->nb[1] / sizeof(float),
        x->nb[2] / sizeof(float), (float *) pre->data, q8_1_out, cuda_ctx->hc_pre_cnt, n_embd, ggml_get_op_params_f32(pre, 0));
    return j - i;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT) {
        if (const int n = ggml_cuda_try_fuse_hc_inject_down(cuda_ctx, cgraph, i)) {
            return n;
        }
        if (const int n = ggml_cuda_try_fuse_hc_up_pre(cuda_ctx, cgraph, i)) {
            return n;
        }
    }

    if (node->op == GGML_OP_SCALE || node->op == GGML_OP_UNARY) {
        if (const int n = ggml_cuda_try_fuse_ew_mmvq(cuda_ctx, cgraph, i)) {
            return n;
        }
    }

    if (node->op == GGML_OP_ADD) {
        if (const int n = ggml_cuda_try_fuse_add_rms_norm(cuda_ctx, cgraph, i)) {
            return n;
        }
    }

    // k11-decode: LFM2 short-conv window (b * x into the conv window) and conv * c, small batches (GGML_CUDA_SHORTCONV_FUSE=0: off)
    if (node->op == GGML_OP_MUL) {
        if (const int n = ggml_cuda_try_fuse_shortconv_window(*cuda_ctx, cgraph, i)) {
            return n;
        }
    } else if (node->op == GGML_OP_SSM_CONV) {
        if (const int n = ggml_cuda_try_fuse_shortconv_conv_mul(*cuda_ctx, cgraph, i)) {
            return n;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID) {
        if (const int n = ggml_cuda_try_fuse_moe_cache_glu(cuda_ctx, cgraph, i)) {
            return n;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                ops.push_back(GGML_OP_RESHAPE);
                if (args.prob_bias) {
                    ops.push_back(GGML_OP_ADD);
                    bias = cgraph->nodes[i + ops.size() - 1]->src[1];
                }
                if (args.n_groups > 0) {
                    for (const auto & p : topk_moe_group_pattern) {
                        ops.push_back(p.op);
                    }
                }
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                out_nodes[0] = i + ops.size() - 2;
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD && ggml_cuda_try_fuse_add_unary_mul(cuda_ctx, cgraph, i)) {
        return 2;
    }

    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (const bool src1_alias = !ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {});
                   !src1_alias || (op == GGML_OP_MUL_MAT && ggml_cuda_glu_mmvq_src1_alias_ok(cgraph, i))) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (!src1_alias && ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    {
        // k15-fn: the multi-section rope after a q / k norm (qwen4exp attention + indexer q), in the norm kernel
        // (GGML_CUDA_DISABLE_NORM_MROPE_FUSION=1 turns it off)
        static const bool mrope_disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_NORM_MROPE_FUSION");
        if (!mrope_disabled && node->op == GGML_OP_RMS_NORM &&
                ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                ggml_cuda_should_fuse_rms_norm_mul_mrope(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
            int out_nodes[] = { i + 2 };
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, out_nodes, 1)) {
                ggml_cuda_op_rms_norm_mul_mrope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
                return 2;
            }
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        // a mul_mat_vec_q reads this output: write its q8_1 copy in the same kernel (same key as ggml_cuda_mul_mat_vec_q
        // uses for a contiguous src1)
        ggml_tensor * mul = cgraph->nodes[i + 1];
        void * q8_1_out = nullptr;
        ggml_cuda_q8_1_cache & cache = cuda_ctx->q8_1_cache;
        if (cache.active() && cuda_ctx->curr_stream_no == 0 && cache.norm_targets.count(mul) && ggml_is_contiguous(mul)) {
            const int64_t ne10_padded = GGML_PAD(mul->ne[0], MATRIX_ROW_PADDING);
            int64_t key[ggml_cuda_q8_1_cache::n_key];
            ggml_cuda_q8_1_cache::key_contiguous(mul->ne[0], ggml_nrows(mul), key);
            q8_1_out = cache.reserve(mul->data, ggml_nbytes(mul), key, ggml_nrows(mul)*ne10_padded/QK8_1*sizeof(block_q8_1));
        }
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, mul, q8_1_out);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        if (cgraph->nodes[i + 3]->src[2] == cgraph->nodes[i + 2]) {
            if (const int n = ggml_cuda_try_fuse_hc_post_norm(cuda_ctx, cgraph, i, 3, i + 3)) {
                return n;
            }
        }
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    if (node->op == GGML_OP_UNARY) {
        const int n_gate = ggml_cuda_try_fuse_unary_mul_gate(cuda_ctx, cgraph, i);
        if (n_gate > 0) {
            return n_gate;
        }
    }

    if (const int n_skip = ggml_cuda_try_fuse_amax_rescale(cuda_ctx, cgraph, i); n_skip > 0) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_hc_post_chain(cuda_ctx, cgraph, i); n_skip > 0) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_ew_chain(cuda_ctx, cgraph, i); n_skip > 0) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_set_rows_pair(cuda_ctx, cgraph, i); n_skip > 0) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_qsa_idx(cuda_ctx, cgraph, i); n_skip > 0) {
        return n_skip;
    }

    return 0;
}

// Side-stream matrix-vector products. In single-token decode some products have only a few rows (a hyper-connection inject
// weight, a shared-expert gate, ...): they run on a handful of SMs for several microseconds while the rest of the GPU waits,
// although their result is read only much later. Such a product is moved right after the last node it reads (before the
// allocation, so the allocator places its output accordingly; its sources are kept alive until its first reader), and the
// evaluation that captures the CUDA graph runs it on a side stream that the main stream joins before that reader (a graph
// evaluated without capture runs it in place, on the main stream). Every node computes the same values in
// the same order: only the overlap changes. GGML_CUDA_DISABLE_SIDE_MATVEC=1 turns it off.
static constexpr int GGML_CUDA_SIDE_MV_STREAM    = GGML_CUDA_MAX_STREAMS - 2; // GGML_CUDA_MAX_STREAMS - 1: MoE cache copies
static constexpr int GGML_CUDA_SIDE_MV_MAX_ROWS  = 64;
static constexpr int GGML_CUDA_SIDE_MV_MIN_GAP   = 24; // nodes between the moved product and its first reader
static constexpr int GGML_CUDA_SIDE_MV_LOOKAHEAD = 16; // the join is placed this many nodes early (a fused group may start there)
static constexpr int GGML_CUDA_SIDE_MV_MAX_COLS  = 64; // a graph with a planned product over more columns (a prompt) does not use the side stream

// planning only looks at the weight and the topology, not at the batch size: the prompt and decode graphs of a model then get
// the same node order and the same allocation dependencies, so switching between them does not force a new allocation plan
// (planning only the decode graphs was measured: the scheduler then re-reserves and synchronizes at every prompt ubatch).
// The side stream itself is only used for single-row products (ggml_cuda_side_mv_run), see ggml_cuda_side_mv_use for when.
static bool ggml_cuda_side_mv_candidate(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || node->view_src != nullptr) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) && src1->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && src0->ne[1] <= GGML_CUDA_SIDE_MV_MAX_ROWS && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 && node->type == GGML_TYPE_F32;
}

// k15-mm2: products of at most 4 rows also at up to GGML_CUDA_SIDE_MV_COLS columns (default 4: the MTP verify windows; 1 = single
// column only, as before), so a verify's hyper-connection inject and shared expert gate run beside the products after them instead
// of before them. Wider weights (alpha / beta) keep their multi-matrix launch at several columns. Same kernels, same values.
static bool ggml_cuda_side_mv_run(const ggml_tensor * node) {
    static const int max_cols = getenv("GGML_CUDA_SIDE_MV_COLS") ? std::max(1, atoi(getenv("GGML_CUDA_SIDE_MV_COLS"))) : 4;
    if (!ggml_cuda_side_mv_candidate(node)) {
        return false;
    }
    const int64_t ncols = node->src[1]->ne[1];
    // at several columns only long-K ones (the hc inject, K = 10240): a short-K tiny product (the shared expert gate, K = 2560) joins
    // the multi-matrix launch of its neighbour instead (the router: one launch fewer), GGML_CUDA_SIDE_MV_COLS_MIN_K (default 4096)
    static const int64_t min_k = getenv("GGML_CUDA_SIDE_MV_COLS_MIN_K") ? atoll(getenv("GGML_CUDA_SIDE_MV_COLS_MIN_K")) : 4096;
    return ncols == 1 || (ncols <= max_cols && node->src[0]->ne[1] <= 4 && node->src[0]->ne[0] >= min_k);
}

// The side stream is only used while a decode-sized graph is captured into a CUDA graph. Used eagerly, it cost more than it saved:
// on an RTX PRO 6000 (sm_120) the main-stream kernels issued afterwards, also those of the next prompt, started later (about
// 0.25 us more per kernel boundary in an nsys trace): Qwen3.6-35B-A3B 8k prompt -1.8% / -1.9% vs GGML_CUDA_DISABLE_SIDE_MATVEC=1,
// and with CUDA graphs off decode -7%. The same branches inside a captured graph cost nothing. A prompt graph (a planned product
// over more than GGML_CUDA_SIDE_MV_MAX_COLS columns) runs all of them on the main stream, as there is nothing to overlap.
static bool ggml_cuda_side_mv_use(const std::unordered_map<const ggml_tensor *, const ggml_tensor *> & side_mv, const bool capturing) {
    if (side_mv.empty() || !capturing) {
        return false;
    }
    for (const auto & [mv, reader] : side_mv) {
        if (mv->src[1]->ne[1] > GGML_CUDA_SIDE_MV_MAX_COLS) {
            return false;
        }
    }
    return true;
}

// k11: see ggml_cuda_src1_memo. Called before node i runs: drop a held copy that node i does not read, and note whether
// node i should keep its converted src1 for the next compute node.
static bool ggml_cuda_src1_memo_begin(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int i, bool allowed) {
    static const bool disabled = getenv("GGML_CUDA_SRC1_MEMO") != nullptr && atoi(getenv("GGML_CUDA_SRC1_MEMO")) == 0;
    ggml_cuda_src1_memo & memo = ctx->src1_memo;
    const ggml_tensor * node = cgraph->nodes[i];
    memo.share_next = false;
    const bool is_mm = node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID;
    if (memo.held && !(is_mm && node->src[1] == memo.src1)) {
        memo.release();
    }
    if (disabled || !allowed || ctx->curr_stream_no != 0 || !is_mm || node->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 8; ++j) {
        const ggml_tensor * next = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(next) || (next->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        return next->op == node->op && next->src[1] == node->src[1] &&
            (node->op == GGML_OP_MUL_MAT || next->src[2] == node->src[2]);
    }
    return false;
}

static void ggml_cuda_src1_memo_end(ggml_backend_cuda_context * ctx) {
    ggml_cuda_src1_memo & memo = ctx->src1_memo;
    if (!memo.share_next) {
        memo.release();
    }
    memo.share_next = false;
}

// k11: a fused RMS_NORM + MUL at node i whose output the next matrix products convert (prompt batches): reserve the copy the
// first converting product would make and let the norm kernel write it (rms_norm_mul_conv_f32), see ggml_cuda_src1_memo.
// The products only use it when their own key matches, so a wrong guess here costs the extra writes, not correctness.
// GGML_CUDA_NORM_CONV=0 turns this part off.
static const ggml_tensor * ggml_cuda_src1_memo_base(const ggml_tensor * t) {
    while (t->view_src != nullptr && t->view_offs == 0 && ggml_nbytes(t) == ggml_nbytes(t->view_src)) {
        t = t->view_src;
    }
    return t;
}

static void ggml_cuda_src1_memo_plan_norm(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = (getenv("GGML_CUDA_SRC1_MEMO") != nullptr && atoi(getenv("GGML_CUDA_SRC1_MEMO")) == 0) ||
        (getenv("GGML_CUDA_NORM_CONV") != nullptr && atoi(getenv("GGML_CUDA_NORM_CONV")) == 0) ||
        getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") != nullptr;
    ggml_cuda_src1_memo & memo = ctx->src1_memo;
    memo.producer = nullptr;
    memo.produced = false;
    if (disabled || i + 1 >= cgraph->n_nodes || ctx->curr_stream_no != 0) {
        return;
    }
    const ggml_tensor * norm = cgraph->nodes[i];
    const ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm) ||
            mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || mul->data == nullptr) {
        return;
    }
    const int cc = ggml_cuda_info().devices[ctx->device].cc;
    for (int j = i + 2; j < cgraph->n_nodes && j <= i + 12; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT || ggml_cuda_src1_memo_base(node->src[1]) != mul) {
            return; // the copy is only shared by the products right after the norm
        }
        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src1 = node->src[1];
        if (!ggml_is_contiguous(src1) || src1->ne[2] != 1 || src1->ne[3] != 1 || (node->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) ||
                ggml_cuda_op_mul_mat_use_fwht(node)) {
            continue;
        }
        const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1];
        int     kind = 0, mode = 0;
        int64_t key[ggml_cuda_src1_memo::n_key] = {};
        size_t  nbytes = 0;
        if (ggml_is_quantized(src0->type) && src0->type != GGML_TYPE_NVFP4 && src0->type != GGML_TYPE_MXFP4 &&
                ne11 > MMVQ_MAX_BATCH_SIZE && ggml_cuda_should_use_mmq(src0->type, cc, ne11, 0)) {
            const mmq_q8_1_ds_layout layout = mmq_get_q8_1_ds_layout(src0->type);
            if (layout != MMQ_Q8_1_DS_LAYOUT_D4 && layout != MMQ_Q8_1_DS_LAYOUT_DS4) {
                continue;
            }
            const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
            if (ne10_padded != ne10) {
                continue;
            }
            kind = 1;
            mode = layout == MMQ_Q8_1_DS_LAYOUT_D4 ? 0 : 1;
            const int64_t k1[ggml_cuda_src1_memo::n_key] = { (int64_t) layout, ne10, ne11, 1, 1,
                (int64_t) src1->nb[1], (int64_t) src1->nb[2], (int64_t) src1->nb[3], ne10_padded, 0 };
            std::copy(k1, k1 + ggml_cuda_src1_memo::n_key, key);
            nbytes = ne11*ne10_padded/QK8_1_MMQ*sizeof(block_q8_1_mmq) + 128*sizeof(block_q8_1_mmq) + 64*sizeof(int);
        } else if (src0->type == GGML_TYPE_BF16 && fast_bf16_hardware_available(cc) && ne11 > MMVF_MAX_BATCH_SIZE &&
                ggml_get_op_params_i32(node, 0) == GGML_PREC_DEFAULT &&
                !ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, ne11, false)) {
            kind = 2;
            mode = 2;
            const int64_t k2[ggml_cuda_src1_memo::n_key] = { (int64_t) GGML_TYPE_BF16, ne10, ne11, 1, 1,
                (int64_t) src1->nb[1], (int64_t) src1->nb[2], (int64_t) src1->nb[3], 0, 0 };
            std::copy(k2, k2 + ggml_cuda_src1_memo::n_key, key);
            nbytes = ne10*ne11*sizeof(nv_bfloat16);
        } else {
            continue;
        }
        memo.release();
        memo.hold(ctx->pool(), kind, src1, key, nbytes);
        memo.producer      = mul;
        memo.producer_mode = mode;
        memo.K_out         = ne10;
        memo.N_out         = ne11;
        return;
    }
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            // q8_1 activation cache, see ggml_cuda_q8_1_cache; the arena lives for the whole evaluation
            ggml_cuda_pool_alloc<char> q8_1_cache_arena(cuda_ctx->pool());
            {
                std::unordered_set<const void *> norm_targets;
                std::unordered_map<const void *, int64_t> norm_target_ne10;
                const size_t slot_size = stream_ctx.concurrent_events.empty() ?
                    ggml_cuda_q8_1_cache_slot_size(cgraph, norm_targets, norm_target_ne10) : 0;
                if (slot_size > 0) {
                    cuda_ctx->q8_1_cache.begin(q8_1_cache_arena.alloc(ggml_cuda_q8_1_cache::n_slots*slot_size), slot_size);
                    cuda_ctx->q8_1_cache.norm_targets = std::move(norm_targets);
                    cuda_ctx->q8_1_cache.norm_target_ne10 = std::move(norm_target_ne10);
                }
            }

            // side-stream matrix-vector products of this graph (see ggml_cuda_side_mv_plan): node index -> join index
            std::unordered_map<int, int> side_mv_at;
            std::vector<std::pair<int, cudaEvent_t>> side_mv_pending; // join index, event
            if (stream_ctx.concurrent_events.empty() && ggml_cuda_side_mv_use(cuda_ctx->side_mv, use_cuda_graph && cuda_graph_update_required)) {
                std::unordered_map<const ggml_tensor *, int> idx;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    idx[cgraph->nodes[i]] = i;
                }
                for (const auto & [mv, reader] : cuda_ctx->side_mv) {
                    const auto a = idx.find(mv);
                    const auto b = idx.find(reader);
                    int j_inj = -1;
                    if (a != idx.end() && b != idx.end() && b->second - a->second >= GGML_CUDA_SIDE_MV_MIN_GAP &&
                        ggml_cuda_side_mv_run(mv) &&
                        // a hyper-connection inject that runs in its down projection's launch (ggml_cuda_try_fuse_hc_inject_down)
                        !ggml_cuda_hc_inject_down_match(cuda_ctx, cgraph, a->second, j_inj, nullptr, nullptr, nullptr)) {
                        side_mv_at[a->second] = b->second;
                    }
                }
                if (!side_mv_at.empty() && cuda_ctx->side_mv_fork == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx->side_mv_fork, cudaEventDisableTiming));
                }
            }
            size_t side_mv_n_events = 0;
            const auto side_mv_join_until = [&](int last) {
                for (size_t k = 0; k < side_mv_pending.size();) {
                    if (side_mv_pending[k].first <= last) {
                        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(cuda_ctx->device, 0), side_mv_pending[k].second, 0));
                        side_mv_pending.erase(side_mv_pending.begin() + k);
                    } else {
                        ++k;
                    }
                }
            };

            cuda_ctx->mmvq_done_early.clear();
            cuda_ctx->gdn_state_in_place.clear();
            cuda_ctx->gdn_gates_in.clear();
            cuda_ctx->conv_state_pending.clear();
            cuda_ctx->conv_cat_elided.clear();
            cuda_ctx->unary_cont_elided.clear();
            ggml_cuda_qsa_maskless_plan(cuda_ctx, cgraph);
            ggml_cuda_qsa_sel_plan(cuda_ctx, cgraph);
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (!side_mv_pending.empty()) {
                    // a fused group starting here may cover the reader, so join a few nodes early
                    side_mv_join_until(i + GGML_CUDA_SIDE_MV_LOOKAHEAD);
                }
                if (!side_mv_at.empty()) {
                    const auto it = side_mv_at.find(i);
                    if (it != side_mv_at.end()) {
                        if (side_mv_n_events == cuda_ctx->side_mv_join.size()) {
                            cudaEvent_t ev;
                            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                            cuda_ctx->side_mv_join.push_back(ev);
                        }
                        cudaEvent_t join = cuda_ctx->side_mv_join[side_mv_n_events++];
                        cudaStream_t side = cuda_ctx->stream(cuda_ctx->device, GGML_CUDA_SIDE_MV_STREAM);
                        CUDA_CHECK(cudaEventRecord(cuda_ctx->side_mv_fork, cuda_ctx->stream(cuda_ctx->device, 0)));
                        CUDA_CHECK(cudaStreamWaitEvent(side, cuda_ctx->side_mv_fork, 0));
                        cuda_ctx->src1_memo.release();
                        cuda_ctx->src1_memo.share_next = false;
                        cuda_ctx->curr_stream_no = GGML_CUDA_SIDE_MV_STREAM;
                        const bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                        GGML_ASSERT(ok);
                        cuda_ctx->curr_stream_no = 0;
                        CUDA_CHECK(cudaEventRecord(join, side));
                        side_mv_pending.push_back({ it->second, join });
                        ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                        prev_i = i;
                        continue;
                    }
                }
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                if (!cuda_ctx->qsa_skip.empty() && cuda_ctx->qsa_skip.count(node)) {
                    // a QSA selection mask node, its flash attention reads the selection instead (ggml_cuda_qsa_maskless_plan)
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }

                if (!cuda_ctx->qsa_sel_skip.empty() && cuda_ctx->qsa_sel_skip.count(node)) {
                    // a node of a QSA selection chain, computed by the launch of the chain's index node (ggml_cuda_qsa_sel_plan)
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }
                if (!cuda_ctx->qsa_sel.empty()) {
                    auto it_sel = cuda_ctx->qsa_sel.find(node);
                    if (it_sel != cuda_ctx->qsa_sel.end()) {
                        ggml_cuda_qsa_sel_launch(cuda_ctx, node, it_sel->second);
                        ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                        continue;
                    }
                }

                if (!cuda_ctx->mmvq_done_early.empty() && cuda_ctx->mmvq_done_early.count(node)) {
                    // computed by an earlier multi-matrix mul_mat_vec_q launch (ggml_cuda_try_fuse_mmvq_multi)
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }

                if (node->op == GGML_OP_CONCAT && ggml_cuda_try_conv_cat_elide(cuda_ctx, cgraph, i)) {
                    // the short convolution that reads this concat reads its two parts (ssm_conv_cat_f32)
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }

                if (node->op == GGML_OP_CPY && ggml_cuda_try_conv_state_defer(cuda_ctx, cgraph, i)) {
                    // the short convolution that reads the same input writes this copy (ssm_conv_state_copies)
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }

                if (node->op == GGML_OP_CONT && ggml_cuda_try_unary_src_cont_elide(cuda_ctx, cgraph, i)) {
                    // the fused UNARY + MUL that reads this CONT reads its source view (unary_gated_op_kernel, x_split)
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }

                if (node->op == GGML_OP_UNARY && !cuda_ctx->unary_cont_elided.empty() && cuda_ctx->unary_cont_elided.count(node->src[0])) {
                    const ggml_tensor * c = node->src[0];
                    cuda_ctx->unary_cont_elided.erase(c);
                    const ggml_unary_op op = ggml_get_unary_op(node);
                    if (i + 1 < cgraph->n_nodes && ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { op })) {
                        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1], c->src[0]);
                        ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 2);
                        cuda_ctx->q8_1_cache.commit(); // the q8_1 copy of the MUL output the fused kernel wrote, if any
                        i += 1;
                        continue;
                    }
                    static int n_warn = 0;
                    if (n_warn++ < 4) {
                        GGML_LOG_WARN("%s: elided CONT %s: %s is not fused with the next node (%s): the CONT runs (not expected)\n", __func__, c->name,
                            node->name, i + 1 < cgraph->n_nodes ? ggml_op_desc(cgraph->nodes[i + 1]) : "none");
                    }
                    ggml_cuda_dup(*cuda_ctx, const_cast<ggml_tensor *>(c));
                }

                if (!cuda_ctx->conv_cat_elided.empty() || !cuda_ctx->unary_cont_elided.empty()) {
                    ggml_cuda_conv_cat_materialize(cuda_ctx, node);
                }

                if (node->op == GGML_OP_FILL) {
                    const int n_pack = ggml_cuda_try_fuse_qsa_key_pack(cuda_ctx, cgraph, i);
                    if (n_pack > 0) {
                        // a QSA layer's packed indexer key row, written into the cache (k_qsa_key_pack)
                        ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, n_pack);
                        i += n_pack - 1;
                        continue;
                    }
                }

                if (node->op == GGML_OP_GET_ROWS) {
                    const int n_pool = ggml_cuda_try_fuse_qsa_pool(cuda_ctx, cgraph, i);
                    if (n_pool > 0) {
                        // a QSA layer's new pooled keys, gathered to scattered (k_qsa_pool)
                        ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, n_pool);
                        i += n_pool - 1;
                        continue;
                    }
                }

                if (node->op == GGML_OP_GET_ROWS && ggml_cuda_try_gdn_state_in_place(cuda_ctx, cgraph, i)) {
                    // the gated_delta_net that reads this row reads it from the cache instead
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                    continue;
                }

                if (node->op == GGML_OP_UNARY || node->op == GGML_OP_ADD) {
                    const int n_gate = ggml_cuda_try_gdn_gates(cuda_ctx, cgraph, i);
                    if (n_gate > 0) {
                        // the gated_delta_net that reads this gate computes it itself
                        ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, n_gate);
                        i += n_gate - 1;
                        continue;
                    }
                }

                // the share flag is set only for the unfused node: a fused group may hold pool memory around its products
                const bool memo_share = ggml_cuda_src1_memo_begin(cuda_ctx, cgraph, i, !is_concurrent_event_active && stream_ctx.concurrent_events.empty());

                if (node->op == GGML_OP_RMS_NORM && !is_concurrent_event_active && stream_ctx.concurrent_events.empty()) {
                    ggml_cuda_src1_memo_plan_norm(cuda_ctx, cgraph, i);
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                bool fused_multi = false; // k15-mm: a multi-matrix mul_mat_vec_q launch took node i (and maybe nodes after it)
                if (nodes_to_skip == 0 && node->op == GGML_OP_MUL_MAT && !is_concurrent_event_active) {
                    int run = ggml_cuda_try_fuse_mmvq_multi(cuda_ctx, cgraph, i);
                    if (run < 0) {
                        run = ggml_cuda_try_fuse_mmvf_multi(cuda_ctx, cgraph, i);
                    }
                    if (run >= 0) {
                        fused_multi   = true;
                        nodes_to_skip = run;
                    }
                }

                {
                    // a copy reserved for the norm and not written (another fusion ran, or the kernel did not apply) is dropped
                    ggml_cuda_src1_memo & memo = cuda_ctx->src1_memo;
                    const bool produced = memo.producer != nullptr && memo.produced && (nodes_to_skip != 0 || fused_multi);
                    if (memo.producer != nullptr && !produced) {
                        memo.release();
                    }
                    if (produced) {
                        memo.n_produced++;
                    }
                    memo.producer = nullptr;
                    memo.produced = false;
                    if ((nodes_to_skip != 0 || fused_multi) && !produced) {
                        memo.release();
                    }
                }

                if (nodes_to_skip != 0 || fused_multi) {
                    cuda_ctx->src1_memo.share_next = false;
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, nodes_to_skip + 1);
                    cuda_ctx->q8_1_cache.commit(); // q8_1 copy written by the fused group itself, if any
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                cuda_ctx->src1_memo.share_next = memo_share;
                cuda_ctx->src1_memo.node_dst   = node;
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                cuda_ctx->src1_memo.node_dst   = nullptr;
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                ggml_cuda_src1_memo_end(cuda_ctx);

                ggml_cuda_q8_1_cache_node_done(cuda_ctx->q8_1_cache, cgraph, i, 1);
                cuda_ctx->q8_1_cache.commit(); // q8_1 copy written by the node itself (DSV4_HC_PRE), if any

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            // a capture must not end with unjoined work on the side stream
            side_mv_join_until(INT_MAX);
            cuda_ctx->src1_memo.release();
            cuda_ctx->src1_memo.share_next = false;
            cuda_ctx->q8_1_cache.end();
            cuda_ctx->mmvq_done_early.clear();
            cuda_ctx->gdn_state_in_place.clear();
            cuda_ctx->gdn_gates_in.clear();
            ggml_cuda_conv_state_flush(cuda_ctx);
            cuda_ctx->conv_cat_elided.clear();
            cuda_ctx->unary_cont_elided.clear();
        }

        // an overlapped MoE cache fetch whose join is not in this graph (another split): the side stream joins the main stream
        // here, so a CUDA graph capture never ends with an unjoined stream and the next graph starts after the copies
        if (cuda_ctx->moe_join_pending) {
            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), cuda_ctx->moe_join, 0));
            cuda_ctx->moe_join_pending = false;
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured
            // k15-fn debug (GGML_CUDA_GRAPH_NODE_STATS=1): the node types of each captured graph (a host<->device memcpy node makes the
            // launch cost grow with the node count: B200 2000 kernels 5.7 us, with memcpy nodes 102 us)
            static const bool node_stats = getenv("GGML_CUDA_GRAPH_NODE_STATS") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_NODE_STATS")) != 0;
            if (node_stats) {
                // =2: also the copy / gather ops of the ggml graph by op and tensor name (layer numbers stripped)
                if (atoi(getenv("GGML_CUDA_GRAPH_NODE_STATS")) >= 2) {
                    std::map<std::string, int> ops;
                    for (int i = 0; i < cgraph->n_nodes; ++i) {
                        const ggml_tensor * t = cgraph->nodes[i];
                        if (t->op != GGML_OP_CPY && t->op != GGML_OP_CONT && t->op != GGML_OP_DUP && t->op != GGML_OP_CONCAT &&
                            t->op != GGML_OP_GET_ROWS && t->op != GGML_OP_SET_ROWS && t->op != GGML_OP_REPEAT) {
                            continue;
                        }
                        std::string nm;
                        for (const char * c = t->name; *c; ++c) {
                            if (*c < '0' || *c > '9') { nm += *c; }
                        }
                        char key[256];
                        const ggml_tensor * s0 = t->src[0];
                        snprintf(key, sizeof(key), "%s %s [%lld,%lld,%lld] %s%s <- %s", ggml_op_name(t->op), nm.c_str(), (long long) t->ne[0], (long long) t->ne[1],
                            (long long) t->ne[2], ggml_type_name(t->type), s0 && !ggml_is_contiguous(s0) ? " (src non-contig)" : "", s0 ? ggml_op_name(s0->op) : "-");
                        ops[key]++;
                    }
                    for (auto & [k, v] : ops) {
                        GGML_LOG_INFO("%s:   copy/gather op x%d: %s\n", __func__, v, k.c_str());
                    }
                }
                for (auto & [k, v] : ggml_cuda_exec_copy_ops[cuda_ctx]) {
                    GGML_LOG_INFO("%s:   executed copy/gather x%d (%.2f MiB): %s\n", __func__, v.first, v.second/1048576.0, k.c_str());
                }
                ggml_cuda_exec_copy_ops[cuda_ctx].clear();
                size_t n_nodes = 0;
                CUDA_CHECK(cudaGraphGetNodes(graph->graph, nullptr, &n_nodes));
                std::vector<cudaGraphNode_t> nodes(n_nodes);
                CUDA_CHECK(cudaGraphGetNodes(graph->graph, nodes.data(), &n_nodes));
                int cnt[16] = { 0 };
                int n_h2d = 0, n_d2h = 0, n_d2d = 0;
                for (cudaGraphNode_t nd : nodes) {
                    cudaGraphNodeType t;
                    CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
                    cnt[std::min((int) t, 15)]++;
                    if (t == cudaGraphNodeTypeMemcpy) {
                        cudaMemcpy3DParms p = {};
                        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(nd, &p));
                        cudaPointerAttributes as = {}, ad = {};
                        cudaPointerGetAttributes(&as, p.srcPtr.ptr);
                        cudaPointerGetAttributes(&ad, p.dstPtr.ptr);
                        (void) cudaGetLastError();
                        const bool sd = as.type == cudaMemoryTypeDevice, dd = ad.type == cudaMemoryTypeDevice;
                        n_h2d += !sd && dd; n_d2h += sd && !dd; n_d2d += sd && dd;
                    }
                }
                GGML_LOG_INFO("%s: captured graph of %d ggml nodes: %zu CUDA nodes, %d kernels, %d memcpy (%d H2D, %d D2H, %d D2D), %d memset, %d host, %d event record, %d event wait, %d empty, %d other\n",
                    __func__, cgraph->n_nodes, n_nodes, cnt[cudaGraphNodeTypeKernel], cnt[cudaGraphNodeTypeMemcpy], n_h2d, n_d2h, n_d2d,
                    cnt[cudaGraphNodeTypeMemset], cnt[cudaGraphNodeTypeHost], cnt[cudaGraphNodeTypeEventRecord], cnt[cudaGraphNodeTypeWaitEvent],
                    cnt[cudaGraphNodeTypeEmpty], (int) n_nodes - cnt[cudaGraphNodeTypeKernel] - cnt[cudaGraphNodeTypeMemcpy] - cnt[cudaGraphNodeTypeMemset] -
                    cnt[cudaGraphNodeTypeHost] - cnt[cudaGraphNodeTypeEventRecord] - cnt[cudaGraphNodeTypeWaitEvent] - cnt[cudaGraphNodeTypeEmpty]);
            }

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        {
            // k15-fn debug (GGML_CUDA_HOST_TIMING=N): the host time of the launch call alone, graphs of over 1000 ggml nodes apart
            static const int lt_every = getenv("GGML_CUDA_HOST_TIMING") ? atoi(getenv("GGML_CUDA_HOST_TIMING")) : 0;
            static int64_t lt_n[2] = { 0, 0 }, lt_t[2] = { 0, 0 }, lt_nodes[2] = { 0, 0 };
            const int64_t lt0 = lt_every > 0 ? ggml_time_us() : 0;
            CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
            if (lt_every > 0) {
                const int bk = cgraph->n_nodes > 1000;
                lt_n[bk]++; lt_t[bk] += ggml_time_us() - lt0; lt_nodes[bk] += cgraph->n_nodes;
                if (lt_n[bk] % lt_every == 0) {
                    GGML_LOG_INFO("%s: cudaGraphLaunch over the last %lld launches of %s graphs: %.1f us per launch, %.0f ggml nodes\n", __func__,
                        (long long) lt_n[bk], bk ? "large" : "small", (double) lt_t[bk] / lt_n[bk], (double) lt_nodes[bk] / lt_n[bk]);
                    lt_n[bk] = lt_t[bk] = lt_nodes[bk] = 0;
                }
            }
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

// k15-fn debug (GGML_CUDA_HOST_TIMING=N): host time of the graph_compute phases, a summary every N calls with at least 64 nodes
struct ggml_cuda_host_timing {
    int every = 0;
    int64_t n = 0, t_check = 0, t_eval = 0, n_graph = 0, n_nodes = 0;
    ggml_cuda_host_timing() { const char * e = getenv("GGML_CUDA_HOST_TIMING"); every = e ? atoi(e) : 0; }
};
static ggml_cuda_host_timing & ggml_cuda_host_timing_get() { static ggml_cuda_host_timing t; return t; }

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_host_timing & ht = ggml_cuda_host_timing_get();
    const bool ht_on = ht.every > 0 && cgraph->n_nodes >= 64;
    const int64_t ht0 = ht_on ? ggml_time_us() : 0;
    int64_t ht1 = ht0;

    ggml_cuda_set_device(cuda_ctx->device);
    cuda_ctx->computes = true;

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
        ggml_cuda_exec_copy_ops[cuda_ctx].clear(); // k15-fn debug: the copies of this capture only
    }

    if (ht_on) {
        ht1 = ggml_time_us();
    }
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    if (ht_on) {
        const int64_t ht2 = ggml_time_us();
        ht.n++; ht.t_check += ht1 - ht0; ht.t_eval += ht2 - ht1; ht.n_graph += use_cuda_graph; ht.n_nodes += cgraph->n_nodes;
        if (ht.n % ht.every == 0) {
            GGML_LOG_INFO("%s: host timing over the last %lld calls: check %.1f us, evaluate/launch %.1f us per call, %.0f%% CUDA graphs, %.0f nodes\n",
                __func__, (long long) ht.n, (double) ht.t_check / ht.n, (double) ht.t_eval / ht.n, 100.0 * ht.n_graph / ht.n, (double) ht.n_nodes / ht.n);
            ht.n = ht.t_check = ht.t_eval = ht.n_graph = ht.n_nodes = 0;
        }
    }
    ggml_cuda_verify_mmq_summary();

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_side_mv_plan(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    // the last node a candidate reads is found by a scan back from it (the first node that is one of its sources: a node appears once
    // in the graph), not from a map of every node kept up to date over the moves (0.4 ms at 4k nodes, rebuilt every graph).
    // opt-in: GGML_CUDA_SIDE_MV_SCAN=1 (default: the map, as before)
    static const bool scan = [] {
        const char * e = getenv("GGML_CUDA_SIDE_MV_SCAN");
        return e != nullptr && atoi(e) != 0;
    }();
    std::unordered_map<const ggml_tensor *, int> idx;
    if (!scan) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            idx[cgraph->nodes[i]] = i;
        }
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!ggml_cuda_side_mv_candidate(node)) {
            continue;
        }
        int p = -1; // the last node it reads
        if (scan) {
            for (int j = i - 1; j >= 0 && p < 0; --j) {
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    if (node->src[k] != nullptr && cgraph->nodes[j] == node->src[k]) {
                        p = j;
                        break;
                    }
                }
            }
        } else {
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                const auto it = node->src[k] ? idx.find(node->src[k]) : idx.end();
                if (it != idx.end()) {
                    p = std::max(p, it->second);
                }
            }
        }
        int c = -1; // its first reader
        for (int j = i + 1; j < cgraph->n_nodes && c < 0; ++j) {
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                if (cgraph->nodes[j]->src[k] == node) {
                    c = j;
                    break;
                }
            }
        }
        if (c < 0 || (node->flags & GGML_TENSOR_FLAG_OUTPUT) || c - p < GGML_CUDA_SIDE_MV_MIN_GAP) {
            continue;
        }
        if (i > p + 1) {
            std::rotate(cgraph->nodes + p + 1, cgraph->nodes + i, cgraph->nodes + i + 1);
            if (!scan) {
                for (int j = p + 1; j <= i; ++j) {
                    idx[cgraph->nodes[j]] = j;
                }
            }
        }
        // the main stream may reuse the memory of the sources once their last reader in the node order is done
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (node->src[k] != nullptr) {
                params->add_alloc_dep(params->user_data, node->src[k], cgraph->nodes[c]);
            }
        }
        params->add_alloc_dep(params->user_data, node, cgraph->nodes[c]);
        cuda_ctx->side_mv[node] = cgraph->nodes[c];
    }
}

// k16-race: see ggml_backend_cuda_graph_optimize. Returns the last node the fused kernel writes (nullptr: no match) and the inputs it reads
static const ggml_tensor * ggml_cuda_fused_match(const ggml_cgraph * cgraph, int i, std::vector<ggml_tensor *> & inputs) {
    static const bool all_alias = ggml_cuda_env_flag("GGML_CUDA_FUSED_ALIAS");
    if (all_alias) {
        return nullptr;
    }
    ggml_tensor * n0 = cgraph->nodes[i];
    inputs.clear();

    // residual add + RMS norm + mul (+ q8_1), ggml_cuda_try_fuse_add_rms_norm (rms_norm_mul_q8_1_f32 with ra / rb): both add inputs.
    // Qwen3.6-35B-A3B MTP spilled (B200): the norm output landed on the previous layer's l_out, the text was not repeatable.
    // GGML_CUDA_ADD_RMS_ALIAS=1
    static const bool add_rms_alias = ggml_cuda_env_flag("GGML_CUDA_ADD_RMS_ALIAS");
    if (!add_rms_alias && n0->op == GGML_OP_ADD && n0->type == GGML_TYPE_F32 && i + 2 < cgraph->n_nodes) {
        ggml_tensor * rms = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        if (rms->op == GGML_OP_RMS_NORM && rms->src[0] == n0 && mul->op == GGML_OP_MUL && (mul->src[0] == rms || mul->src[1] == rms) &&
                mul->type == GGML_TYPE_F32) {
            for (int k = 0; k < 2; ++k) {
                if (n0->src[k] != nullptr) {
                    inputs.push_back(n0->src[k]);
                }
            }
            return mul;
        }
    }

    // DSV4_HC_POST computed inside the grouped norm that reads it (+ mul, q8_1), ggml_cuda_try_fuse_hc_post_norm: block output, residual
    // and the post weights (the input of the SCALE / SIGMOID chain the kernel applies). Flash-Next f0: not repeatable.
    // GGML_CUDA_HC_POST_NORM_ALIAS=1
    static const bool hc_alias = ggml_cuda_env_flag("GGML_CUDA_HC_POST_NORM_ALIAS");
    if (!hc_alias && n0->op == GGML_OP_DSV4_HC_POST && n0->src[3] == nullptr) {
        int irn = i + 1;
        while (irn < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[irn])) {
            irn++;
        }
        if (irn + 1 < cgraph->n_nodes) {
            ggml_tensor * rn  = cgraph->nodes[irn];
            ggml_tensor * mul = cgraph->nodes[irn + 1];
            if (rn->op == GGML_OP_RMS_NORM && rn->src[0] == n0 && mul->op == GGML_OP_MUL && (mul->src[0] == rn || mul->src[1] == rn)) {
                for (int k = 0; k < 2; ++k) {
                    if (n0->src[k] != nullptr) {
                        inputs.push_back(n0->src[k]);
                    }
                }
                ggml_tensor * w = n0->src[2];
                for (int k = 0; w != nullptr && k <= GGML_CUDA_HC_POST_NORM_MAX_OPS; ++k) {
                    inputs.push_back(w);
                    const bool chain = w->op == GGML_OP_SCALE || (w->op == GGML_OP_UNARY && ggml_get_unary_op(w) == GGML_UNARY_OP_SIGMOID);
                    w = chain ? w->src[0] : nullptr;
                }
                return mul;
            }
        }
    }

    // QSA dead-slot index map CPY(i32 -> f32) -> SUB -> MUL -> ADD -> CPY(f32 -> i32) in one kernel (ggml_cuda_try_fuse_qsa_idx, k15-fn):
    // the index input, dump and live. G4 fn4 32 GB: the map's output landed in memory freed inside the chain at an offset, the 8k md5
    // varied (k15-fn stages the overlapping case in one block; with the inputs kept alive it does not occur). GGML_CUDA_QSA_IDX_ALIAS=1
    static const bool qsa_alias = ggml_cuda_env_flag("GGML_CUDA_QSA_IDX_ALIAS");
    if (!qsa_alias && n0->op == GGML_OP_CPY && n0->type == GGML_TYPE_F32 && n0->src[0] != nullptr && n0->src[0]->type == GGML_TYPE_I32 &&
            i + 4 < cgraph->n_nodes) {
        ggml_tensor * sub = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        ggml_tensor * add = cgraph->nodes[i + 3];
        ggml_tensor * c2  = cgraph->nodes[i + 4];
        if (sub->op == GGML_OP_SUB && sub->src[0] == n0 && mul->op == GGML_OP_MUL && mul->src[0] == sub && add->op == GGML_OP_ADD &&
                add->src[0] == mul && add->src[1] == sub->src[1] && c2->op == GGML_OP_CPY && c2->src[0] == add && c2->type == GGML_TYPE_I32) {
            inputs = { n0->src[0], sub->src[1], mul->src[1] };
            return c2;
        }
    }

    // QSA selection chain in one staged block at its index node (ggml_cuda_qsa_sel_plan, k15-fn): the chain's inputs (top_k, pool
    // members, tail cells, scores) stay allocated until the index node is, so no node that still runs between the chain's first skipped
    // reader and the index node is placed over them (the plan then never has to refuse a chain for it). GGML_CUDA_QSA_SEL_ALIAS=1
    static const bool qsa_sel_alias = ggml_cuda_env_flag("GGML_CUDA_QSA_SEL_ALIAS");
    if (!qsa_sel_alias && n0->op == GGML_OP_CPY && n0->type == GGML_TYPE_I32) {
        ggml_backend_cuda_context::qsa_sel_args a = {};
        std::vector<const ggml_tensor *> chain, ins;
        if (ggml_cuda_qsa_sel_match(cgraph, n0, a, chain, ins)) {
            for (const ggml_tensor * t : ins) {
                inputs.push_back(const_cast<ggml_tensor *>(t));
            }
            return n0;
        }
    }

    // short convolution (+ bias ADD) + SILU in one launch, which also writes the deferred conv-state copies (ggml_cuda_op_ssm_conv,
    // ggml_cuda_try_conv_state_defer, k15-fn): the conv input and the weights stay off the SILU output (the allocator could place it in
    // the input freed after the convolution: the fused kernel reads the input while it writes). GGML_CUDA_CONV_STATE_ALIAS=1
    static const bool conv_alias = ggml_cuda_env_flag("GGML_CUDA_CONV_STATE_ALIAS");
    if (!conv_alias && n0->op == GGML_OP_SSM_CONV && i + 1 < cgraph->n_nodes) {
        ggml_tensor * nx = cgraph->nodes[i + 1];
        ggml_tensor * bias = nullptr;
        if (nx->op == GGML_OP_ADD && (nx->src[0] == n0 || nx->src[1] == n0) && i + 2 < cgraph->n_nodes) {
            bias = nx->src[0] == n0 ? nx->src[1] : nx->src[0];
            nx = cgraph->nodes[i + 2];
        }
        if (nx->op == GGML_OP_UNARY && ggml_get_unary_op(nx) == GGML_UNARY_OP_SILU && nx->src[0] != nullptr &&
                (nx->src[0] == n0 || nx->src[0] == cgraph->nodes[i + 1])) {
            inputs = { n0->src[0], n0->src[1] };
            if (bias != nullptr) {
                inputs.push_back(bias);
            }
            return nx;
        }
    }

    // QSA pooled keys in one launch (ggml_cuda_try_fuse_qsa_pool, k15-fn): the member ids, the norm weight, the rope positions and
    // factors and the scatter rows stay allocated until the scatter is (the kernel writes only the cache). GGML_CUDA_QSA_POOL_ALIAS=1
    static const bool pool_alias = ggml_cuda_env_flag("GGML_CUDA_QSA_POOL_ALIAS");
    if (!pool_alias && n0->op == GGML_OP_GET_ROWS) {
        ggml_cuda_qsa_pool_chain m;
        if (ggml_cuda_qsa_pool_match(cgraph, i, m, /*runtime=*/false) >= 0) {
            inputs = { m.get_rows->src[1], m.mul->src[0] == m.rms_norm ? m.mul->src[1] : m.mul->src[0], m.rope->src[1],
                       m.set_rows->src[1] };
            if (m.rope->src[2] != nullptr) {
                inputs.push_back(m.rope->src[2]);
            }
            return m.set_rows;
        }
    }

    // QSA packed indexer key row in one kernel at its FILL (ggml_cuda_try_fuse_qsa_key_pack, k15-fn): the raw key and the rows stay
    // allocated until the scatter is. GGML_CUDA_QSA_KEY_PACK_ALIAS=1
    static const bool pack_alias = ggml_cuda_env_flag("GGML_CUDA_QSA_KEY_PACK_ALIAS");
    if (!pack_alias && n0->op == GGML_OP_FILL) {
        ggml_cuda_qsa_key_pack m;
        if (ggml_cuda_qsa_key_pack_match(cgraph, i, m, /*runtime=*/false) >= 0) {
            inputs = { m.concat->src[0], m.set_rows->src[1] };
            return m.set_rows;
        }
    }

    // CONT of a strided view read by its fused UNARY + MUL instead (ggml_cuda_try_unary_src_cont_elide, k15-fn): the view (and so the
    // tensor it views) stays allocated until the MUL is. GGML_CUDA_UNARY_SRC_CONT_ALIAS=1
    static const bool cont_alias = ggml_cuda_env_flag("GGML_CUDA_UNARY_SRC_CONT_ALIAS");
    if (!cont_alias && n0->op == GGML_OP_CONT) {
        const int u = ggml_cuda_unary_src_cont_match(cgraph, i, /*runtime=*/false);
        if (u >= 0) {
            inputs = { n0->src[0] };
            return cgraph->nodes[u + 1];
        }
    }

    // conv-input CONCAT read by its short convolution instead of run (ggml_cuda_try_conv_cat_elide, k15-fn): the state columns and the
    // new tokens stay allocated until the convolution's last output is, so no node between them, nor the convolution's output, is
    // placed over them (the runtime check then never has to refuse it). GGML_CUDA_CONV_CAT_ALIAS=1
    static const bool cat_alias = ggml_cuda_env_flag("GGML_CUDA_CONV_CAT_ALIAS");
    if (!cat_alias && n0->op == GGML_OP_CONCAT && ggml_get_op_params_i32(n0, 0) == 0 && n0->type == GGML_TYPE_F32 && n0->src[1] != nullptr &&
            n0->src[1]->ne[0] <= 8) {
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + 24; ++j) {
            ggml_tensor * conv = cgraph->nodes[j];
            if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != n0) {
                continue;
            }
            ggml_tensor * last = conv;
            int k = j + 1;
            if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_ADD && (cgraph->nodes[k]->src[0] == conv || cgraph->nodes[k]->src[1] == conv)) {
                last = cgraph->nodes[k++];
            }
            if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[k]) == GGML_UNARY_OP_SILU &&
                    cgraph->nodes[k]->src[0] == last) {
                last = cgraph->nodes[k];
            }
            inputs = { n0->src[0], n0->src[1] };
            return last;
        }
    }
    return nullptr;
}

static void ggml_cuda_fused_keep_inputs(ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    std::vector<ggml_tensor *> inputs;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * last = ggml_cuda_fused_match(cgraph, i, inputs);
        if (last == nullptr) {
            continue;
        }
        for (ggml_tensor * in : inputs) {
            if (in != nullptr) {
                params->add_alloc_dep(params->user_data, in, const_cast<ggml_tensor *>(last));
            }
        }
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    const bool disable_fusion = ggml_cuda_fusion_disabled();

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    // the side-stream matrix-vector products of the previous graph do not apply to this one
    cuda_ctx->side_mv.clear();

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                // the routed experts of the next layer; a second chain over the same input (the experts fetched by an overlapped
                // MoE cache lookup, see build_moe_ffn) is skipped, so the shared expert is computed with the first one, beside the copies
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID &&
                        cgraph->nodes[j]->src[1] != cgraph->nodes[i]->src[1]) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

    // k16-race: fused kernels whose outputs must not land on the inputs they read. Each matcher below follows the op pattern its
    // fusion takes (ggml_cuda_try_fuse_*) and names the tensors the fused kernel reads and the last node it writes; graph_optimize keeps
    // those inputs alive until that node is allocated, so the allocator places no output (also none in place) over them. The separate
    // kernels each finish reading before the next writes; a fused kernel writes while it (or the kernels around it, with PDL) still
    // reads. Same kernels and values, only the compute buffer holds the inputs a few nodes longer.
    // Kill switches: GGML_CUDA_FUSED_ALIAS=1 (all), or per fusion as listed with each matcher
    if (!disable_fusion) {
        ggml_cuda_fused_keep_inputs(cgraph, params);
    }

    {
        static const bool side_mv_disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_SIDE_MATVEC");
        // GGML_CUDA_GRAPH_OPT=1 runs its own concurrent streams
        const char * graph_opt = getenv("GGML_CUDA_GRAPH_OPT");
        // the side stream only pays where a few-row product leaves SMs idle long enough to cover the fork/join events:
        // measured slower on Turing (T4), so it is on for NVIDIA Ampere and newer only (GGML_CUDA_DISABLE_SIDE_MATVEC=1 still turns it off)
        const int side_mv_cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        const bool side_mv_arch_ok = GGML_CUDA_CC_IS_NVIDIA(side_mv_cc) && side_mv_cc >= GGML_CUDA_CC_AMPERE;
        if (!disable_fusion && !side_mv_disabled && side_mv_arch_ok && !(graph_opt != nullptr && atoi(graph_opt) == 1)) {
            ggml_cuda_side_mv_plan(cuda_ctx, cgraph, params);
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

// maps host memory into the device address space so kernels read it directly (zero-copy), returns the device pointer or nullptr
// mappings are reference counted: if *registered is set, call ggml_backend_cuda_host_unmap(buffer) when done
// pageable memory is never read through HMM/ATS, it is an order of magnitude slower
// the device pointer is taken per device: HIP maps registered memory into the address space of the current device
struct ggml_cuda_host_mapping {
    size_t size;
    void * dev_ptr[GGML_CUDA_MAX_DEVICES];
    int    refs;
};

static std::mutex & ggml_cuda_host_mappings_mutex() {
    static std::mutex m;
    return m;
}

static std::unordered_map<void *, ggml_cuda_host_mapping> & ggml_cuda_host_mappings() {
    static std::unordered_map<void *, ggml_cuda_host_mapping> m;
    return m;
}

static void * ggml_cuda_host_map(int device, void * buffer, size_t size, bool * registered) {
    *registered = false;
    void * dev_ptr = nullptr;

    std::lock_guard<std::mutex> lock(ggml_cuda_host_mappings_mutex());
    auto & mappings = ggml_cuda_host_mappings();

    ggml_cuda_set_device(device);

    // registered here before: one more reference, with the device pointer of this device
    auto it = mappings.find(buffer);
    if (it != mappings.end()) {
        if (size > it->second.size) {
            return nullptr;
        }
        if (it->second.dev_ptr[device] == nullptr) {
            if (cudaHostGetDevicePointer(&dev_ptr, buffer, 0) != cudaSuccess || dev_ptr == nullptr) {
                (void)cudaGetLastError();
                return nullptr;
            }
            it->second.dev_ptr[device] = dev_ptr;
        }
        it->second.refs++;
        *registered = true;
        return it->second.dev_ptr[device];
    }

    // pinned by someone else, for example a CUDA host buffer (registering it again fails): not owned here
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, buffer) == cudaSuccess && attr.type == cudaMemoryTypeHost) {
        if (cudaHostGetDevicePointer(&dev_ptr, buffer, 0) == cudaSuccess && dev_ptr != nullptr) {
            return dev_ptr;
        }
    }
    (void)cudaGetLastError();

    // read-only registration also accepts read-only mappings (mmap), but not every device supports it
    const unsigned int flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
    bool read_only_ok = false;
#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    read_only_ok = cudaHostRegister(buffer, size, flags | cudaHostRegisterReadOnly) == cudaSuccess;
    if (!read_only_ok) {
        (void)cudaGetLastError();
    }
#endif
    cudaError_t err = read_only_ok ? cudaSuccess : cudaHostRegister(buffer, size, flags);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: failed to map %.2f MiB of host memory: %s\n", __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    if (cudaHostGetDevicePointer(&dev_ptr, buffer, 0) != cudaSuccess || dev_ptr == nullptr) {
        (void)cudaGetLastError();
        cudaHostUnregister(buffer);
        (void)cudaGetLastError();
        return nullptr;
    }
    ggml_cuda_host_mapping & m = mappings[buffer];
    m = {};
    m.size = size;
    m.dev_ptr[device] = dev_ptr;
    m.refs = 1;
    *registered = true;
    return dev_ptr;
}

// maps the pages of [ptr, ptr + size) (a part of a host buffer, e.g. the expert tensors of a memory-mapped model file) read-only into
// the device address space: the pages at either end that a mapping of this module already holds are skipped (cudaHostRegister refuses
// any overlap, and neighbouring tensors share their boundary pages), so [ptr, ptr + size) may span up to three mappings. Returns ptr
// as the device pointer (only where the device reads registered memory at its host address, else nullptr) and in *base the key of the
// mapping that holds the reference to release with ggml_backend_cuda_host_unmap (*registered); nullptr if any part could not be mapped
static void * ggml_cuda_host_map_range(int device, void * ptr, size_t size, void ** base, bool * registered) {
    *registered = false;
    *base       = nullptr;
    if (ptr == nullptr || size == 0) {
        return nullptr;
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    // not validated there: the whole buffers are mapped or nothing
    GGML_UNUSED(device);
    return nullptr;
#else
    int can_host_ptr = 0;
    if (cudaDeviceGetAttribute(&can_host_ptr, cudaDevAttrCanUseHostPointerForRegisteredMem, device) != cudaSuccess || !can_host_ptr) {
        (void)cudaGetLastError();
        return nullptr;
    }
#endif
    const uintptr_t page = 4096;
    uintptr_t b = (uintptr_t) ptr & ~(page - 1);
    uintptr_t e = ((uintptr_t) ptr + size + page - 1) & ~(page - 1);

    std::lock_guard<std::mutex> lock(ggml_cuda_host_mappings_mutex());
    auto & mappings = ggml_cuda_host_mappings();
    ggml_cuda_set_device(device);
    // clip the ends to the mappings already there; a mapping strictly inside the range is not supported
    for (int pass = 0; pass < 2; ++pass) {
        for (auto & [mb, m] : mappings) {
            const uintptr_t lo = (uintptr_t) mb;
            const uintptr_t hi = lo + m.size;
            if (b >= lo && e <= hi) {
                // already mapped whole: one more reference on that mapping
                m.refs++;
                *base       = mb;
                *registered = true;
                return ptr;
            }
            if (b >= lo && b < hi) {
                b = (hi + page - 1) & ~(page - 1);
            }
            if (e > lo && e <= hi) {
                e = lo & ~(page - 1);
            }
        }
    }
    for (const auto & [mb, m] : mappings) {
        const uintptr_t lo = (uintptr_t) mb;
        if (lo >= b && lo < e) {
            return nullptr;
        }
    }
    if (b >= e) {
        return nullptr; // only boundary pages, held by two other mappings: no reference to take
    }
    void * pb = (void *) b;
    const size_t len = e - b;
    const unsigned int flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
    bool read_only_ok = false;
#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    read_only_ok = cudaHostRegister(pb, len, flags | cudaHostRegisterReadOnly) == cudaSuccess;
    if (!read_only_ok) {
        (void)cudaGetLastError();
    }
#endif
    if (!read_only_ok && cudaHostRegister(pb, len, flags) != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: failed to map %.2f MiB of host memory\n", __func__, len/1024.0/1024.0);
        return nullptr;
    }
    ggml_cuda_host_mapping & m = mappings[pb];
    m = {};
    m.size = len;
    m.dev_ptr[device] = pb;
    m.refs = 1;
    *base       = pb;
    *registered = true;
    return ptr;
}

// bytes of host memory registered by ggml_backend_cuda_host_map and ggml_backend_cuda_host_map_range (all devices)
static size_t ggml_backend_cuda_host_mapped_bytes(void) {
    std::lock_guard<std::mutex> lock(ggml_cuda_host_mappings_mutex());
    size_t n = 0;
    for (const auto & [mb, m] : ggml_cuda_host_mappings()) {
        n += m.size;
    }
    return n;
}

// releases a reference taken by ggml_backend_cuda_host_map, the memory is unregistered with the last one
static void ggml_backend_cuda_host_unmap(void * buffer) {
    std::lock_guard<std::mutex> lock(ggml_cuda_host_mappings_mutex());
    auto & mappings = ggml_cuda_host_mappings();
    auto it = mappings.find(buffer);
    if (it == mappings.end() || --it->second.refs > 0) {
        return;
    }
    mappings.erase(it);
    if (cudaHostUnregister(buffer) != cudaSuccess) {
        (void)cudaGetLastError();
    }
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

// maps host memory into the address space of the device dev (see ggml_cuda_host_map)
// single-graph MoE mailbox (k11, see ggml-moe-mailbox.h): mapped host memory the device writes and reads, and its device control block
static void * ggml_backend_cuda_moe_mailbox_alloc(void * dev, size_t size, void ** mbox_dev, void ** ctl_dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)((ggml_backend_dev_t) dev)->context;
    ggml_cuda_set_device(ctx->device);
    void * host = nullptr;
    *mbox_dev = nullptr;
    *ctl_dev  = nullptr;
    if (cudaHostAlloc(&host, size, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    memset(host, 0, size);
    if (cudaHostGetDevicePointer(mbox_dev, host, 0) != cudaSuccess || *mbox_dev == nullptr ||
            cudaMalloc(ctl_dev, sizeof(ggml_moe_mailbox_ctl)) != cudaSuccess ||
            cudaMemset(*ctl_dev, 0, sizeof(ggml_moe_mailbox_ctl)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        (void)cudaGetLastError();
        if (*ctl_dev != nullptr) {
            (void)cudaFree(*ctl_dev);
        }
        (void)cudaFreeHost(host);
        *mbox_dev = nullptr;
        *ctl_dev  = nullptr;
        return nullptr;
    }
    return host;
}

static void ggml_backend_cuda_moe_mailbox_free(void * dev, void * host, void * ctl_dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)((ggml_backend_dev_t) dev)->context;
    ggml_cuda_set_device(ctx->device);
    if (ctl_dev != nullptr) {
        (void)cudaFree(ctl_dev);
    }
    if (host != nullptr) {
        (void)cudaFreeHost(host);
    }
    (void)cudaGetLastError();
}

static void * ggml_backend_cuda_host_map(ggml_backend_dev_t dev, void * buffer, size_t size, bool * registered) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_cuda_host_map(ctx->device, buffer, size, registered);
}

static void * ggml_backend_cuda_host_map_range(ggml_backend_dev_t dev, void * ptr, size_t size, void ** base, bool * registered) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_cuda_host_map_range(ctx->device, ptr, size, base, registered);
}

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;

    // debug (tests on a shared card): GGML_CUDA_FAKE_FREE_MIB=N reports N MiB less what this process allocated (ggml_cuda_own_bytes; the
    // CUDA context and CUDA graphs are not counted), at most the real free memory
    static const char * fake_env = getenv("GGML_CUDA_FAKE_FREE_MIB");
    if (fake_env != nullptr && atoll(fake_env) > 0) {
        int64_t own = 0;
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_own_mutex);
            own = ggml_cuda_own_bytes[ctx->device];
        }
        const int64_t cap = (int64_t) atoll(fake_env) << 20;
        *free = (size_t) std::max<int64_t>(0, std::min<int64_t>(cap - own, (int64_t) *free));
    }
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_MXFP8:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_MOE_CACHE_LOOKUP:
            return ggml_cuda_moe_cache_lookup_supported(op);
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

// an event that measures time (the scheduler measures how long splits wait for their prefetched copies), freed with ggml_backend_event_free
static ggml_backend_event_t ggml_backend_cuda_timer_new(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    if (cudaEventCreate(&event) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
}

// milliseconds from t0 to t1, false (no wait) if either has not completed yet
static bool ggml_backend_cuda_timer_elapsed(ggml_backend_event_t t0, ggml_backend_event_t t1, float * ms) {
    cudaEvent_t e0 = (cudaEvent_t) t0->context;
    cudaEvent_t e1 = (cudaEvent_t) t1->context;
    if (cudaEventQuery(e1) != cudaSuccess || cudaEventQuery(e0) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaEventElapsedTime(ms, e0, e1) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// host memory registered by ggml_backend_cuda_host_map (pinned)
static bool ggml_cuda_host_is_mapped(const void * ptr, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_host_mappings_mutex());
    for (const auto & [base, m] : ggml_cuda_host_mappings()) {
        if ((const char *) ptr >= (const char *) base && (const char *) ptr + size <= (const char *) base + m.size) {
            return true;
        }
    }
    return false;
}

// device copies of a copy-only instance (the scheduler copy queue): the kernel, except copy engines on compute capability 10.x where they measured faster (GGML_CUDA_COPY_QUEUE_KERNEL=0/1 overrides)
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
static bool ggml_cuda_copy_queue_kernel(int device) {
    static const std::vector<bool> kernel = [] {
        const char * env = getenv("GGML_CUDA_COPY_QUEUE_KERNEL");
        std::vector<bool> res(ggml_cuda_info().device_count, true);
        for (int id = 0; id < (int) res.size(); id++) {
            const int cc = ggml_cuda_info().devices[id].cc;
            res[id] = env != nullptr ? atoi(env) != 0 : !(cc >= 1000 && cc < 1100);
            GGML_LOG_INFO("ggml_cuda_copy_queue_kernel: device %d: compute capability %d.%d, device copies of a copy queue with the %s\n", id, cc/100, (cc%100)/10,
                res[id] ? "kernel" : "copy engines");
        }
        return res;
    }();
    return device >= 0 && device < (int) kernel.size() ? kernel[device] : true;
}
#endif

// many small copies in two calls: uploads as one cudaMemcpyBatchAsync (CUDA >= 12.8), device copies with one kernel
// GGML_CUDA_COPY_REGIONS_KERNEL=0 puts the device copies in the batch, GGML_CUDA_COPY_BATCH=0 disables both
static bool ggml_backend_cuda_copy_regions_async(ggml_backend_t backend, const ggml_backend_copy_region * regions, size_t n_regions) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || !defined(CUDART_VERSION) || CUDART_VERSION < 12080
    GGML_UNUSED(backend);
    GGML_UNUSED(regions);
    GGML_UNUSED(n_regions);
    return false;
#else
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_COPY_BATCH");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || !ggml_backend_is_cuda(backend)) {
        return false;
    }
    if (n_regions == 0) {
        return true;
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool d2d_kernel = [] {
        const char * env = getenv("GGML_CUDA_COPY_REGIONS_KERNEL");
        return env == nullptr || atoi(env) != 0;
    }();

    std::vector<void *>       dsts;
    std::vector<const void *> srcs;
    std::vector<size_t>       sizes;
    std::vector<void *>       d2d_dsts;
    std::vector<const void *> d2d_srcs;
    std::vector<size_t>       d2d_sizes;
    dsts.reserve(n_regions);
    srcs.reserve(n_regions);
    sizes.reserve(n_regions);
    for (size_t i = 0; i < n_regions; i++) {
        const ggml_backend_copy_region & r = regions[i];
        ggml_backend_buffer_t buf_dst = r.dst->view_src ? r.dst->view_src->buffer : r.dst->buffer;
        ggml_backend_buffer_t buf_src = r.src->view_src ? r.src->view_src->buffer : r.src->buffer;
        if (buf_dst == nullptr || !ggml_backend_buffer_is_cuda(buf_dst) ||
            ((ggml_backend_cuda_buffer_context *) buf_dst->context)->device != cuda_ctx->device) {
            return false;
        }
        // device memory of the same device, or pinned host memory (pageable memory cannot be read in stream order)
        const bool src_dev  = buf_src != nullptr && ggml_backend_buffer_is_cuda(buf_src) &&
            ((ggml_backend_cuda_buffer_context *) buf_src->context)->device == cuda_ctx->device;
        const bool src_host = buf_src != nullptr && !src_dev &&
            (ggml_backend_buft_is_cuda_host(ggml_backend_buffer_get_type(buf_src)) ||
             (ggml_backend_buffer_is_host(buf_src) && ggml_cuda_host_is_mapped((const char *) r.src->data + r.src_offset, r.size)));
        if (!src_dev && !src_host) {
            return false;
        }
        GGML_ASSERT(r.dst_offset + r.size <= ggml_nbytes(r.dst) && r.src_offset + r.size <= ggml_nbytes(r.src));
        if (r.size == 0) {
            continue;
        }
        void       * d = (char *) r.dst->data + r.dst_offset;
        const void * s = (const char *) r.src->data + r.src_offset;
        // a copy-only instance uses the kernel only where the copy engines are slow for device copies (see above), its device copies
        // that run on the compute instance (ggml_cuda_copy_peer_begin) use the kernel as the compute instance does
        if (src_dev && d2d_kernel && (cuda_ctx->computes || cuda_ctx->copy_peer != nullptr || ggml_cuda_copy_queue_kernel(cuda_ctx->device))) {
            d2d_dsts.push_back(d);
            d2d_srcs.push_back(s);
            d2d_sizes.push_back(r.size);
        } else {
            dsts.push_back(d);
            srcs.push_back(s);
            sizes.push_back(r.size);
        }
    }

    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx = 0;

    ggml_cuda_set_device(cuda_ctx->device);
    if (!d2d_dsts.empty()) {
        // the device copies and the uploads write different regions, the uploads do not wait for the device copies
        cudaStream_t stream = ggml_cuda_copy_peer_begin(cuda_ctx);
        ggml_cuda_copy_regions_d2d(stream, ggml_cuda_info().devices[cuda_ctx->device].nsm,
            d2d_dsts.data(), d2d_srcs.data(), d2d_sizes.data(), (int) d2d_dsts.size());
        if (stream != cuda_ctx->stream()) {
            // the uploads are queued before the copy queue waits for the device copies
            if (!dsts.empty()) {
#if CUDART_VERSION >= 13000
                CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, cuda_ctx->stream()));
#else
                size_t fail_idx = 0;
                CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), const_cast<void **>(srcs.data()), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, cuda_ctx->stream()));
#endif
                dsts.clear();
            }
            ggml_cuda_copy_peer_end(cuda_ctx, stream);
        }
    }
    if (!dsts.empty()) {
#if CUDART_VERSION >= 13000
        CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, cuda_ctx->stream()));
#else
        size_t fail_idx = 0;
        CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), const_cast<void **>(srcs.data()), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, cuda_ctx->stream()));
#endif
    }
    return true;
#endif
}

// whether an overlapped MoE cache fetch (ggml_moe_cache_fetch with overlap) is expected to be faster on the device dev:
// measured faster on compute capability 7.5 and 8.9 (T4, L4) at every budget; on 12.0 (RTX PRO 6000), where kernels launch with PDL,
// the second stream made the experts and the other kernels slower, a loss with few missing experts (up to -11%), so it is off there
// by default (and on the other PDL and AMD devices, not measured)
static bool ggml_backend_cuda_moe_fetch_overlap_default(ggml_backend_dev_t dev) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dev);
    return false;
#else
    const int cc = ggml_cuda_info().devices[((ggml_backend_cuda_device_context *) dev->context)->device].cc;
    return cc < GGML_CUDA_CC_HOPPER;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// the overlap pays when enough expert bytes are copied per layer and token to hide the cost of its extra nodes: none below compute
// capability 9.0 (faster on T4 and L4 at every measured budget); with PDL (9.0 and newer) the second stream slows the kernels beside it
// (RTX PRO 6000: -11% with 0.26 missing experts of gemma-4 26B per layer and token, +6% with 3.4), see llama_moe_cache
static size_t ggml_backend_cuda_moe_fetch_overlap_min_bytes(ggml_backend_dev_t dev) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dev);
    return SIZE_MAX;
#else
    const int cc = ggml_cuda_info().devices[((ggml_backend_cuda_device_context *) dev->context)->device].cc;
    return cc < GGML_CUDA_CC_HOPPER ? 0 : (size_t) 4 << 20;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_moe_fetch_overlap_min_bytes") == 0) {
        return (void *)ggml_backend_cuda_moe_fetch_overlap_min_bytes;
    }
    if (strcmp(name, "ggml_backend_moe_fetch_overlap_default") == 0) {
        return (void *)ggml_backend_cuda_moe_fetch_overlap_default;
    }
    if (strcmp(name, "ggml_backend_copy_regions_async") == 0) {
        return (void *)ggml_backend_cuda_copy_regions_async;
    }
    if (strcmp(name, "ggml_backend_set_copy_peer") == 0) {
        return (void *)ggml_backend_cuda_set_copy_peer;
    }
    if (strcmp(name, "ggml_backend_graph_evict_since") == 0) {
        return (void *)ggml_backend_cuda_graph_evict_since;
    }
    if (strcmp(name, "ggml_backend_timer_new") == 0) {
        return (void *)ggml_backend_cuda_timer_new;
    }
    if (strcmp(name, "ggml_backend_timer_elapsed") == 0) {
        return (void *)ggml_backend_cuda_timer_elapsed;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_host_map") == 0) {
        return (void *)ggml_backend_cuda_host_map;
    }
    if (strcmp(name, "ggml_backend_moe_mailbox_alloc") == 0) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        return (void *)ggml_backend_cuda_moe_mailbox_alloc;
#else
        return nullptr;
#endif
    }
    if (strcmp(name, "ggml_backend_moe_mailbox_free") == 0) {
        return (void *)ggml_backend_cuda_moe_mailbox_free;
    }
    if (strcmp(name, "ggml_backend_host_map_range") == 0) {
        return (void *)ggml_backend_cuda_host_map_range;
    }
    if (strcmp(name, "ggml_backend_set_tensors_batch_async") == 0) {
        return (void *)ggml_backend_cuda_set_tensors_batch_async;
    }
    if (strcmp(name, "ggml_backend_host_mapped_bytes") == 0) {
        return (void *)ggml_backend_cuda_host_mapped_bytes;
    }
    if (strcmp(name, "ggml_backend_host_unmap") == 0) {
        return (void *)ggml_backend_cuda_host_unmap;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
