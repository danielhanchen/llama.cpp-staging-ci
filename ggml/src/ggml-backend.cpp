// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-alloc-ext.h"
#include "ggml-impl.h"
#include "ggml-moe-prefetch.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    // like ggml_backend_tensor_set: nothing to copy (an empty graph input, e.g. a recurrent state copy list of 0 rows);
    // Metal cannot wrap a 0-byte host range (newBufferWithBytes returns nil -> GGML_ASSERT(buf_src))
    if (size == 0) {
        return;
    }
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    if (size == 0) {
        return;
    }
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

typedef bool (*ggml_backend_copy_regions_async_t)(ggml_backend_t backend, const struct ggml_backend_copy_region * regions, size_t n_regions);

bool ggml_backend_tensor_copy_regions_async(ggml_backend_t backend, const struct ggml_backend_copy_region * regions, size_t n_regions) {
    GGML_ASSERT(backend);
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg == NULL) {
        return false;
    }
    auto fn = (ggml_backend_copy_regions_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_regions_async");
    return fn != NULL && fn(backend, regions, n_regions);
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

// parts of the graph allocation in memory lent by the user (see ggml_backend_sched_lend_copy)
#define GGML_SCHED_MAX_GRAPH_PARTS 4

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

#ifndef GGML_SCHED_OV_FUSE_WINDOW
#define GGML_SCHED_OV_FUSE_WINDOW 8
#endif

#ifndef GGML_SCHED_OV_MIN_BYTES_PER_SPLIT
#define GGML_SCHED_OV_MIN_BYTES_PER_SPLIT (4u*1024u*1024u)
#endif

#ifndef GGML_SCHED_MAX_PREFETCH_DEPTH
#define GGML_SCHED_MAX_PREFETCH_DEPTH 8
#endif

#ifndef GGML_SCHED_MAX_LENT_BUFFERS
#define GGML_SCHED_MAX_LENT_BUFFERS 8
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
    // host weight prefetch: index among the prefetched splits of its backend (-1 if none), the copies use staging slot pf_ordinal % depth
    int  pf_ordinal;
    size_t pf_size; // bytes of its weight copies in the slot
    size_t pf_offset; // arena offset of its copies with the byte ring layout (depth > 2, see ggml_backend_sched_prefetch_layout)
    // copies issued ahead on the copy queue, else the regular path fills the slot (a callback changed since the graph was allocated)
    bool pf_async;
    // some of its copies are placed in memory lent by the user (see ggml_backend_sched_compute_splits)
    bool pf_lent;
    // hybrid MoE cache overlap [ggml_backend_sched_set_overlap]: number of leading nodes that do not depend on the previous (CPU) split, -1 if not computed yet
    int  ov_prefix;
    // CPU/GPU overlap: a host split copies its inputs from backend ov_src with async copies and one event (-1 if not),
    // issued at the start of split ov_issue_at (<= this split), so that the GPU splits in between run while the host split waits or computes
    int  ov_src;
    int  ov_issue_at;
    // index of the host split whose copies are issued at the start of this split (-1 if none)
    int  ov_issue_for;
    bool ov_issued; // the copies of this host split were issued during the current compute
};

// host weight prefetch state of a backend: the copies live in a staging arena of `depth` slots outside the graph allocator, so they never alias
// depth = 2: even ordinals start at the arena start, odd ones end at its end, so the arena needs the largest sum of two consecutive splits
// slot s: copy queue waits retire[s], fills s, records ready[s]; compute waits ready[s], reads s, records retire[s]
struct ggml_backend_sched_prefetch {
    bool checked;  // eligibility evaluated
    bool eligible; // the backend supports async copies and events
    bool failed;   // initialization or allocation failed, use the regular path

    ggml_backend_t        copy_backend; // second instance on the same device, used only for transfers
    ggml_backend_buffer_t arena;
    size_t                arena_size;
    ggml_backend_event_t  ready [GGML_SCHED_MAX_PREFETCH_DEPTH];
    ggml_backend_event_t  retire[GGML_SCHED_MAX_PREFETCH_DEPTH];
    ggml_backend_event_t  start; // recorded on the backend when a graph computation starts, the copy queue waits for it
    int                   depth; // staging slots of the current binding (sched->pf_depth when the graph was bound)
    bool                  ring;  // the copies of the current binding use the byte ring layout (pf_offset), else uniform slots

    // cross-graph arm [GGML_SCHED_PREFETCH_CROSS]: copies of the first splits of the next computation of the same bound graph, issued
    // after the last split of the current one (they wait for its compute), so they overlap the host work between the two
    int      armed;     // number of armed splits
    uint32_t armed_gen; // binding they were armed for

    // adaptive depth [GGML_SCHED_PREFETCH_ADAPT]: timing events around the waits of the prefetched splits of the last graph
    ggml_backend_event_t * timers;
    int  n_timers_cap;
    int  n_timed;   // prefetched splits timed in the last computed graph (2 events each, plus the first and last)
    bool timed;     // the last computed graph recorded timers that were not read yet
    bool shrink;    // the adaptive depth went back to a smaller depth: reallocate a larger arena at the next binding

    // prefetched splits of the current graph
    int * split_ids;
    int   n_bound;
    int   n_async;
    int   split_ids_capacity;
    size_t need;    // largest split of the current graph (uniform slot size)
    size_t staging; // arena size needed by the current graph

    // progress during graph compute
    int issued;
    int consumed;
    bool pending; // work was queued on the copy queue since it was last synchronized

    // copies placed in memory lent by the user (see ggml_backend_sched_lend_callback)
    struct ggml_backend_sched_lend_copy * lend_copies;
    int    lend_copies_capacity;
    ggml_backend_buffer_t lent_bufs[GGML_SCHED_MAX_LENT_BUFFERS]; // buffers holding copies of the current graph
    enum ggml_backend_buffer_usage lent_usage[GGML_SCHED_MAX_LENT_BUFFERS]; // their own usage while the splits are computed
    int    n_lent_bufs;
    bool   lent_all;  // the graph allocator uses lent memory: the lent buffers behave like compute buffers in every split
    bool   ensure_refused; // k15-fn: the staging arena of a merged graph did not allocate (the graph is refused, the prefetch kept)
    size_t lent;      // bytes of the copies and of the graph allocator in lent memory
    size_t lent_prev; // logged value
    size_t ext_prev;  // logged size of the graph allocator in lent memory
    ggml_backend_buffer_t ext_buf; // lent memory of the graph allocator, its tensors are not prefetched copies
    ggml_backend_buffer_t own_buf; // k15-fn: the graph allocator's own buffer as the first region of a lent graph allocation (see prefetch_bind)
    size_t own_size;
    size_t own_rest;               // the lent part it needs after the own buffer
    size_t ext_offset;
    size_t ext_size;
    bool   lending;   // the user lent memory for the last graph, it is told when the memory is no longer used
    bool   sync_lend_log; // logged that SYNC copies are placed in lent memory (the graph allocator could not grow by them)
    // the prefetched copies placed in lent memory (sorted): with the graph allocator in lent memory too, the buffer alone does not tell
    // a prefetched copy from a graph tensor (e.g. the copy of an input from another backend, which must not go through the copy queue)
    const struct ggml_tensor ** lent_copies;
    int    n_lent_copies;
    int    lent_copies_capacity;
};

// CPU/GPU split overlap state of a device backend [GGML_SCHED_OVERLAP]
struct ggml_backend_sched_overlap {
    bool checked;
    bool eligible;
    bool relaxed_ok;             // uploads are ordered after the work queued before them (one queue), they may skip the synchronization
    ggml_backend_event_t ready;  // recorded after the async copies of the inputs of a host split
    bool ready_pending;          // copies were queued and their host split has not waited for them yet
    ggml_backend_event_t upload; // recorded after uploads that read host compute memory without a synchronization
    bool upload_pending;         // `upload` has not been waited on yet
};

// k15-fn: a backend's batched host -> device upload (ggml_backend_set_tensors_batch_async proc of its registry), see ggml_backend_sched_upload_flush
typedef bool (*ggml_backend_set_tensors_batch_async_t)(ggml_backend_t backend, ggml_tensor * const * tensors, const void * const * data, const size_t * sizes, int n);

struct ggml_backend_sched_upload_batch {
    ggml_backend_t backend = NULL;
    std::vector<ggml_tensor *> tensors;
    std::vector<const void *>  data;
    std::vector<size_t>        sizes;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    // hash_set.size is the size in use: hash_size_full as allocated, or hash_size_small (if set) after every reset, for the usual
    // graphs; a larger graph asks for the full size after its reset (ggml_backend_sched_use_full_hash)
    size_t                hash_size_full;
    size_t                hash_size_small;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;
    bool callback_eval_nosync; // see ggml_backend_sched_set_eval_callback_sync

    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    void * callback_realloc_user_data;

    ggml_backend_sched_prefetch_callback callback_prefetch;
    void * callback_prefetch_user_data;

    // overlap of a CPU split with the next split [ggml_backend_sched_set_overlap]
    bool overlap;
    // a host weight whose copy exists on the node's backend does not start a new split (ggml_backend_sched_set_reuse_weight_copies)
    bool reuse_weight_copies;
    // the next allocations fail without allocating if a device compute buffer would grow too much (ggml_backend_sched_set_alloc_limit)
    bool   alloc_limit_on;
    size_t alloc_limit;
    bool   alloc_lend_graph; // k11: growth past the free memory may be lent (ggml_backend_sched_set_alloc_lend_graph)
    ggml_backend_sched_upload_batch * upload_batch; // k15-fn: the graph-input uploads of the split being prepared (GGML_SCHED_UPLOAD_BATCH)
    size_t alloc_over; // growth the last refused allocation needed, 0 if it was not refused
    int  pre_split; // split whose first pre_nodes nodes were launched before the CPU split preceding it, -1 if none
    int  pre_nodes;

    ggml_backend_sched_lend_callback callback_lend;
    int  lend_sync; // bindings with only SYNC copies and no arena are offered to the lender: 0 never, 1 (default) if the graph allocator cannot grow, 2 always (GGML_SCHED_LEND_SYNC)
    void * callback_lend_user_data;

    // reads of the prefetched host weights ahead [ggml_backend_sched_set_prefetch_plan_callback]
    ggml_backend_sched_prefetch_plan_callback callback_plan;
    void * callback_plan_user_data;
    ggml_backend_sched_prefetch_source_callback callback_source;
    void * callback_source_user_data;
    std::vector<const struct ggml_tensor *> * plan_srcs;

    // host weight prefetch [GGML_SCHED_PREFETCH]
    int  pf_mode;      // 0 = off, 1 = CUDA-family backends, 2 = any backend with async copies and events
    int  pf_depth;     // number of staging slots per backend (the target of the next binding)
    bool pf_active;    // the current graph has prefetched splits
    int  pf_depth0;    // depth at the start (GGML_SCHED_PREFETCH_DEPTH or the default)
    bool pf_window;    // depth > 2: byte ring layout sized by the live bytes of `depth` consecutive splits [GGML_SCHED_PREFETCH_WINDOW]
    int  pf_cross;     // cross-graph arm of the first copies [GGML_SCHED_PREFETCH_CROSS]: 0 off (default), 1 when the binding is expected to be
                       // computed again (the previous binding was computed more times than this one so far), 2 after every computation
    uint32_t pf_gen;   // incremented by every binding of a graph
    int  pf_runs;      // computations of the current binding
    int  pf_prev_runs; // computations of the previous binding (llama reuses the graph of a ubatch while the KV window does not grow)
    int64_t pf_n_armed;    // stats: splits armed
    int64_t pf_n_arm_used; // stats: armed splits the next computation used
    // adaptive depth [GGML_SCHED_PREFETCH_ADAPT]: grow the depth while splits wait for their copies, keep a step only if it cuts the waits
    // and makes the graphs faster (fewer waits alone do not pay when the deeper prefetch slows the work around them)
    bool   pf_adapt;      // adapting (not settled yet)
    int    pf_adapt_max;  // largest depth tried
    int    pf_adapt_from; // depth before the last step (0: no step taken yet)
    double pf_adapt_prev; // milliseconds waited per graph before the last step
    double pf_adapt_prev_busy; // milliseconds per timed graph span before the last step
    double pf_adapt_gain; // a step is kept only if the timed graph span gets at least this much shorter [GGML_SCHED_PREFETCH_ADAPT_GAIN]
    double pf_adapt_min_wait; // a step is tried only while the splits wait more than this share of the graph [GGML_SCHED_PREFETCH_ADAPT_MIN_WAIT]
    double pf_acc_stall;  // milliseconds waited for copies since the last decision
    double pf_acc_busy;   // milliseconds of the timed spans
    int    pf_acc_n;      // graphs measured since the last decision
    ggml_backend_event_t (*pf_timer_new)(ggml_backend_dev_t);
    bool (*pf_timer_elapsed)(ggml_backend_event_t, ggml_backend_event_t, float *);
    bool pf_check;     // a staging arena was allocated since the headroom was last checked
    size_t pf_headroom; // free device memory a reserve / graph allocation must leave with a staging arena, else the prefetch is dropped
    struct ggml_backend_sched_prefetch pf[GGML_SCHED_MAX_BACKENDS];

    // CPU/GPU split overlap [GGML_SCHED_OVERLAP]
    int  ov_mode;   // 0 = off, 1 = CUDA-family backends, 2 = any backend with async copies and events
    int  ov_parts;  // GGML_SCHED_OVERLAP_PARTS bits: 1 = reorder (opt-in), 2 = copies issued ahead, 4 = uploads without a sync (default 6)
    bool ov_active; // the current graph uses the overlap
    int  ov_logged[4];
    int64_t n_input_relaxed; // k13-pp: graph-input uploads queued without a device sync (GGML_SCHED_INPUT_RELAXED)
    bool    input_relax_ok;  // k13-pp: the current graph has no host split after a device split (set in compute_splits)
    struct ggml_backend_sched_overlap ov[GGML_SCHED_MAX_BACKENDS];

    // k15 [GGML_SCHED_HOST_SYNC_MERGE]: graphs without the overlap (e.g. hybrid MoE cache graphs): the device inputs of a host split are
    // read with async copies and one device sync, and the upload of a host split's output to the device is not waited for at once
    // (the next host split's sync, or the end of the graph, waits for it)
    int     hsm_mode;          // 0 = off, 1 = CUDA family (default)
    bool    hsm_active;        // the current graph uses it
    int     hsm_pending;       // backend id with an upload not yet waited for, -1 = none
    ggml_backend_event_t hsm_event[GGML_SCHED_MAX_BACKENDS]; // recorded after that upload (NULL: the end of the graph synchronizes)
    bool    hsm_event_tried[GGML_SCHED_MAX_BACKENDS];
    int64_t hsm_n_reads, hsm_n_defer; // inputs read without their own syncs, uploads not waited for at once

    char * context_buffer;
    size_t context_buffer_size;
    size_t context_graph_size; // the graph_size of ggml_backend_sched_new: the most nodes the context is sized for

    bool op_offload;

    int debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int graph_size;
    int prev_graph_size;

    int debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

//
// CPU/GPU split overlap [GGML_SCHED_OVERLAP]
// a host split (e.g. MoE experts in host memory) waits for one event after async copies of its inputs instead of
// synchronizing the device per input, and device work that is independent of it is queued before it, so that both run concurrently
//

static const struct ggml_tensor * ggml_backend_sched_ov_root(const struct ggml_tensor * t) {
    while (t->view_src != NULL) {
        t = t->view_src;
    }
    return t;
}

static bool ggml_backend_sched_ov_is_host(ggml_backend_sched_t sched, int backend_id) {
    return ggml_backend_buft_is_host(sched->bufts[backend_id]);
}

// device backend whose work can be ordered with events against the host splits
static bool ggml_backend_sched_ov_backend_ok(ggml_backend_sched_t sched, int backend_id) {
    struct ggml_backend_sched_overlap * ov = &sched->ov[backend_id];
    if (!ov->checked) {
        ov->checked  = true;
        ov->eligible = false;

        ggml_backend_t     backend = sched->backends[backend_id];
        ggml_backend_dev_t dev     = backend->device;
        if (sched->ov_mode == 0 || sched->n_copies > 1 || dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ||
            ggml_backend_sched_ov_is_host(sched, backend_id)) {
            return false;
        }
        if (backend->iface.event_record == NULL || backend->iface.get_tensor_async == NULL || backend->iface.set_tensor_async == NULL) {
            return false;
        }
        struct ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.events) {
            return false;
        }
        // a device that reads host compute memory directly could race with a host split
        for (int b = 0; b < sched->n_backends; b++) {
            if (ggml_backend_sched_ov_is_host(sched, b) && ggml_backend_supports_buft(backend, sched->bufts[b])) {
                return false;
            }
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";
        const bool cuda_family = strcmp(reg_name, "CUDA") == 0 || strcmp(reg_name, "ROCm") == 0 || strcmp(reg_name, "MUSA") == 0;
        if (sched->ov_mode == 1 && !cuda_family) {
            // other backends have not been validated yet, they can be enabled with GGML_SCHED_OVERLAP=2
            return false;
        }
        // the CUDA family queues uploads on the stream of the compute; other backends may upload on a separate queue
        // (e.g. the Vulkan transfer queue) that does not wait for earlier compute still reading the destination
        ov->relaxed_ok = cuda_family;
        if (!cuda_family && (sched->ov_parts & 4)) {
            GGML_LOG_INFO("%s: sched overlap: %s uploads keep their synchronization (upload ordering with compute not known)\n",
                __func__, ggml_backend_name(backend));
        }
        ov->ready  = ggml_backend_event_new(dev);
        ov->upload = ggml_backend_event_new(dev);
        if (ov->ready == NULL || ov->upload == NULL) {
            ggml_backend_event_free(ov->ready);
            ggml_backend_event_free(ov->upload);
            ov->ready  = NULL;
            ov->upload = NULL;
            return false;
        }
        ov->eligible = true;
    }
    return ov->eligible;
}

// ops that may be moved: no hidden reads or writes outside of their sources and their own result
static bool ggml_backend_sched_ov_movable_op(const struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_MAP_CUSTOM3:
        case GGML_OP_CUSTOM:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_MOE_CACHE_LOOKUP:
            return false;
        default:
            break;
    }
    // writes into another tensor (in-place ops, cpy, set_rows)
    return t->view_src == NULL || ggml_op_is_empty(t->op);
}

// the graph has no ops with hidden side effects and no device node reads experts or rows of host weights: their copies
// may go through the copy callback, which can also write device memory (e.g. the slots of a MoE cache) at the start of a split
static bool ggml_backend_sched_ov_graph_ok(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MOE_CACHE_LOOKUP || node->op == GGML_OP_CUSTOM ||
            node->op == GGML_OP_MAP_CUSTOM1 || node->op == GGML_OP_MAP_CUSTOM2 || node->op == GGML_OP_MAP_CUSTOM3) {
            return false;
        }
        if (node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_GET_ROWS) {
            continue;
        }
        const int b = tensor_backend_id(node);
        if (b < 0 || ggml_backend_sched_ov_is_host(sched, b)) {
            continue;
        }
        const struct ggml_tensor * root = ggml_backend_sched_ov_root(node->src[0]);
        if (root->buffer != NULL && ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(root->buffer)) {
            return false;
        }
    }
    return true;
}

// reorders the nodes around each host run C between device runs P and N of the same device backend G:
//   A (P nodes needed by C, and the P nodes that cannot move with their sources), D (the rest of P),
//   H (N nodes that need neither C nor the rest of N), C, N \ H
// a new split starts at D u H, so the copies of the inputs of C are issued after A and C runs while the device computes D u H
// only independent pure nodes move, so the results are the same; returns the number of reordered windows
static int ggml_backend_sched_ov_reorder(ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        std::unordered_set<const ggml_tensor *> & cuts, int & n_moved) {
    const int n = graph->n_nodes;
    if (n == 0) {
        return 0;
    }

    // roots written by nodes: readers of them keep their place
    std::unordered_set<const ggml_tensor *> written;
    for (int i = 0; i < n; i++) {
        const struct ggml_tensor * node = graph->nodes[i];
        if (node->view_src != NULL && !ggml_op_is_empty(node->op)) {
            written.insert(ggml_backend_sched_ov_root(node));
        }
        // the optimizer steps also update sources other than their result (e.g. the AdamW moments)
        if (node->op == GGML_OP_OPT_STEP_ADAMW || node->op == GGML_OP_OPT_STEP_SGD) {
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (node->src[j] != NULL) {
                    written.insert(ggml_backend_sched_ov_root(node->src[j]));
                }
            }
        }
    }

    // a source outside the window that a moved node may read: an earlier result, an input or an unmodified weight
    auto outside_ok = [&](const struct ggml_tensor * s) {
        const struct ggml_tensor * root = ggml_backend_sched_ov_root(s);
        if (written.count(root) > 0) {
            return false;
        }
        if (root->op != GGML_OP_NONE) {
            return true;
        }
        if (root->flags & GGML_TENSOR_FLAG_INPUT) {
            return true;
        }
        return root->buffer != NULL && ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    };

    std::vector<int> be(n);
    for (int i = 0; i < n; i++) {
        be[i] = tensor_backend_id(graph->nodes[i]);
    }

    // runs of non-view nodes of the same backend, as split in pass 5 (views belong to the run of the node before them)
    struct run { int start; int end; int backend; };
    std::vector<run> runs;
    auto find_runs = [&]() {
        runs.clear();
        for (int i = 0; i < n; i++) {
            if (ggml_is_view_op(graph->nodes[i]->op)) {
                continue;
            }
            if (runs.empty()) {
                runs.push_back({0, n, be[i]});
            } else if (be[i] != runs.back().backend) {
                runs.back().end = i;
                runs.push_back({i, n, be[i]});
            }
        }
    };

    // direct consumers of each node (through sources and view sources)
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> consumers;
    for (int i = 0; i < n; i++) {
        const struct ggml_tensor * node = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != NULL) {
                consumers[node->src[j]].push_back(node);
            }
        }
        if (node->view_src != NULL && node->view_src != node->src[0]) {
            consumers[node->view_src].push_back(node);
        }
    }

    std::unordered_map<const ggml_tensor *, int> wpos; // window node -> index in the window
    std::vector<char> cls;                             // per window index: 0 = A, 1 = D, 2 = C, 3 = H, 4 = N \ H
    std::vector<int> stack;
    std::vector<ggml_tensor *> order;

    // srcs of t and its view source
    auto for_srcs = [](const struct ggml_tensor * t, const std::function<void(const struct ggml_tensor *)> & f) {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (t->src[j] != NULL) {
                f(t->src[j]);
            }
        }
        if (t->view_src != NULL) {
            f(t->view_src);
        }
    };

    int n_windows = 0;
    int pos = 0;
    while (true) {
        find_runs();
        int r = -1;
        for (int k = 1; k < (int) runs.size(); k++) {
            if (runs[k].start >= pos && ggml_backend_sched_ov_is_host(sched, runs[k].backend)) {
                r = k;
                break;
            }
        }
        if (r < 0) {
            break;
        }
        const int G  = runs[r - 1].backend;
        const int ps = runs[r - 1].start;
        const int cs = runs[r].start;
        const int ce = runs[r].end;
        const int ne = (r + 1 < (int) runs.size() && runs[r + 1].backend == G) ? runs[r + 1].end : ce;
        pos = ce;
        if (ggml_backend_sched_ov_is_host(sched, G) || !ggml_backend_sched_ov_backend_ok(sched, G)) {
            continue;
        }

        // the order of a window that holds cuts of an earlier window is kept
        bool has_cut = false;
        for (int i = ps; i < ne && !has_cut; i++) {
            has_cut = cuts.count(graph->nodes[i]) > 0;
        }
        if (has_cut) {
            continue;
        }

        wpos.clear();
        for (int i = ps; i < ne; i++) {
            wpos[graph->nodes[i]] = i - ps;
        }
        cls.assign(ne - ps, 1);
        for (int i = cs; i < ce; i++) {
            cls[i - ps] = 2;
        }
        for (int i = ce; i < ne; i++) {
            cls[i - ps] = 4;
        }

        // the last node stays last: some users read the result of a graph as its last node
        auto movable = [&](int i) {
            const struct ggml_tensor * t = graph->nodes[i];
            if (be[i] != G || i == n - 1 || !ggml_backend_sched_ov_movable_op(t)) {
                return false;
            }
            bool ok = true;
            for_srcs(t, [&](const struct ggml_tensor * s) {
                // a tensor written in place keeps the order of its readers and writers, also inside the window
                if (written.count(ggml_backend_sched_ov_root(s)) > 0 || (wpos.count(s) == 0 && !outside_ok(s))) {
                    ok = false;
                }
            });
            return ok;
        };

        // A: ancestors in P of C and of the P nodes that stay
        auto close_a = [&]() {
            stack.clear();
            for (int i = ps; i < ce; i++) {
                if (cls[i - ps] == 0 || cls[i - ps] == 2) {
                    stack.push_back(i);
                }
            }
            while (!stack.empty()) {
                const int i = stack.back();
                stack.pop_back();
                for_srcs(graph->nodes[i], [&](const struct ggml_tensor * s) {
                    auto it = wpos.find(s);
                    if (it != wpos.end() && it->second < cs - ps && cls[it->second] == 1) {
                        cls[it->second] = 0;
                        stack.push_back(ps + it->second);
                    }
                });
            }
        };
        for (int i = ps; i < cs; i++) {
            if (!movable(i)) {
                cls[i - ps] = 0;
            }
        }
        close_a();

        // H: N nodes whose window sources are in P or H
        auto close_h = [&](bool init) {
            for (int i = ce; i < ne; i++) {
                if (init ? !movable(i) : cls[i - ps] != 3) {
                    continue;
                }
                bool ok = true;
                for_srcs(graph->nodes[i], [&](const struct ggml_tensor * s) {
                    auto it = wpos.find(s);
                    if (it != wpos.end() && it->second >= cs - ps && cls[it->second] != 3) {
                        ok = false;
                    }
                });
                cls[i - ps] = ok ? 3 : 4;
            }
        };
        close_h(true);

        // keep possible fusion groups of the device backend whole (a backend fuses consecutive nodes that consume each
        // other, e.g. rms_norm + mul, matmul + glu, or the top-k / weights group of a MoE router), so that the moved
        // nodes leave no group: a node and the next node (views skipped) that consumes it, a node and the next two that
        // a third one consumes, and a node with several consumers within GGML_SCHED_OV_FUSE_WINDOW nodes stay on one
        // side; the moving nodes of a mixed group are pinned
        auto is_mover  = [&](int k) { return cls[k] == 1 || cls[k] == 3; };
        auto is_stayer = [&](int k) { return cls[k] == 0 || cls[k] == 4; };
        auto consumes  = [&](const struct ggml_tensor * x, const struct ggml_tensor * y) {
            bool r = false;
            for_srcs(x, [&](const struct ggml_tensor * s) {
                for (const struct ggml_tensor * v = s; v != NULL && !r; v = v->view_src) {
                    r = v == y;
                }
            });
            return r;
        };
        auto pin = [&](int k) {
            if (cls[k] == 1) {
                cls[k] = 0;
            } else if (cls[k] == 3) {
                cls[k] = 4;
            }
        };
        std::vector<int> seq; // non-view window indices of one region, in order
        for (bool changed = true; changed; ) {
            changed = false;
            for (int region = 0; region < 2; region++) {
                const int r0 = region == 0 ? 0 : ce - ps;
                const int r1 = region == 0 ? cs - ps : ne - ps;
                seq.clear();
                for (int k = r0; k < r1; k++) {
                    if (!ggml_is_view_op(graph->nodes[ps + k]->op)) {
                        seq.push_back(k);
                    }
                }
                for (size_t q = 1; q < seq.size(); q++) {
                    const int a = seq[q - 1];
                    const int b = seq[q];
                    bool link = consumes(graph->nodes[ps + b], graph->nodes[ps + a]);
                    int lo = a;
                    if (!link && q >= 2) {
                        const int a2 = seq[q - 2];
                        if (consumes(graph->nodes[ps + b], graph->nodes[ps + a2]) && consumes(graph->nodes[ps + b], graph->nodes[ps + a])) {
                            link = true;
                            lo   = a2;
                        }
                    }
                    if (!link) {
                        continue;
                    }
                    bool m = false;
                    bool st = false;
                    for (size_t u = 0; u < seq.size(); u++) {
                        if (seq[u] >= lo && seq[u] <= b) {
                            m  |= is_mover(seq[u]);
                            st |= is_stayer(seq[u]);
                        }
                    }
                    if (m && st) {
                        for (size_t u = 0; u < seq.size(); u++) {
                            if (seq[u] >= lo && seq[u] <= b) {
                                pin(seq[u]);
                            }
                        }
                        changed = true;
                    }
                }
                for (int t = r0; t < r1; t++) {
                    auto it = consumers.find(graph->nodes[ps + t]);
                    if (it == consumers.end() || it->second.size() < 2) {
                        continue;
                    }
                    bool local = true;
                    bool m = is_mover(t);
                    bool st = is_stayer(t);
                    for (const struct ggml_tensor * u : it->second) {
                        auto ip = wpos.find(u);
                        if (ip == wpos.end() || ip->second <= t || ip->second > t + GGML_SCHED_OV_FUSE_WINDOW) {
                            local = false;
                            break;
                        }
                        if (ip->second < r1) {
                            m  |= is_mover(ip->second);
                            st |= is_stayer(ip->second);
                        }
                    }
                    if (local && m && st) {
                        pin(t);
                        for (const struct ggml_tensor * u : it->second) {
                            const int k = wpos[u];
                            if (k < r1) {
                                pin(k);
                            }
                        }
                        changed = true;
                    }
                }
            }
            if (changed) {
                close_a();
                close_h(false);
            }
        }

        int n_a  = 0;
        int n_dh = 0;
        size_t mm_bytes = 0; // weights read by the moved matrix multiplications
        for (int i = ps; i < ne; i++) {
            const char c = cls[i - ps];
            n_a += c == 0;
            if (c == 1 || c == 3) {
                n_dh++;
                const struct ggml_tensor * t = graph->nodes[i];
                if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
                    mm_bytes += ggml_nbytes(t->src[0]);
                }
            }
        }
        if (n_dh == 0) {
            continue;
        }

        order.clear();
        for (char c : { 0, 1, 3, 2, 4 }) {
            for (int i = ps; i < ne; i++) {
                if (cls[i - ps] == c) {
                    order.push_back(graph->nodes[i]);
                }
            }
        }
        GGML_ASSERT((int) order.size() == ne - ps);
        // a split starts wherever two nodes of the same backend become neighbors, so that no new fusion appears
        std::vector<const ggml_tensor *> new_cuts;
        for (int k = 1; k < (int) order.size(); k++) {
            if (wpos[order[k]] == wpos[order[k - 1]] + 1) {
                continue;
            }
            for (int j = k; j < (int) order.size(); j++) {
                if (!ggml_is_view_op(order[j]->op)) {
                    if (be[ps + wpos[order[j]]] == G && (new_cuts.empty() || new_cuts.back() != order[j])) {
                        new_cuts.push_back(order[j]);
                    }
                    break;
                }
            }
        }
        // each extra split costs a launch: worth it only with enough device work to overlap
        if (mm_bytes < new_cuts.size() * (size_t) GGML_SCHED_OV_MIN_BYTES_PER_SPLIT) {
            continue;
        }
        cuts.insert(new_cuts.begin(), new_cuts.end());
        for (int i = ps; i < ne; i++) {
            graph->nodes[i] = order[i - ps];
            be[i] = tensor_backend_id(graph->nodes[i]);
        }
        n_moved += n_dh;
        n_windows++;
        // continue after C in the new order
        pos = ps + n_a + n_dh + (ce - cs);
    }
    return n_windows;
}

// a host split input copied from device backend ov_src with an async copy (decided again at compute time, when the buffers are known)
static bool ggml_backend_sched_ov_early_input(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split, struct ggml_tensor * input) {
    const int G = split->ov_src;
    if (G < 0 || tensor_backend_id(input) != G) {
        return false;
    }
    for (const struct ggml_tensor * t = input; t != NULL; t = t->view_src) {
        if (t->flags & GGML_TENSOR_FLAG_INPUT) {
            return false;
        }
    }
    if (input->data == NULL || input->buffer == NULL ||
        ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
        ggml_backend_buffer_get_type(input->buffer) != ggml_backend_get_default_buffer_type(sched->backends[G])) {
        return false;
    }
    const struct ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
    return input_cpy != NULL && input_cpy->data != NULL && input_cpy->buffer != NULL && ggml_backend_buffer_is_host(input_cpy->buffer);
}

// for each host split: the device backend of its inputs and the earliest split at whose start their copies can be issued
//   the splits in between run on that backend, write none of the copied tensors, and (after the first one) read no host
//   compute memory, which the copies may overwrite (their destinations are allocated at the host split)
static void ggml_backend_sched_ov_plan(ggml_backend_sched_t sched, int & n_early, int & n_ahead) {
    n_early = 0;
    n_ahead = 0;
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->ov_src       = -1;
        split->ov_issue_at  = -1;
        split->ov_issue_for = -1;
        split->ov_issued    = false;
    }
    if (!sched->ov_active) {
        return;
    }
    std::unordered_set<const ggml_tensor *> roots;
    for (int s = 0; s < sched->n_splits; s++) {
        struct ggml_backend_sched_split * split = &sched->splits[s];
        if (!ggml_backend_sched_ov_is_host(sched, split->backend_id)) {
            continue;
        }
        int G = -1;
        roots.clear();
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            const int b = tensor_backend_id(input);
            if (b < 0 || ggml_backend_sched_ov_is_host(sched, b) || !ggml_backend_sched_ov_backend_ok(sched, b) || (G >= 0 && b != G)) {
                continue;
            }
            G = b;
            roots.insert(ggml_backend_sched_ov_root(input));
        }
        if (G < 0) {
            continue;
        }
        int e = s;
        for (int j = s - 1; j >= 0; j--) {
            const struct ggml_backend_sched_split * sj = &sched->splits[j];
            if (sj->backend_id != G) {
                break;
            }
            bool writes = false;
            for (int k = 0; k < sj->graph.n_nodes && !writes; k++) {
                writes = roots.count(ggml_backend_sched_ov_root(sj->graph.nodes[k])) > 0;
            }
            if (writes) {
                break;
            }
            e = j;
            // host memory that the graph allocator owns may be where the copies are written: results of earlier splits and
            // graph inputs (leafs without a buffer, e.g. INPUT tensors, which are freed after their last upload)
            bool host_input = false;
            for (int k = 0; k < sj->n_inputs && !host_input; k++) {
                const int b = tensor_backend_id(sj->inputs[k]);
                const struct ggml_tensor * root = ggml_backend_sched_ov_root(sj->inputs[k]);
                host_input = b < 0 || (ggml_backend_sched_ov_is_host(sched, b) && (root->op != GGML_OP_NONE || root->buffer == NULL ||
                    ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE));
            }
            if (host_input) {
                break;
            }
        }
        if (!(sched->ov_parts & 2)) {
            e = s;
        }
        split->ov_src      = G;
        split->ov_issue_at = e;
        if (e < s) {
            sched->splits[e].ov_issue_for = s;
            n_ahead++;
        }
        n_early++;
    }
}

// wait for the uploads that read host compute memory, except those queued on backend `except` (ordered with its later work)
static void ggml_backend_sched_ov_wait_uploads(ggml_backend_sched_t sched, int except) {
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_overlap * ov = &sched->ov[b];
        if (ov->upload_pending && b != except) {
            ggml_backend_event_synchronize(ov->upload);
            ov->upload_pending = false;
        }
    }
}

// queue the async copies of the device inputs of host split s and record its event
static void ggml_backend_sched_ov_issue(ggml_backend_sched_t sched, int s) {
    struct ggml_backend_sched_split * split = &sched->splits[s];
    const int G = split->ov_src;
    ggml_backend_t backend = sched->backends[G];
    // a pending upload of another backend may still read the host memory that the copies write
    ggml_backend_sched_ov_wait_uploads(sched, G);
    for (int j = 0; j < split->n_inputs; j++) {
        struct ggml_tensor * input = split->inputs[j];
        if (ggml_backend_sched_ov_early_input(sched, split, input)) {
            struct ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
            ggml_backend_tensor_get_async(backend, input, input_cpy->data, 0, ggml_nbytes(input));
        }
    }
    ggml_backend_event_record(sched->ov[G].ready, backend);
    sched->ov[G].ready_pending = true;
    split->ov_issued = true;
}

// wait for all async copies into host memory (a computation stopped before their host split, or memory is about to be reallocated)
static void ggml_backend_sched_ov_drain(ggml_backend_sched_t sched) {
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->ov[b].ready_pending) {
            ggml_backend_event_synchronize(sched->ov[b].ready);
            sched->ov[b].ready_pending = false;
        }
    }
    ggml_backend_sched_ov_wait_uploads(sched, -1);
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;
    sched->pf_active = false;

    ggml_free(sched->ctx);
    sched->ctx = NULL;

    // the context holds the tensors made while splitting (input copies, views): sized for this graph by the same bound as before
    // (at most one split per node), grown only, at most the scheduler's graph_size. A scheduler sized for merged layer-major graphs
    // (17x the nodes) reserved 7.9 GB here per context at load (Qwen3.6-35B-A3B + MTP: VmData +11.9 GB, Windows commit +16 GB) that
    // only a merged graph can use. GGML_SCHED_CTX_FULL=1: the full size at the first split, as before
    {
        static const bool full = getenv("GGML_SCHED_CTX_FULL") != nullptr && atoi(getenv("GGML_SCHED_CTX_FULL")) != 0;
        const size_t n    = full ? sched->context_graph_size : std::min(sched->context_graph_size, (size_t) std::max(graph->n_nodes, graph->n_leafs) + 1);
        const size_t need = n*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(n, false);
        if (sched->context_buffer == NULL || sched->context_buffer_size < need) {
            free(sched->context_buffer);
            sched->context_buffer_size = need;
            sched->context_buffer = (char *) malloc(need);
            GGML_ASSERT(sched->context_buffer != NULL);
        }
    }

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 4b: queue the device work that does not depend on a host split before it [GGML_SCHED_OVERLAP]
    std::unordered_set<const ggml_tensor *> ov_cuts;
    int ov_windows = 0;
    int ov_moved   = 0;
    sched->ov_active = sched->ov_mode != 0 && sched->n_copies == 1 && ggml_backend_sched_ov_graph_ok(sched, graph);
    if (sched->ov_active && (sched->ov_parts & 1)) {
        ov_windows = ggml_backend_sched_ov_reorder(sched, graph, ov_cuts, ov_moved);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        split->pf_ordinal = -1;
        split->pf_async = false;
        split->pf_lent  = false;
        split->ov_prefix = -1;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id) &&
                                !(sched->reuse_weight_copies && tensor_id_copy(hash_id(src), cur_backend_id, 0) != NULL)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            // the device work moved ahead of a host split starts its own split
            if (node_backend_id == cur_backend_id && !ov_cuts.empty() && ov_cuts.count(node) > 0) {
                need_new_split = true;
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                split->pf_ordinal = -1;
                split->pf_async = false;
                split->pf_lent  = false;
                split->ov_prefix = -1;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    {
        int ov_early = 0;
        int ov_ahead = 0;
        ggml_backend_sched_ov_plan(sched, ov_early, ov_ahead);
        const int stats[4] = { ov_early, ov_ahead, ov_windows, ov_moved };
        if ((ov_early > 0 || ov_windows > 0) && memcmp(stats, sched->ov_logged, sizeof(stats)) != 0) {
            memcpy(sched->ov_logged, stats, sizeof(stats));
            GGML_LOG_INFO("%s: sched overlap: %d host splits with async input copies (%d issued ahead), %d windows reordered (%d nodes moved)\n",
                __func__, ov_early, ov_ahead, ov_windows, ov_moved);
        }
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_prefetch_headroom_ok(ggml_backend_sched_t sched);
static bool ggml_backend_sched_prefetch_disable(ggml_backend_sched_t sched);

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    // like ggml_backend_sched_reserve: a staging arena allocated for this graph must leave the headroom, also when the graph fits the current buffers
    if (sched->pf_check && !ggml_backend_sched_prefetch_headroom_ok(sched)) {
        ggml_backend_sched_prefetch_disable(sched);
    }
    sched->pf_check = false;

    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        const bool graph_size_changed = sched->prev_graph_size != sched->graph_size;

        if (sched->callback_realloc != NULL) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && !graph_size_changed;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        // the larger graph buffers next to the staging arena must leave the headroom, else drop the arena and reserve again
        if (!ggml_backend_sched_prefetch_headroom_ok(sched) && ggml_backend_sched_prefetch_disable(sched) &&
            !ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

//
// host weight prefetch: the host weight copies of a split use one of `depth` staging slots and are uploaded on a copy queue up to `depth` splits ahead, ordered with events
//

static bool ggml_backend_sched_prefetch_backend_ok(ggml_backend_sched_t sched, int backend_id) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    if (!pf->checked) {
        pf->checked  = true;
        pf->eligible = false;

        ggml_backend_t     backend = sched->backends[backend_id];
        ggml_backend_dev_t dev     = backend->device;
        if (dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU || ggml_backend_buft_is_host(sched->bufts[backend_id])) {
            return false;
        }
        if (backend->iface.event_record == NULL || backend->iface.event_wait == NULL) {
            return false;
        }
        struct ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.events) {
            return false;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";
        if (sched->pf_mode != 2) {
            // other backends have not been validated yet, they can be enabled with GGML_SCHED_PREFETCH=2
            if (strcmp(reg_name, "CUDA") != 0 && strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0) {
                return false;
            }
        }
        // SYCL: weights in mmap'ed memory must be uploaded through the pinned staging of set_tensor, which the async copies skip
        if (strcmp(reg_name, "SYCL") == 0) {
            return false;
        }
        pf->eligible = true;
    }
    return pf->eligible && !pf->failed;
}

static const struct ggml_tensor * ggml_backend_sched_view_root(const struct ggml_tensor * t) {
    while (t->view_src != NULL) {
        t = t->view_src;
    }
    return t;
}

// written: the host weights that nodes of the graph write to (directly or through a view)
// first_write: the first split that writes each other host tensor (e.g. a KV cache), split_id: the split that reads input
static enum ggml_backend_sched_prefetch_mode ggml_backend_sched_prefetch_candidate(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input,
        const std::unordered_set<const ggml_tensor *> & written, const std::unordered_map<const ggml_tensor *, int> & first_write, int split_id) {
    // inputs from the user (also through views) may be overwritten as soon as the computation is queued
    for (const struct ggml_tensor * t = input; t != NULL; t = t->view_src) {
        if (t->flags & GGML_TENSOR_FLAG_INPUT) {
            return GGML_BACKEND_SCHED_PREFETCH_NONE;
        }
    }
    if (!ggml_backend_sched_is_host_weight(input)) {
        // other host memory only if the user asks for it and no split up to the reader writes it, so the copy can be made early
        const struct ggml_tensor * root = ggml_backend_sched_view_root(input);
        if (sched->callback_prefetch == NULL || root->buffer == NULL || !ggml_backend_buffer_is_host(root->buffer) ||
            ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return GGML_BACKEND_SCHED_PREFETCH_NONE;
        }
        auto it = first_write.find(root);
        if (it != first_write.end() && it->second <= split_id) {
            return GGML_BACKEND_SCHED_PREFETCH_NONE;
        }
        const struct ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
        const enum ggml_backend_sched_prefetch_mode mode = sched->callback_prefetch(input, input_cpy, &split->graph, sched->callback_prefetch_user_data);
        return mode == GGML_BACKEND_SCHED_PREFETCH_ASYNC ? mode : GGML_BACKEND_SCHED_PREFETCH_NONE;
    }
    // the copy of a weight written by the graph must follow the split that writes it
    if (written.count(ggml_backend_sched_view_root(input)) > 0) {
        return GGML_BACKEND_SCHED_PREFETCH_NONE;
    }
    if (sched->callback_prefetch != NULL) {
        const struct ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
        return sched->callback_prefetch(input, input_cpy, &split->graph, sched->callback_prefetch_user_data);
    }
    // the copy callback may be required for some weights, so prefetch only if the user tells which ones
    return sched->callback_copy == NULL ? GGML_BACKEND_SCHED_PREFETCH_ASYNC : GGML_BACKEND_SCHED_PREFETCH_NONE;
}

// wait for the work queued on the copy queues (graphs without prefetched splits, e.g. generation, queue none)
static void ggml_backend_sched_prefetch_drain(ggml_backend_sched_t sched) {
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->pf[b].copy_backend != NULL && sched->pf[b].pending) {
            ggml_backend_synchronize(sched->pf[b].copy_backend);
        }
        sched->pf[b].pending = false;
    }
}

static void ggml_backend_sched_prefetch_sync(ggml_backend_sched_t sched) {
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_synchronize(sched->backends[b]);
    }
    ggml_backend_sched_prefetch_drain(sched);
}

static void ggml_backend_sched_prefetch_release(ggml_backend_sched_t sched, int backend_id) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    for (int s = 0; s < GGML_SCHED_MAX_PREFETCH_DEPTH; s++) {
        if (pf->ready[s] != NULL) {
            ggml_backend_event_free(pf->ready[s]);
            pf->ready[s] = NULL;
        }
        if (pf->retire[s] != NULL) {
            ggml_backend_event_free(pf->retire[s]);
            pf->retire[s] = NULL;
        }
    }
    if (pf->start != NULL) {
        ggml_backend_event_free(pf->start);
        pf->start = NULL;
    }
    for (int i = 0; i < pf->n_timers_cap; i++) {
        if (pf->timers[i] != NULL) {
            ggml_backend_event_free(pf->timers[i]);
        }
    }
    free(pf->timers);
    pf->timers       = NULL;
    pf->n_timers_cap = 0;
    pf->timed        = false;
    pf->armed        = 0;
    if (pf->arena != NULL) {
        ggml_backend_buffer_free(pf->arena);
        pf->arena = NULL;
    }
    pf->arena_size = 0;
    if (pf->copy_backend != NULL) {
        ggml_backend_free(pf->copy_backend);
        pf->copy_backend = NULL;
    }
}

// make sure that the copy queue of a backend exists
static bool ggml_backend_sched_prefetch_queue(ggml_backend_sched_t sched, int backend_id) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    ggml_backend_t backend = sched->backends[backend_id];

    if (pf->copy_backend != NULL) {
        return true;
    }
    pf->copy_backend = ggml_backend_dev_init(backend->device, NULL);
    bool ok = pf->copy_backend != NULL &&
        pf->copy_backend->iface.set_tensor_async != NULL &&
        pf->copy_backend->iface.event_record     != NULL &&
        pf->copy_backend->iface.event_wait       != NULL &&
        ggml_backend_supports_buft(pf->copy_backend, sched->bufts[backend_id]);
    if (ok) {
        pf->start = ggml_backend_event_new(backend->device);
        ok = pf->start != NULL;
    }
    // every slot: the depth of a later binding may be larger (adaptive depth)
    for (int s = 0; ok && s < GGML_SCHED_MAX_PREFETCH_DEPTH; s++) {
        pf->ready[s]  = ggml_backend_event_new(backend->device);
        pf->retire[s] = ggml_backend_event_new(backend->device);
        ok = pf->ready[s] != NULL && pf->retire[s] != NULL;
    }
    if (ok) {
        // the copy queue may run its device-to-device copies on the backend (e.g. CUDA, see ggml_backend_cuda_set_copy_peer)
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(backend->device);
        auto set_copy_peer = (void (*)(ggml_backend_t, ggml_backend_t)) (reg ? ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_copy_peer") : NULL);
        if (set_copy_peer != NULL) {
            set_copy_peer(pf->copy_backend, backend);
        }
        if (sched->pf_timer_new == NULL && reg != NULL) {
            sched->pf_timer_new     = (ggml_backend_event_t (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_timer_new");
            sched->pf_timer_elapsed = (bool (*)(ggml_backend_event_t, ggml_backend_event_t, float *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_timer_elapsed");
            if (sched->pf_timer_new == NULL || sched->pf_timer_elapsed == NULL) {
                sched->pf_timer_new     = NULL;
                sched->pf_timer_elapsed = NULL;
            }
        }
    }
    if (!ok) {
        GGML_LOG_WARN("%s: %s: cannot create a copy queue, host weight prefetch disabled\n", __func__, ggml_backend_name(backend));
        ggml_backend_sched_prefetch_release(sched, backend_id);
        pf->failed = true;
        return false;
    }
    return true;
}

// make sure that the arena of a backend can hold pf->staging bytes (none if every copy is in lent memory)
static size_t ggml_backend_sched_prefetch_staging(ggml_backend_sched_t sched, int backend_id);

// k15-fn: GGML_SCHED_LEND_GUARD=0 restores the allocation failures of merged graphs that turned the host-weight prefetch off
static bool ggml_backend_sched_lend_guard_on() {
    static const bool on = getenv("GGML_SCHED_LEND_GUARD") == nullptr || atoi(getenv("GGML_SCHED_LEND_GUARD")) != 0;
    return on;
}

static bool ggml_backend_sched_prefetch_ensure(ggml_backend_sched_t sched, int backend_id) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    ggml_backend_t backend = sched->backends[backend_id];

    if (pf->staging == 0) {
        if (pf->arena != NULL) {
            // the old arena may still be read or written
            ggml_backend_sched_prefetch_sync(sched);
            ggml_backend_buffer_free(pf->arena);
            pf->arena      = NULL;
            pf->arena_size = 0;
        }
        return true;
    }

    // a binding with only SYNC copies reuses the arena kept from an earlier graph: only an async binding gives the larger arena back
    if (pf->arena != NULL && pf->arena_size >= pf->staging && !(pf->shrink && pf->n_async > 0 && pf->arena_size > pf->staging)) {
        return true;
    }
    if (pf->n_async > 0) {
        pf->shrink = false;
    }

    // a step of the adaptive depth grows the arena past the memory reserved for the graph: only with headroom left on the device for the
    // memory the backends allocate while computing (e.g. the CUDA pool, GGML_SCHED_PREFETCH_ADAPT_HEADROOM_MIB, default 128), else this
    // binding and the next ones use the depth before the step (the copies stay in the arena, never in the graph allocator). The growth
    // itself is exact, the headroom only covers the backends: Qwen3.6 MTP on a 12 GB T4 with 498 MiB free stepped both contexts with 0
    // and ran 16k prompts with 104 MiB left (without the check a 320 MiB arena failed), 512 blocked the step of Flash-Next IQ1_S on 8 GB
    // (694 MiB of growth, 890 MiB free)
    if (pf->depth > sched->pf_depth0 && pf->staging > pf->arena_size) {
        static const size_t headroom = (size_t) std::max(0, getenv("GGML_SCHED_PREFETCH_ADAPT_HEADROOM_MIB") ?
            atoi(getenv("GGML_SCHED_PREFETCH_ADAPT_HEADROOM_MIB")) : 128) * 1024 * 1024;
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        size_t free = 0, total = 0;
        if (dev != NULL) {
            ggml_backend_dev_memory(dev, &free, &total);
        }
        // nor during a graph whose allocation was checked against the free memory before this step (ggml_backend_sched_set_alloc_limit,
        // a merged layer-major graph): the arena would take the memory its compute buffers were measured to fit in (G4, Qwen3.6-35B-A3B
        // UD-Q3_K_M + MTP at 8 GB: the MTP context stepped to depth 3 after a chat, its 432 MiB arena was allocated with the 8k prompt's
        // merged graph, whose compute buffers then failed, which turned the prefetch off for good)
        if (dev != NULL && total > 0 && (free < pf->staging - pf->arena_size + headroom || sched->alloc_limit_on)) {
            const int back = sched->pf_adapt_from > 0 && sched->pf_adapt_from < pf->depth ? sched->pf_adapt_from : sched->pf_depth0;
            GGML_LOG_INFO("%s: %s: depth %d needs %.2f MiB more staging memory, %.2f MiB free (headroom %.0f MiB%s): settled at depth %d\n", __func__,
                ggml_backend_name(backend), pf->depth, (pf->staging - pf->arena_size) / 1024.0 / 1024.0, free / 1024.0 / 1024.0,
                headroom / 1024.0 / 1024.0, sched->alloc_limit_on ? ", in a graph with an allocation limit" : "", back);
            sched->pf_depth = back;
            sched->pf_adapt = false;
            pf->depth   = back;
            pf->staging = ggml_backend_sched_prefetch_staging(sched, backend_id);
            if (pf->arena != NULL && pf->arena_size >= pf->staging) {
                return true;
            }
        }
    }

    // the old arena (if any) may still be read or written
    ggml_backend_sched_prefetch_sync(sched);

    if (pf->arena != NULL) {
        ggml_backend_buffer_free(pf->arena);
        pf->arena      = NULL;
        pf->arena_size = 0;
    }

    pf->arena = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], pf->staging);
    // k15-fn GGML_SCHED_PREFETCH_FAIL_AT=N (debug): the Nth staging allocation fails, as cudaMalloc does when the device is full
    {
        static const int fail_at = getenv("GGML_SCHED_PREFETCH_FAIL_AT") ? atoi(getenv("GGML_SCHED_PREFETCH_FAIL_AT")) : 0;
        static int n_ensure = 0;
        if (fail_at > 0 && ++n_ensure == fail_at && pf->arena != NULL) {
            GGML_LOG_WARN("%s: GGML_SCHED_PREFETCH_FAIL_AT=%d: staging allocation %d made to fail\n", __func__, fail_at, n_ensure);
            ggml_backend_buffer_free(pf->arena);
            pf->arena = NULL;
        }
    }
    if (pf->arena == NULL) {
        if (pf->depth > sched->pf_depth0) {
            // a step of the adaptive depth: the next binding uses the starting depth again
            GGML_LOG_WARN("%s: %s: failed to allocate %.2f MiB of staging memory for depth %d, back to depth %d\n",
                __func__, ggml_backend_name(backend), pf->staging / 1024.0 / 1024.0, pf->depth, sched->pf_depth0);
            sched->pf_depth = sched->pf_depth0;
            sched->pf_adapt = false;
            ggml_backend_sched_prefetch_release(sched, backend_id);
            return false;
        }
        // k15-fn (GGML_SCHED_LEND_GUARD=0 off): in a merged graph (its caller splits a refused graph) the staging that neither the lender
        // took nor fits in free memory refuses the graph instead of turning the prefetch off for the rest of the process
        if (ggml_backend_sched_lend_guard_on() && sched->alloc_limit_on) {
            GGML_LOG_INFO("%s: %s: %.2f MiB of staging memory do not allocate in a merged graph: the graph is refused, the prefetch kept\n",
                __func__, ggml_backend_name(backend), pf->staging / 1024.0 / 1024.0);
            ggml_backend_sched_prefetch_release(sched, backend_id);
            pf->ensure_refused = true;
            return false;
        }
        GGML_LOG_WARN("%s: %s: failed to allocate %.2f MiB of staging memory, host weight prefetch disabled\n",
            __func__, ggml_backend_name(backend), pf->staging / 1024.0 / 1024.0);
        ggml_backend_sched_prefetch_release(sched, backend_id);
        pf->failed = true;
        return false;
    }
    // the copies behave like graph allocator tensors (e.g. kernels clear the padding of quantized tensors in compute buffers)
    ggml_backend_buffer_set_usage(pf->arena, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    pf->arena_size = pf->staging;
    sched->pf_check = true;

    GGML_LOG_INFO("%s: %s host weight prefetch: depth = %d, staging = %.2f MiB (largest split %.2f MiB), %.2f MiB in lent memory, eligible splits = %d (%d async) of %d\n",
        __func__, ggml_backend_name(backend), pf->depth, pf->arena_size / 1024.0 / 1024.0, pf->need / 1024.0 / 1024.0,
        pf->lent / 1024.0 / 1024.0, pf->n_bound, pf->n_async, sched->n_splits);

    return true;
}

// depth > 2, byte ring: each split is placed after the previous one (or at the arena start when it does not fit before the end) and must not
// overlap the depth - 1 splits before it, which may still be live when its copy is issued; the copy waits for the split depth before it
// (the slot ordinal % depth), so every earlier split it overlaps has been computed (compute is in order)
static bool ggml_backend_sched_prefetch_ring(ggml_backend_sched_t sched, int backend_id, int depth, size_t arena, bool assign) {
    const struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    size_t off[GGML_SCHED_MAX_PREFETCH_DEPTH];
    size_t len[GGML_SCHED_MAX_PREFETCH_DEPTH];
    size_t cur = 0;
    for (int k = 0; k < pf->n_bound; k++) {
        struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[k]];
        const size_t size = split->pf_size;
        if (size > arena) {
            return false;
        }
        if (cur + size > arena) {
            cur = 0;
        }
        for (int j = std::max(0, k - depth + 1); j < k; j++) {
            const size_t o = off[j % depth];
            const size_t l = len[j % depth];
            if (size > 0 && l > 0 && cur < o + l && o < cur + size) {
                return false;
            }
        }
        off[k % depth] = cur;
        len[k % depth] = size;
        if (assign) {
            split->pf_offset = cur;
        }
        cur += size;
    }
    return true;
}

// arena bytes for the prefetched splits of a backend at `depth` slots: depth = 2 needs the largest sum of two consecutive splits,
// depth > 2 the smallest byte ring that holds every `depth` consecutive splits (GGML_SCHED_PREFETCH_WINDOW, else depth x largest split)
// ring: set if the byte ring is used, the offsets of the splits are then assigned when `assign`
static size_t ggml_backend_sched_prefetch_layout(ggml_backend_sched_t sched, int backend_id, int depth, bool * ring, bool assign) {
    const struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    size_t need = 0;
    size_t staging = 0;
    if (ring) {
        *ring = false;
    }
    for (int k = 0; k < pf->n_bound; k++) {
        need = std::max(need, sched->splits[pf->split_ids[k]].pf_size);
    }
    if (depth == 2) {
        for (int k = 0; k < pf->n_bound; k++) {
            size_t live = sched->splits[pf->split_ids[k]].pf_size;
            if (k + 1 < pf->n_bound) {
                live += sched->splits[pf->split_ids[k + 1]].pf_size;
            }
            staging = std::max(staging, live);
        }
        return staging;
    }
    const size_t uniform = depth * need;
    if (depth < 2 || !sched->pf_window || need == 0) {
        return uniform;
    }
    // smallest ring: from the largest sum of `depth` consecutive splits, in steps of 1/8 of the largest split
    size_t window = 0;
    for (int k = 0; k < pf->n_bound; k++) {
        size_t live = 0;
        for (int j = k; j < std::min(pf->n_bound, k + depth); j++) {
            live += sched->splits[pf->split_ids[j]].pf_size;
        }
        window = std::max(window, live);
    }
    const size_t align = ggml_backend_buft_get_alignment(sched->bufts[backend_id]);
    const size_t step  = GGML_PAD(std::max(need / 8, align), align);
    for (size_t arena = GGML_PAD(window, align); arena < uniform; arena += step) {
        if (ggml_backend_sched_prefetch_ring(sched, backend_id, depth, arena, false)) {
            if (assign) {
                ggml_backend_sched_prefetch_ring(sched, backend_id, depth, arena, true);
            }
            if (ring) {
                *ring = true;
            }
            return arena;
        }
    }
    return uniform;
}

static size_t ggml_backend_sched_prefetch_staging(ggml_backend_sched_t sched, int backend_id) {
    return ggml_backend_sched_prefetch_layout(sched, backend_id, sched->pf[backend_id].depth, NULL, false);
}

// the user takes back the memory it lent for the copies of a backend
static void ggml_backend_sched_lend_end(ggml_backend_sched_t sched, int backend_id) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    ggml_gallocr_set_external(sched->galloc, backend_id, NULL, 0, 0);
    pf->ext_buf = NULL;
    if (pf->lending && sched->callback_lend != NULL) {
        sched->callback_lend(sched->backends[backend_id], NULL, 0, pf->depth, false, sched->callback_lend_user_data);
    }
    pf->lending       = false;
    pf->n_lent_bufs   = 0;
    pf->n_lent_copies = 0;
    pf->lent        = 0;
}

static bool ggml_backend_sched_is_lent(const struct ggml_backend_sched_prefetch * pf, const struct ggml_tensor * t) {
    // with LLAMA_MOE_CACHE_LEND=2 the graph allocator places tensors (e.g. other inputs of a prefetched split) in the same buffer
    if (t->buffer == pf->ext_buf && t->buffer != NULL) {
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(t->buffer) + pf->ext_offset;
        if ((uintptr_t) t->data >= base && (uintptr_t) t->data < base + pf->ext_size) {
            return false;
        }
    }
    for (int i = 0; i < pf->n_lent_bufs; i++) {
        if (t->buffer == pf->lent_bufs[i]) {
            return true;
        }
    }
    return false;
}

// bind the weight copies of the prefetched splits to their staging slots, must run after splitting the graph and before the graph allocator
static char ggml_backend_sched_dry_placeholder;

// dry: only measure (no device memory is allocated), the copies get a placeholder address so that the graph allocator skips them
static void ggml_backend_sched_prefetch_bind(ggml_backend_sched_t sched, bool dry) {
    sched->pf_active = false;
    sched->pf_gen++;
    if (!dry) {
        sched->pf_prev_runs = sched->pf_runs;
        sched->pf_runs      = 0;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        sched->pf[b].depth   = sched->pf_depth;
        sched->pf[b].ring    = false;
        sched->pf[b].armed   = 0;
        sched->pf[b].n_bound = 0;
        sched->pf[b].n_async = 0;
        sched->pf[b].need    = 0;
        sched->pf[b].staging = 0;
        if (!dry) {
            sched->pf[b].ensure_refused = false;
            sched->pf[b].n_lent_bufs   = 0;
            sched->pf[b].n_lent_copies = 0;
            sched->pf[b].lent          = 0;
        }
    }
    for (int i = 0; i < sched->n_splits; i++) {
        sched->splits[i].pf_ordinal = -1;
        sched->splits[i].pf_async   = false;
        sched->splits[i].pf_lent    = false;
        sched->splits[i].pf_offset  = 0;
    }

    // backends whose user lent memory for the previous graph and lends none for this one take it back
    bool ext[GGML_SCHED_MAX_BACKENDS] = { false }; // the graph allocator of the backend uses lent memory
    auto lend_end_unused = [&]() {
        if (dry) {
            return;
        }
        for (int b = 0; b < sched->n_backends; b++) {
            if (!ext[b]) {
                ggml_gallocr_set_external(sched->galloc, b, NULL, 0, 0);
                sched->pf[b].ext_buf = NULL;
            }
            if (sched->pf[b].lending && sched->pf[b].n_lent_bufs == 0) {
                ggml_backend_sched_lend_end(sched, b);
            }
        }
    };

    // pipeline parallelism keeps n_copies versions of every input, it is never used together with host weights
    if (sched->n_copies > 1) {
        lend_end_unused();
        return;
    }
    // without prefetch (GGML_SCHED_PREFETCH=0) the user may still lend memory to the graph allocator
    const bool prefetch = sched->pf_mode != 0;

    // host weights written by the graph, and the first split that writes other host tensors
    std::unordered_set<const ggml_tensor *> written;
    std::unordered_map<const ggml_tensor *, int> first_write;
    for (int i = 0; i < sched->n_splits; i++) {
        const struct ggml_backend_sched_split * split = &sched->splits[i];
        for (int n = 0; n < split->graph.n_nodes; n++) {
            const struct ggml_tensor * node = split->graph.nodes[n];
            if (ggml_op_is_empty(node->op)) {
                continue;
            }
            const struct ggml_tensor * root = ggml_backend_sched_view_root(node);
            if (ggml_backend_sched_is_host_weight(root)) {
                written.insert(root);
            } else if (root != node && root->buffer != NULL && ggml_backend_buffer_is_host(root->buffer)) {
                first_write.emplace(root, i);
            }
        }
    }

    // candidate copies, the split that copies them and how
    std::unordered_map<const ggml_tensor *, int> owner;
    std::unordered_set<const ggml_tensor *> sync_copies;
    for (int i = 0; i < sched->n_splits && prefetch; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        if (!ggml_backend_sched_prefetch_backend_ok(sched, split->backend_id)) {
            continue;
        }
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            const enum ggml_backend_sched_prefetch_mode mode = ggml_backend_sched_prefetch_candidate(sched, split, input, written, first_write, i);
            if (mode != GGML_BACKEND_SCHED_PREFETCH_NONE) {
                const ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
                owner[input_cpy] = i;
                if (mode == GGML_BACKEND_SCHED_PREFETCH_SYNC) {
                    sync_copies.insert(input_cpy);
                }
            }
        }
    }

    // a weight used by several splits of the same backend has a single copy, made by the first split
    // the later splits read it after the slot may have been reused, so the copy stays with the graph allocator
    std::unordered_set<const ggml_tensor *> shared;
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        for (int n = 0; n < split->graph.n_nodes; n++) {
            const struct ggml_tensor * node = split->graph.nodes[n];
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                for (const struct ggml_tensor * t = node->src[k]; t != NULL; t = t->view_src) {
                    auto it = owner.find(t);
                    if (it != owner.end() && it->second != i) {
                        shared.insert(t);
                    }
                }
            }
        }
    }

    // the copies of split i made by its prefetch
    auto is_split_copy = [&](const ggml_tensor * input_cpy, int i) {
        auto it = owner.find(input_cpy);
        return it != owner.end() && it->second == i && shared.count(input_cpy) == 0;
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        const int b = split->backend_id;
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        const size_t align = ggml_backend_buft_get_alignment(sched->bufts[b]);

        size_t bytes = 0;
        bool   async = true;
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], b, sched->cur_copy);
            if (!is_split_copy(input_cpy, i)) {
                continue;
            }
            bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(sched->bufts[b], input_cpy), align);

            // e.g. small MoE batches upload only the routed experts, which needs the ids computed by an earlier split
            if (sync_copies.count(input_cpy) > 0) {
                async = false;
            }
        }
        // a split with a SYNC copy is not prefetched: its copies are made when it starts, like the regular path, but still in a staging slot
        if (bytes == 0) {
            continue;
        }

        if (pf->n_bound >= pf->split_ids_capacity) {
            pf->split_ids_capacity = std::max(16, 2*pf->split_ids_capacity);
            pf->split_ids = (int *) realloc(pf->split_ids, pf->split_ids_capacity*sizeof(int));
            GGML_ASSERT(pf->split_ids != NULL);
        }
        split->pf_ordinal = pf->n_bound;
        split->pf_async   = async;
        split->pf_size    = bytes; // bytes in the arena, without the copies placed in lent memory below
        pf->split_ids[pf->n_bound++] = i;
        pf->n_async += async ? 1 : 0;
    }

    // only SYNC copies and no arena to reuse: the graph allocator holds them (e.g. prefetch disabled by the threshold), no arena is created for them
    // (before the lend below, which then neither measures nor places them)
    // with a lender the arena is missing because the copies of the large graphs live in lent memory (e.g. MoE cache slots): the graph
    // allocator would have to grow by every copy of the graph after the lent memory was handed out (e.g. the last, shorter ubatch of a
    // prompt), which the device may not hold; the lender is asked to place them first (they share one slot, see sync_lend below), and only
    // if it does not place all of them they go to the graph allocator as before
    // GGML_SCHED_LEND_SYNC=1 (default): only when the growth of the graph allocator does not fit in the free memory of the device (with
    // GGML_SCHED_PREFETCH_HEADROOM_MIB, default 512, for what the backends allocate while computing), so a binding the graph allocator
    // already holds (e.g. the reserved seeding ubatch of the MoE cache) or can hold keeps the old path; 2: always; 0: never
    bool sync_lend[GGML_SCHED_MAX_BACKENDS] = { false };
    bool any_sync_lend = false;
    for (int b = 0; b < sched->n_backends; b++) {
        const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        sync_lend[b] = pf->n_bound > 0 && pf->n_async == 0 && !dry && pf->arena == NULL && sched->callback_lend != NULL && sched->lend_sync != 0;
        any_sync_lend = any_sync_lend || sync_lend[b];
    }
    if (any_sync_lend && sched->lend_sync == 1) {
        // the graph allocation with the SYNC copies of these backends in it (the bound copies of the other backends are not allocated by it)
        static char sync_placeholder;
        std::vector<ggml_tensor *> bound;
        for (int b = 0; b < sched->n_backends; b++) {
            const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
            for (int k = 0; k < pf->n_bound && !sync_lend[b]; k++) {
                const struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[k]];
                for (int j = 0; j < split->n_inputs; j++) {
                    struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], b, sched->cur_copy);
                    if (is_split_copy(input_cpy, pf->split_ids[k]) && input_cpy->data == NULL) {
                        input_cpy->data = &sync_placeholder;
                        bound.push_back(input_cpy);
                    }
                }
            }
        }
        size_t sizes[GGML_SCHED_MAX_BACKENDS] = { 0 };
        ggml_gallocr_measure_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
        for (ggml_tensor * t : bound) {
            t->data = NULL;
        }
        static const size_t headroom = [] {
            const char * env = getenv("GGML_SCHED_PREFETCH_HEADROOM_MIB");
            return (size_t) (env != NULL ? std::max(0, atoi(env)) : 512) * 1024 * 1024;
        }();
        for (int b = 0; b < sched->n_backends; b++) {
            if (!sync_lend[b]) {
                continue;
            }
            const size_t cur  = ggml_gallocr_get_buffer_size(sched->galloc, b);
            const size_t grow = sizes[b] > cur ? sizes[b] - cur : 0;
            size_t free = 0, total = 0;
            ggml_backend_dev_t dev = ggml_backend_get_device(sched->backends[b]);
            if (grow > 0 && dev != NULL) {
                ggml_backend_dev_memory(dev, &free, &total);
            }
            const bool fits = grow == 0 || dev == NULL || total == 0 || free >= grow + headroom;
            if (fits) {
                sync_lend[b] = false;
            } else if (!sched->pf[b].sync_lend_log) {
                GGML_LOG_INFO("%s: %s: the SYNC copies of a graph need %.2f MiB more graph allocation, %.2f MiB free (headroom %.0f MiB): "
                    "they are placed in lent memory (GGML_SCHED_LEND_SYNC=0 to disable)\n", __func__, ggml_backend_name(sched->backends[b]),
                    grow/1024.0/1024.0, free/1024.0/1024.0, headroom/1024.0/1024.0);
                sched->pf[b].sync_lend_log = true;
            }
        }
    }
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        if (sync_lend[b]) {
            continue;
        }
        if (pf->n_bound > 0 && pf->n_async == 0 && (dry || pf->arena == NULL)) {
            for (int k = 0; k < pf->n_bound; k++) {
                sched->splits[pf->split_ids[k]].pf_ordinal = -1;
            }
            pf->n_bound = 0;
            pf->need    = 0;
        }
    }

    // memory lent by the user (e.g. free slots of a cache): the copies it places do not use the arena
    // and a graph that does not fit in the buffer of the graph allocator (e.g. the activations of a large batch) may be allocated there
    size_t gsize[GGML_SCHED_MAX_BACKENDS] = { 0 };
    for (int b = 0; b < sched->n_backends; b++) {
        sched->pf[b].own_buf  = NULL;
        sched->pf[b].own_size = 0;
        sched->pf[b].own_rest = 0;
    }
    if (sched->callback_lend != NULL && !dry) {
        // the prefetched copies are not allocated by the graph allocator
        static char measure_placeholder;
        std::vector<ggml_tensor *> bound;
        for (int b = 0; b < sched->n_backends; b++) {
            const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
            for (int k = 0; k < pf->n_bound; k++) {
                const struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[k]];
                for (int j = 0; j < split->n_inputs; j++) {
                    struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], b, sched->cur_copy);
                    if (is_split_copy(input_cpy, pf->split_ids[k]) && input_cpy->data == NULL) {
                        input_cpy->data = &measure_placeholder;
                        bound.push_back(input_cpy);
                    }
                }
            }
        }
        size_t sizes[GGML_SCHED_MAX_BACKENDS] = { 0 };
        ggml_gallocr_measure_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
        for (ggml_tensor * t : bound) {
            t->data = NULL;
        }
        for (int b = 0; b < sched->n_backends; b++) {
            if (ggml_backend_sched_prefetch_backend_ok(sched, b) && sizes[b] > ggml_gallocr_get_buffer_size(sched->galloc, b)) {
                gsize[b] = GGML_PAD(sizes[b], ggml_backend_buft_get_alignment(sched->bufts[b]));
            }
        }
        // k15-fn (GGML_SCHED_LEND_OWN=0 off): the own buffer as the first region of a lent graph allocation, the user lends the rest; the
        // rest is measured with the plan the allocator then makes (the own buffer limits the first chunk, the lent region is not a limit)
        static const bool lend_own_env = getenv("GGML_SCHED_LEND_OWN") == nullptr || atoi(getenv("GGML_SCHED_LEND_OWN")) != 0;
        for (int b = 0; b < sched->n_backends && lend_own_env; b++) {
            struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
            pf->own_buf  = NULL;
            pf->own_size = 0;
            pf->own_rest = 0;
            ggml_backend_buffer_t ob = NULL;
            size_t os = 0;
            const size_t align = ggml_backend_buft_get_alignment(sched->bufts[b]);
            if (gsize[b] == 0 || !ggml_gallocr_get_own_region(sched->galloc, b, &ob, &os) || os < align || ggml_backend_buffer_get_type(ob) != sched->bufts[b]) {
                continue;
            }
            size_t s0 = 0;
            size_t rest = 0;
            for (ggml_tensor * t : bound) {
                t->data = &measure_placeholder;
            }
            ggml_gallocr_measure_limited(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, b, os, &s0, &rest);
            for (ggml_tensor * t : bound) {
                t->data = NULL;
            }
            if (rest == SIZE_MAX || rest == 0 || s0 > os || GGML_PAD(rest, align) >= gsize[b]) {
                continue;
            }
            pf->own_buf  = ob;
            pf->own_size = os;
            pf->own_rest = GGML_PAD(rest, align);
        }
    }

    std::unordered_map<const ggml_tensor *, std::pair<ggml_backend_buffer_t, size_t>> lent;
    for (int i = 0; i < sched->n_splits && !dry; i++) {
        sched->splits[i].pf_lent = false;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        const size_t staging_all = ggml_backend_sched_prefetch_staging(sched, b);
        if (!dry) {
            pf->lent_all = false;
        }
        if (sched->callback_lend == NULL || (pf->n_bound == 0 && gsize[b] == 0) || (!dry && pf->n_bound > 0 && !ggml_backend_sched_prefetch_queue(sched, b))) {
            continue;
        }
        const size_t align = ggml_backend_buft_get_alignment(sched->bufts[b]);
        int n_copies = 0;
        auto push = [&](const struct ggml_backend_sched_lend_copy & lc) {
            if (n_copies >= pf->lend_copies_capacity) {
                pf->lend_copies_capacity = std::max(64, 2*pf->lend_copies_capacity);
                pf->lend_copies = (struct ggml_backend_sched_lend_copy *) realloc(pf->lend_copies, pf->lend_copies_capacity*sizeof(pf->lend_copies[0]));
                GGML_ASSERT(pf->lend_copies != NULL);
            }
            pf->lend_copies[n_copies++] = lc;
        };
        // the graph allocator: live during the whole graph, as GGML_SCHED_MAX_GRAPH_PARTS entries first with src = NULL, ordinal -1 and
        // size = the bytes of the whole allocation; the user may place it in several parts, at most one per entry, and sets size to the
        // size of the part (the graph allocator plans the parts in order, a tensor never spans two parts), unused entries keep buffer NULL
        // k15-fn: with the own buffer as the first region (measured above) the user lends one part of the rest; if it cannot, the whole
        // allocation as before (lend_full). G4 Qwen3.8-Flash-Next IQ1_S 16 GB + device MTP drafter, 16k prompt: a 3316 MiB graph needed 2227
        // slots, over the cache's cap (2003), so the group ran as two graphs streaming the experts twice; its own buffer held 2244 MiB
        bool lend_full = pf->own_buf == NULL || dry;
        // k17-lendoom (GGML_SCHED_LEND_FREE_CHECK=0 off): with less device memory free than GGML_SCHED_PREFETCH_HEADROOM_MIB (default 512,
        // what the backends allocate while computing, e.g. the CUDA pool of mul_mat_q) the user is asked for the whole graph allocation in
        // one part first, and the own buffer is freed for it; if it cannot, the own region and the rest as before. G4 Qwen3.8-Flash-Next
        // 24 GB, 16k context: a chat prompt grew the own buffer by its SYNC copies (847 MiB of 976 MiB free, the cache does not lend to
        // small batches), the 8k prompt after it used that 861 MiB buffer as the first region and the CUDA pool ran out of memory
        // (only an own buffer of 64 MiB or more: a small one, e.g. of the decode graphs, frees little and is allocated again every prompt)
        static const bool lend_free_check = getenv("GGML_SCHED_LEND_FREE_CHECK") == nullptr || atoi(getenv("GGML_SCHED_LEND_FREE_CHECK")) != 0;
        static const size_t lend_headroom = [] {
            const char * env = getenv("GGML_SCHED_PREFETCH_HEADROOM_MIB");
            return (size_t) (env != NULL ? std::max(0, atoi(env)) : 512) * 1024 * 1024;
        }();
        bool   full_first = false;
        bool   free_own   = false;
        size_t full_free  = 0;
        size_t full_own   = 0;
        if (lend_free_check && !dry && gsize[b] > 0) {
            ggml_backend_buffer_t ob = NULL;
            size_t os = 0;
            ggml_backend_dev_t dev = ggml_backend_get_device(sched->backends[b]);
            if (dev != NULL && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                    ggml_gallocr_get_own_region(sched->galloc, b, &ob, &os) && os >= 64ull*1024*1024) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                if (total > 0 && free < lend_headroom) {
                    full_first = true;
                    full_free  = free;
                    full_own   = os;
                    lend_full  = true;
                }
            }
        }
        lend_again:
        ggml_backend_buffer_t own_buf  = lend_full ? NULL : pf->own_buf;
        const size_t          own_size = lend_full ? 0 : pf->own_size;
        const size_t          gask     = lend_full ? gsize[b] : pf->own_rest;
        const int             n_graph  = gsize[b] > 0 ? (lend_full ? GGML_SCHED_MAX_GRAPH_PARTS : 1) : 0;
        n_copies = 0;
        for (int k = 0; k < n_graph; k++) {
            push({ NULL, gask, -1, NULL, 0 });
        }
        for (int k = 0; k < pf->n_bound; k++) {
            const int i = pf->split_ids[k];
            const struct ggml_backend_sched_split * split = &sched->splits[i];
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], b, sched->cur_copy);
                if (!is_split_copy(input_cpy, i)) {
                    continue;
                }
                push({ split->inputs[j], GGML_PAD(ggml_backend_buft_get_alloc_size(sched->bufts[b], input_cpy), align), k, NULL, 0 });
            }
        }
        // SYNC copies are written on the compute stream when their split starts, after the splits before it: like in the arena (offset 0
        // for every split) the copies of different splits may share bytes, depth 1 for the lender
        const int lend_depth = sync_lend[b] ? 1 : pf->depth;
        sched->callback_lend(sched->backends[b], pf->lend_copies, n_copies, lend_depth, dry, sched->callback_lend_user_data);
        if (full_first && n_graph > 0) {
            full_first = false;
            bool one = pf->lend_copies[0].buffer != NULL && pf->lend_copies[0].size >= gsize[b];
            for (int k = 1; k < n_graph; k++) {
                one = one && (pf->lend_copies[k].buffer == NULL || pf->lend_copies[k].size == 0);
            }
            if (one) {
                free_own = true;
            } else if (pf->own_buf != NULL) {
                // not in one part: the own region and the rest, as without the check
                lend_full = false;
                goto lend_again;
            }
        }
        if (!lend_full && n_graph > 0 && (pf->lend_copies[0].buffer == NULL || pf->lend_copies[0].size < gask)) {
            // the user could not lend the rest in one part: the whole allocation, as without the own region
            lend_full = true;
            goto lend_again;
        }

        // a dry run only measures, the user may not have allocated its memory (e.g. no_alloc)
        auto valid = [&](const struct ggml_backend_sched_lend_copy & lc) {
            if (lc.buffer == NULL || ggml_backend_buffer_get_type(lc.buffer) != sched->bufts[b] || (!dry &&
                    (((uintptr_t) ggml_backend_buffer_get_base(lc.buffer) + lc.offset) % align != 0 ||
                     lc.offset + lc.size > ggml_backend_buffer_get_size(lc.buffer)))) {
                return false;
            }
            if (!dry && std::find(pf->lent_bufs, pf->lent_bufs + pf->n_lent_bufs, lc.buffer) == pf->lent_bufs + pf->n_lent_bufs) {
                if (pf->n_lent_bufs >= GGML_SCHED_MAX_LENT_BUFFERS) {
                    return false;
                }
                pf->lent_bufs[pf->n_lent_bufs++] = lc.buffer;
            }
            return true;
        };

        int c = 0;
        bool any = false;
        if (n_graph > 0) {
            ggml_backend_buffer_t gbuf = NULL;
            size_t offs [GGML_SCHED_MAX_GRAPH_PARTS + 1];
            size_t sizes[GGML_SCHED_MAX_GRAPH_PARTS + 1];
            ggml_backend_buffer_t gbufs[GGML_SCHED_MAX_GRAPH_PARTS + 1];
            int n_parts = 0;
            size_t total = 0;
            bool ok = true;
            if (own_buf != NULL) {
                gbufs[0] = own_buf;
                offs [0] = 0;
                sizes[0] = own_size;
                n_parts  = 1;
            }
            const int part0 = n_parts;
            for (int k = 0; k < n_graph; k++) {
                const struct ggml_backend_sched_lend_copy & lc = pf->lend_copies[c++];
                if (lc.buffer == NULL || lc.size == 0) {
                    continue;
                }
                if (dry || !valid(lc) || (gbuf != NULL && lc.buffer != gbuf)) {
                    ok = false;
                    continue;
                }
                gbuf = lc.buffer;
                gbufs[n_parts] = lc.buffer;
                offs [n_parts] = lc.offset;
                sizes[n_parts] = lc.size;
                total += lc.size;
                n_parts++;
            }
            if (!dry && ok && n_parts > part0) {
                if (free_own && n_parts == 1 && sizes[0] >= gsize[b]) {
                    // the whole graph is in lent memory: the own buffer is not needed by it, the graphs after it allocate it again
                    // (the graph before it may still read it)
                    ggml_backend_synchronize(sched->backends[b]);
                    ggml_gallocr_free_buffer(sched->galloc, b);
                    pf->own_buf  = NULL;
                    pf->own_size = 0;
                    pf->own_rest = 0;
                    GGML_LOG_INFO("%s: %s: %.2f MiB of device memory free, below the headroom of %.0f MiB: the graph allocation of %.2f MiB "
                        "is lent in one part and the own compute buffer of %.2f MiB is freed (GGML_SCHED_LEND_FREE_CHECK=0 to disable)\n", __func__,
                        ggml_backend_name(sched->backends[b]), full_free/1024.0/1024.0, lend_headroom/1024.0/1024.0, gsize[b]/1024.0/1024.0,
                        full_own/1024.0/1024.0);
                }
                ggml_gallocr_set_external_bufs(sched->galloc, b, gbufs, offs, sizes, n_parts);
                ggml_gallocr_set_external_last_free(sched->galloc, b, part0 > 0);
                // the first lent part (with more parts the list of the prefetched copies tells them apart, see ggml_backend_sched_prefetch_is_bound)
                pf->ext_buf    = gbuf;
                pf->ext_offset = offs[part0];
                pf->ext_size   = sizes[part0];
                pf->lent_all   = true;
                ext[b] = true;
                any    = true;
                if (total != pf->ext_prev) {
                    GGML_LOG_INFO("%s: %s: graph allocation of %.2f MiB in lent memory, %d part%s of %.2f MiB (own compute buffer %.2f MiB%s)\n", __func__,
                        ggml_backend_name(sched->backends[b]), gsize[b] / 1024.0 / 1024.0, n_parts - part0, n_parts - part0 > 1 ? "s" : "",
                        total / 1024.0 / 1024.0, ggml_gallocr_get_buffer_size(sched->galloc, b) / 1024.0 / 1024.0, part0 > 0 ? ", the first region" : "");
                    pf->ext_prev = total;
                }
            }
        }
        for (int k = 0; k < pf->n_bound; k++) {
            const int i = pf->split_ids[k];
            struct ggml_backend_sched_split * split = &sched->splits[i];
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], b, sched->cur_copy);
                if (!is_split_copy(input_cpy, i)) {
                    continue;
                }
                const struct ggml_backend_sched_lend_copy & lc = pf->lend_copies[c++];
                if (!valid(lc)) {
                    continue;
                }
                lent[input_cpy] = { lc.buffer, lc.offset };
                split->pf_size -= lc.size;
                split->pf_lent = split->pf_lent || !dry;
                any = true;
            }
        }
        GGML_ASSERT(c == n_copies);
        // GGML_SCHED_LEND_CHECK=1 (debug): the placements of copies that are live together must not overlap
        static const bool lend_check = getenv("GGML_SCHED_LEND_CHECK") != nullptr && atoi(getenv("GGML_SCHED_LEND_CHECK")) != 0;
        if (lend_check && !dry) {
            int n_bad = 0;
            for (int i = 0; i < n_copies; i++) {
                const struct ggml_backend_sched_lend_copy & a = pf->lend_copies[i];
                if (a.buffer == NULL) {
                    continue;
                }
                for (int j = i + 1; j < n_copies; j++) {
                    const struct ggml_backend_sched_lend_copy & d = pf->lend_copies[j];
                    if (d.buffer != a.buffer) {
                        continue;
                    }
                    const bool live = a.ordinal < 0 || d.ordinal < 0 || abs(a.ordinal - d.ordinal) < lend_depth;
                    if (live && a.offset < d.offset + d.size && d.offset < a.offset + a.size) {
                        if (n_bad++ < 4) {
                            GGML_LOG_ERROR("%s: lent placements overlap: [%zu, %zu) ordinal %d and [%zu, %zu) ordinal %d\n", __func__,
                                a.offset, a.offset + a.size, a.ordinal, d.offset, d.offset + d.size, d.ordinal);
                        }
                    }
                }
            }
            GGML_LOG_INFO("%s: lend check: %d copies, %d overlaps\n", __func__, n_copies, n_bad);
        }
        // SYNC copies the lender did not all place: no arena is created for them, the graph allocator holds all of them (as without a lender)
        if (sync_lend[b]) {
            bool all_placed = true;
            for (int k = 0; k < pf->n_bound; k++) {
                all_placed = all_placed && sched->splits[pf->split_ids[k]].pf_size == 0;
            }
            if (!all_placed) {
                for (int k = 0; k < pf->n_bound; k++) {
                    struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[k]];
                    for (int j = 0; j < split->n_inputs; j++) {
                        lent.erase(tensor_copy(split->inputs[j], b, sched->cur_copy));
                    }
                    split->pf_lent    = false;
                    split->pf_ordinal = -1;
                }
                pf->n_bound = 0;
                pf->need    = 0;
                if (!ext[b]) {
                    pf->n_lent_bufs = 0;
                    any = false;
                }
            }
        }
        if (!dry) {
            pf->lent = pf->n_bound > 0 ? staging_all - ggml_backend_sched_prefetch_staging(sched, b) : 0;
        }
        if (!dry && any) {
            pf->lending = true;
        }
    }
    lend_end_unused();

    // arena size: two consecutive splits are live at once with depth = 2 (the copy of k + 1 overlaps the compute of k)
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        for (int k = 0; k < pf->n_bound; k++) {
            pf->need = std::max(pf->need, sched->splits[pf->split_ids[k]].pf_size);
        }
        if (pf->n_bound > 0 && pf->n_async == 0) {
            // the SYNC copies of a split are made on the compute stream after the previous split, so all of them can use the same slot
            // the arena of an earlier graph holds them: the graph allocator never stages the same weights next to it (e.g. small batches, the ubatch that seeds the MoE cache)
            pf->staging = pf->need;
            continue;
        }
        pf->staging = ggml_backend_sched_prefetch_staging(sched, b);
    }

    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];

        if (pf->n_bound == 0) {
            continue;
        }

        if (!dry && (!ggml_backend_sched_prefetch_queue(sched, b) || !ggml_backend_sched_prefetch_ensure(sched, b))) {
            for (int k = 0; k < pf->n_bound; k++) {
                sched->splits[pf->split_ids[k]].pf_ordinal = -1;
                sched->splits[pf->split_ids[k]].pf_async   = false;
            }
            pf->n_bound = 0;
            pf->n_async = 0;
            pf->need    = 0;
            pf->staging = 0;
            ggml_backend_sched_lend_end(sched, b);
            continue;
        }

        // like the graph allocator, release the per-tensor state of the previous graph (e.g. tensor extras)
        if (!dry && pf->arena != NULL) {
            ggml_backend_buffer_reset(pf->arena);
        }

        // only SYNC copies: they are written on the compute stream, which does not wait for the copy queue (e.g. the copies armed for another binding)
        if (!dry && pf->n_async == 0 && pf->pending && pf->copy_backend != NULL) {
            ggml_backend_synchronize(pf->copy_backend);
            pf->pending = false;
        }

        if (!dry && pf->lent != pf->lent_prev) {
            GGML_LOG_INFO("%s: %s host weight prefetch: depth = %d, staging = %.2f MiB (%.2f MiB in lent memory), eligible splits = %d (%d async) of %d\n",
                __func__, ggml_backend_name(sched->backends[b]), pf->depth, pf->staging / 1024.0 / 1024.0, pf->lent / 1024.0 / 1024.0,
                pf->n_bound, pf->n_async, sched->n_splits);
            pf->lent_prev = pf->lent;
        }

        // depth > 2: the byte ring offsets of the final sizes (the arena may be larger, kept from an earlier graph)
        if (pf->depth > 2) {
            ggml_backend_sched_prefetch_layout(sched, b, pf->depth, &pf->ring, true);
        }

        const size_t align = ggml_backend_buft_get_alignment(sched->bufts[b]);
        for (int k = 0; k < pf->n_bound; k++) {
            struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[k]];
            // depth = 2: odd ordinals end at the arena end so consecutive splits never overlap (the arena size is only known when not dry)
            size_t offset = (size_t) (k % pf->depth) * pf->need;
            if (pf->n_async == 0) {
                offset = 0;
            } else if (pf->depth == 2) {
                offset = k % 2 == 0 ? 0 : (dry ? pf->staging : pf->arena_size) - split->pf_size;
            } else if (pf->ring) {
                offset = split->pf_offset;
            }
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], b, sched->cur_copy);
                if (!is_split_copy(input_cpy, pf->split_ids[k])) {
                    continue;
                }
                GGML_ASSERT(input_cpy->data == NULL && input_cpy->buffer == NULL && input_cpy->view_src == NULL);
                auto it = lent.find(input_cpy);
                if (dry) {
                    input_cpy->data = &ggml_backend_sched_dry_placeholder;
                } else if (it != lent.end()) {
                    // initialized like the arena copies, as a compute tensor (e.g. CUDA does not clear the padding on another stream)
                    ggml_backend_buffer_t buf = it->second.first;
                    const enum ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(buf);
                    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    enum ggml_status st = ggml_backend_tensor_alloc(buf, input_cpy, (char *) ggml_backend_buffer_get_base(buf) + it->second.second);
                    ggml_backend_buffer_set_usage(buf, usage);
                    GGML_ASSERT(st == GGML_STATUS_SUCCESS);
                    if (pf->n_lent_copies >= pf->lent_copies_capacity) {
                        pf->lent_copies_capacity = std::max(64, 2*pf->lent_copies_capacity);
                        pf->lent_copies = (const struct ggml_tensor **) realloc((void *) pf->lent_copies, pf->lent_copies_capacity*sizeof(pf->lent_copies[0]));
                        GGML_ASSERT(pf->lent_copies != NULL);
                    }
                    pf->lent_copies[pf->n_lent_copies++] = input_cpy;
                } else {
                    void * base = ggml_backend_buffer_get_base(pf->arena);
                    enum ggml_status st = ggml_backend_tensor_alloc(pf->arena, input_cpy, (char *) base + offset);
                    GGML_ASSERT(st == GGML_STATUS_SUCCESS);
                }
                if (it == lent.end()) {
                    offset += GGML_PAD(ggml_backend_buft_get_alloc_size(sched->bufts[b], input_cpy), align);
                }
            }
        }
        if (!dry) {
            std::sort(pf->lent_copies, pf->lent_copies + pf->n_lent_copies);
            sched->pf_active = true;
        }
    }
}

static bool ggml_backend_sched_prefetch_is_bound(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split, const struct ggml_tensor * input) {
    if (!sched->pf_active || split->pf_ordinal < 0) {
        return false;
    }
    const struct ggml_backend_sched_prefetch * pf = &sched->pf[split->backend_id];
    const struct ggml_tensor * input_cpy = tensor_copy(const_cast<ggml_tensor *>(input), split->backend_id, sched->cur_copy);
    if (input_cpy->buffer == NULL) {
        return false;
    }
    if (input_cpy->buffer == pf->arena) {
        return true;
    }
    return ggml_backend_sched_is_lent(pf, input_cpy) &&
        std::binary_search(pf->lent_copies, pf->lent_copies + pf->n_lent_copies, (const struct ggml_tensor *) input_cpy);
}

// without a lender (the MoE cache) a staging arena is memory that a measurement (e.g. --fit) did not count (see llama_context::sched_reserve):
// it only takes what is left, a device with less than GGML_SCHED_PREFETCH_HEADROOM_MIB (512) free after the reserve drops the prefetch,
// the backends allocate more while computing (e.g. the CUDA pool)
static bool ggml_backend_sched_prefetch_headroom_ok(ggml_backend_sched_t sched) {
    const size_t headroom = sched->pf_headroom;
    if (sched->callback_lend != NULL || headroom == 0) {
        return true;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        ggml_backend_dev_t dev = sched->backends[b]->device;
        if (pf->arena == NULL || dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        if (total > 0 && free < headroom) {
            GGML_LOG_WARN("%s: %s: %.2f MiB free after %.2f MiB of staging memory, less than %.0f MiB of headroom: host weight prefetch disabled "
                "(GGML_SCHED_PREFETCH_HEADROOM_MIB)\n", __func__, ggml_backend_name(sched->backends[b]), free/1024.0/1024.0,
                pf->arena_size/1024.0/1024.0, headroom/1024.0/1024.0);
            return false;
        }
    }
    return true;
}

// a starting depth above the default (ggml_backend_sched_set_prefetch_depth_init, no adaptive step taken yet) whose arena, with the graph
// buffers reserved next to it, leaves less than the prefetch headroom free on a device. ggml_backend_sched_prefetch_headroom_ok skips the
// check while a lender is set, also when it lends nothing, and the memory fit measures the arena at the default depth: L4 (22.5 GB),
// Qwen3.8-Flash-Next UD-IQ1_S + MTP drafter, --fit on -c 16384: depth 3 took 1350 MiB (lending off), 16 MiB stayed free and the first
// request failed to allocate the cuBLAS workspace (out of memory); depth 2 (656 MiB) left 710 MiB
static bool ggml_backend_sched_prefetch_depth_init_short(ggml_backend_sched_t sched) {
    if (sched->pf_depth <= sched->pf_depth0 || sched->pf_adapt_from > 0 || sched->pf_headroom == 0) {
        return false;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        ggml_backend_dev_t dev = sched->backends[b]->device;
        if (pf->arena == NULL || pf->depth <= sched->pf_depth0 || dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        if (total > 0 && free < sched->pf_headroom) {
            GGML_LOG_INFO("%s: %s: %.2f MiB free after %.2f MiB of staging memory at depth %d, less than %.0f MiB of headroom: starting at depth %d\n",
                __func__, ggml_backend_name(sched->backends[b]), free/1024.0/1024.0, pf->arena_size/1024.0/1024.0, pf->depth,
                sched->pf_headroom/1024.0/1024.0, sched->pf_depth0);
            return true;
        }
    }
    return false;
}

// the graph allocator failed with the staging arenas allocated: unbind the copies, free the arenas and disable prefetch, so the allocation can retry with that memory
// returns false if nothing was freed
static bool ggml_backend_sched_prefetch_disable(ggml_backend_sched_t sched) {
    bool freed = false;
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        for (int j = 0; j < split->n_inputs; j++) {
            if (ggml_backend_sched_prefetch_is_bound(sched, split, split->inputs[j])) {
                struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], split->backend_id, sched->cur_copy);
                input_cpy->data   = NULL;
                input_cpy->buffer = NULL;
            }
        }
        split->pf_ordinal = -1;
        split->pf_async   = false;
        split->pf_lent    = false;
    }
    sched->pf_active = false;

    ggml_backend_sched_prefetch_sync(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        pf->n_bound = 0;
        pf->n_async = 0;
        pf->need    = 0;
        pf->staging = 0;
        ggml_backend_sched_lend_end(sched, b);
        if (pf->arena != NULL || pf->copy_backend != NULL) {
            GGML_LOG_WARN("%s: %s: graph allocation failed, host weight prefetch disabled\n", __func__, ggml_backend_name(sched->backends[b]));
            ggml_backend_sched_prefetch_release(sched, b);
            pf->failed = true;
            freed = true;
        }
    }
    return freed;
}

// issue the copies of the next prefetched splits of every backend, up to depth - 1 splits ahead of the `consumed` splits queued for compute
// a copy only waits for computes queued before the last one: with depth = 2, the copy of split k + 1 overlaps the compute of split k
// the copies of a prefetched split of backend b on its copy queue, after waiting for `wait`
static void ggml_backend_sched_prefetch_copy(ggml_backend_sched_t sched, int b, struct ggml_backend_sched_split * split, int slot, ggml_backend_event_t wait) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
    if (wait != NULL) {
        ggml_backend_event_wait(pf->copy_backend, wait);
    }
    for (int j = 0; j < split->n_inputs; j++) {
        struct ggml_tensor * input = split->inputs[j];
        if (ggml_backend_sched_prefetch_is_bound(sched, split, input)) {
            struct ggml_tensor * input_cpy = tensor_copy(input, b, sched->cur_copy);
            // the copy callback may copy parts of the weight from device memory (e.g. a cache of MoE experts)
            if (sched->callback_copy == NULL ||
                !sched->callback_copy(pf->copy_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
                // the source callback may supply the bytes from elsewhere (e.g. read ahead from the model file into pinned memory)
                if (sched->callback_source == NULL ||
                    !sched->callback_source(pf->copy_backend, input, input_cpy, 0, ggml_nbytes(input), sched->callback_source_user_data)) {
                    ggml_backend_tensor_set_async(pf->copy_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                }
            }
        }
    }
    ggml_backend_event_record(pf->ready[slot], pf->copy_backend);
    pf->pending = true;
}

// splits issued ahead of the computed ones: the depth of the binding (its staging slots), or a smaller depth the adaptive depth went
// back to since the binding (a step that did not pay ends at once, not at the next binding: llama reuses the graph of a prompt's ubatches)
static int ggml_backend_sched_prefetch_ahead(ggml_backend_sched_t sched, const struct ggml_backend_sched_prefetch * pf) {
    return std::max(std::min(pf->depth, sched->pf_depth) - 1, 1);
}

// early = true at the start of a split (k15-pp): also issue the copies the end of the previous prefetched split would issue next, i.e. up
// to depth - 1 ahead of the splits consumed. Their slots were last read by splits that are already queued (the slot of copy c + depth - 1
// was last read by split c - 1, whose retire event is recorded), so the copies are the same, only queued before this split's work. A split
// that stops the host part-way (an eval callback that synchronizes, e.g. the layer-major seed of the MoE cache in the split that holds a
// layer's router) then no longer holds back the next layer's copies: G4, Qwen3.6-35B-A3B Q4 + MTP at 12 GB, 2k prompt: 42 copy-engine gaps
// of ~2 ms in a 357 ms layer-major pass. Opt-in (GGML_SCHED_PREFETCH_EARLY=1): on k14 it made G4 q4x 12 GB 2k -8%, 8k -12% (the
// [512] pass that follows the layer-major group lost most), plans/k15_pp.md
static void ggml_backend_sched_prefetch_issue_ex(ggml_backend_sched_t sched, bool early) {
    static const bool early_env = getenv("GGML_SCHED_PREFETCH_EARLY") != nullptr && atoi(getenv("GGML_SCHED_PREFETCH_EARLY")) != 0;
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        int ahead = ggml_backend_sched_prefetch_ahead(sched, pf);
        if (early && early_env && pf->depth >= 2) {
            ahead = std::min(ahead + 1, pf->depth);
        }
        while (pf->issued < pf->n_bound && pf->issued < pf->consumed + ahead) {
            struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[pf->issued]];
            if (split->pf_async) {
                const int slot = pf->issued % pf->depth;
                // the slot is free once the split that read it last has been computed
                ggml_backend_sched_prefetch_copy(sched, b, split, slot, pf->retire[slot]);
            }
            pf->issued++;
        }
    }
}

static void ggml_backend_sched_prefetch_issue(ggml_backend_sched_t sched) {
    ggml_backend_sched_prefetch_issue_ex(sched, false);
}

// cross-graph arm: a split whose copies are plain host weights read by regular ops (not the experts of MUL_MAT_ID, which a copy
// callback may copy from device memory the next computation changes, e.g. a cache of MoE experts) and not placed in lent memory
static bool ggml_backend_sched_prefetch_armable(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split) {
    if (!split->pf_async || split->pf_lent) {
        return false;
    }
    for (int n = 0; n < split->graph.n_nodes; n++) {
        const struct ggml_tensor * node = split->graph.nodes[n];
        if (node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_ADD_ID) {
            continue;
        }
        for (int j = 0; j < split->n_inputs; j++) {
            const struct ggml_tensor * cpy = tensor_copy(split->inputs[j], split->backend_id, sched->cur_copy);
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                for (const struct ggml_tensor * t = node->src[k]; t != NULL; t = t->view_src) {
                    if (t == cpy) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// after the last split of a graph: issue the copies of the first prefetched splits again for the next computation of the same binding
// (llama reuses the graph of identical ubatches), they wait for the last prefetched split of this one, so every split of it has been computed
static void ggml_backend_sched_prefetch_arm(ggml_backend_sched_t sched) {
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        pf->armed = 0;
        if (pf->n_bound == 0 || pf->copy_backend == NULL || pf->consumed != pf->n_bound) {
            continue;
        }
        const int ahead = std::min(ggml_backend_sched_prefetch_ahead(sched, pf), pf->n_bound);
        ggml_backend_event_t wait = pf->retire[(pf->n_bound - 1) % pf->depth];
        for (int k = 0; k < ahead; k++) {
            struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[k]];
            if (!ggml_backend_sched_prefetch_armable(sched, split)) {
                break;
            }
            ggml_backend_sched_prefetch_copy(sched, b, split, k % pf->depth, k == 0 ? wait : NULL);
            pf->armed++;
            sched->pf_n_armed++;
        }
        pf->armed_gen = sched->pf_gen;
    }
}

// GGML_SCHED_PREFETCH_STATS=1 (diagnostic, default off): time the waits of every prefetched graph, also after the adaptive depth settled,
// and log them per graph (depth, waits, staging and lent bytes)
static bool ggml_backend_sched_prefetch_stats() {
    static const bool on = getenv("GGML_SCHED_PREFETCH_STATS") != NULL && atoi(getenv("GGML_SCHED_PREFETCH_STATS")) != 0;
    return on;
}

// adaptive depth: read the timers of the last computed graph if they completed (never waits), and decide on a step every few graphs
static void ggml_backend_sched_prefetch_adapt(ggml_backend_sched_t sched) {
    if (sched->pf_timer_elapsed == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        if (!pf->timed) {
            continue;
        }
        pf->timed = false;
        if ((!sched->pf_adapt && !ggml_backend_sched_prefetch_stats()) || pf->n_timed == 0) {
            continue;
        }
        float stall = 0.0f;
        float busy  = 0.0f;
        bool  ok    = true;
        for (int i = 0; i < pf->n_timed && ok; i++) {
            float ms = 0.0f;
            ok = sched->pf_timer_elapsed(pf->timers[2*i], pf->timers[2*i + 1], &ms);
            stall += ms;
        }
        // first wait to the end of the last timed split
        ok = ok && sched->pf_timer_elapsed(pf->timers[0], pf->timers[2*pf->n_timed], &busy);
        if (!ok || busy <= 0.0f) {
            continue;
        }
        if (ggml_backend_sched_prefetch_stats()) {
            // the binding may have changed since the graph ran (the next graph is bound before this is read): only the timers are its own
            GGML_LOG_INFO("%s: prefetch stats: depth %d, %d timed splits waited %.2f of %.2f ms (%.1f%%) for their copies\n",
                __func__, pf->depth, pf->n_timed, stall, busy, 100.0*stall/busy);
        }
        if (!sched->pf_adapt) {
            continue;
        }
        sched->pf_acc_stall += stall;
        sched->pf_acc_busy  += busy;
        sched->pf_acc_n++;
    }
    // 3 graphs before the first step, 2 to judge a step: its graphs run at the deeper depth until the decision (a trial that does not
    // pay costs its graphs)
    if (!sched->pf_adapt || sched->pf_acc_n < (sched->pf_adapt_from > 0 ? 2 : 3)) {
        return;
    }
    const double ratio = sched->pf_acc_stall / sched->pf_acc_busy;
    const double stall = sched->pf_acc_stall / sched->pf_acc_n; // the wait time itself: a deeper prefetch also shortens the graph
    const int    from  = sched->pf_depth;
    // staging in lent memory (e.g. slots of a MoE cache): a larger depth borrows more of it, which costs the lender more than the waits it saves
    bool lent = false;
    for (int b = 0; b < sched->n_backends; b++) {
        lent = lent || sched->pf[b].lent > 0;
    }
    const double busy  = sched->pf_acc_busy / sched->pf_acc_n;  // the timed span per graph (first wait to the end of the last prefetched split)
    // a step is judged by the graph span, not only by the waits: fewer waits can come with slower work around them (a deeper prefetch
    // overlaps more copies with the compute, and its larger arena takes device memory), so the graph does not get faster;
    // the measured graphs of a step come from later computations (often later in a prompt, with more context), which favours the old depth
    const bool   stepped  = sched->pf_adapt_from > 0;
    const bool   no_waits = stepped && stall > 0.75*sched->pf_adapt_prev;
    const bool   no_gain  = stepped && busy > (1.0 - sched->pf_adapt_gain)*sched->pf_adapt_prev_busy;
    if (no_waits || no_gain) {
        // the last step did not cut the waits enough or did not make the graphs faster: back to the depth before it, and give back the
        // memory of the larger arena
        sched->pf_depth = sched->pf_adapt_from;
        sched->pf_adapt = false;
        for (int b = 0; b < sched->n_backends; b++) {
            sched->pf[b].shrink = true;
        }
    } else if (ratio > sched->pf_adapt_min_wait && sched->pf_depth < sched->pf_adapt_max && !lent) {
        // a step can at most save the waits: below a few percent of the graph it does not pay for its trial (and the deeper prefetch
        // tends to slow the work around the copies by more)
        sched->pf_adapt_prev      = stall;
        sched->pf_adapt_prev_busy = busy;
        sched->pf_adapt_from      = sched->pf_depth;
        sched->pf_depth++;
    } else {
        sched->pf_adapt = false;
    }
    char judged[160] = "";
    if (stepped) {
        snprintf(judged, sizeof(judged), ", graph %.1f -> %.1f ms (%+.1f%%, a step needs %.1f%% less)%s", sched->pf_adapt_prev_busy, busy,
            100.0*(busy/sched->pf_adapt_prev_busy - 1.0), 100.0*sched->pf_adapt_gain, no_waits ? ", waits not cut" : no_gain ? ", no gain" : "");
    }
    GGML_LOG_INFO("%s: host weight prefetch: splits waited %.1f%% of %.1f ms (%.1f ms per graph) for their copies at depth %d (%d graphs%s): %s depth %d%s\n", __func__,
        100.0*ratio, sched->pf_acc_busy, stall, from, sched->pf_acc_n, judged, sched->pf_adapt ? "trying" : "settled at", sched->pf_depth,
        lent && !sched->pf_adapt ? " (staging in lent memory)" : "");
    sched->pf_acc_stall = 0.0;
    sched->pf_acc_busy  = 0.0;
    sched->pf_acc_n     = 0;
}

// timers for the prefetched splits of backend b in this graph: 2 per async split (around its wait) and one after the last
static bool ggml_backend_sched_prefetch_timers(ggml_backend_sched_t sched, int b) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
    const int need = 2*pf->n_bound + 1;
    if (pf->n_timers_cap >= need) {
        return true;
    }
    pf->timers = (ggml_backend_event_t *) realloc(pf->timers, need*sizeof(ggml_backend_event_t));
    GGML_ASSERT(pf->timers != NULL);
    for (int i = pf->n_timers_cap; i < need; i++) {
        pf->timers[i] = sched->pf_timer_new(sched->backends[b]->device);
        if (pf->timers[i] == NULL) {
            pf->n_timers_cap = i;
            return false;
        }
    }
    pf->n_timers_cap = need;
    return true;
}

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
        return;
    }

    // wait for the split backend to finish using the input before overwriting it
    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } else {
        ggml_backend_synchronize(split_backend);
    }

    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
        return;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        ggml_backend_synchronize(input_backend);
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
    }
}

// k15: a CUDA-family device backend (one in-order stream: an async copy queued on it runs after the work queued before it)
static bool ggml_backend_sched_hsm_backend_ok(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend != NULL ? ggml_backend_get_device(backend) : NULL;
    if (dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";
    return strcmp(reg_name, "CUDA") == 0 || strcmp(reg_name, "ROCm") == 0 || strcmp(reg_name, "MUSA") == 0;
}

// k15: wait for an upload left pending by ggml_backend_sched_compute_splits_impl (its event if it has one, else the backend)
static void ggml_backend_sched_hsm_wait(ggml_backend_sched_t sched) {
    if (sched->hsm_pending < 0) {
        return;
    }
    const int b = sched->hsm_pending;
    if (sched->hsm_event[b] != NULL) {
        ggml_backend_event_synchronize(sched->hsm_event[b]);
    } else {
        ggml_backend_synchronize(sched->backends[b]);
    }
    sched->hsm_pending = -1;
}

// without pipeline parallelism, a host split input is queued as an async upload instead of a sync copy (2 host syncs per input)
// the caller synchronizes the split backend once after all of them, so the host data can be overwritten when compute_splits moves on
// with the CPU/GPU overlap, an upload from host compute memory (written only by later host splits) neither waits for the backend
// nor is waited for here: `relaxed` is set and the caller records an event that host work waits for before it overwrites the memory
// k15-fn (GGML_SCHED_UPLOAD_BATCH=0 off): the async graph-input uploads of a split are queued and issued together by
// ggml_backend_sched_upload_flush (one cudaMemcpyBatchAsync): issued one by one, the device idled between the small copies
// (Flash-Next verify: 21 uploads, 0.63 ms of idle per step on a B200)
static ggml_backend_set_tensors_batch_async_t ggml_backend_sched_batch_fn(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    static thread_local ggml_backend_reg_t last_reg = NULL;
    static thread_local ggml_backend_set_tensors_batch_async_t last_fn = NULL;
    if (reg != last_reg) {
        last_reg = reg;
        last_fn  = reg ? (ggml_backend_set_tensors_batch_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_tensors_batch_async") : NULL;
    }
    return last_fn;
}

static bool ggml_backend_sched_upload_queue(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_tensor * dst, const void * data, size_t size) {
    static const bool enabled = getenv("GGML_SCHED_UPLOAD_BATCH") == NULL || atoi(getenv("GGML_SCHED_UPLOAD_BATCH")) != 0;
    if (!enabled || size == 0 || ggml_backend_sched_batch_fn(backend) == NULL) {
        return false;
    }
    if (sched->upload_batch == NULL) {
        sched->upload_batch = new ggml_backend_sched_upload_batch();
    }
    ggml_backend_sched_upload_batch * ub = sched->upload_batch;
    if (ub->backend != NULL && ub->backend != backend) {
        return false; // one backend per batch (a split has one)
    }
    ub->backend = backend;
    ub->tensors.push_back(dst);
    ub->data.push_back(data);
    ub->sizes.push_back(size);
    return true;
}

// issues the queued uploads (in queue order) before anything that must follow them on the stream
static void ggml_backend_sched_upload_flush(ggml_backend_sched_t sched) {
    ggml_backend_sched_upload_batch * ub = sched->upload_batch;
    if (ub == NULL || ub->tensors.empty()) {
        return;
    }
    const int n = (int) ub->tensors.size();
    ggml_backend_set_tensors_batch_async_t fn = ggml_backend_sched_batch_fn(ub->backend);
    if (n == 1 || fn == NULL || !fn(ub->backend, ub->tensors.data(), ub->data.data(), ub->sizes.data(), n)) {
        for (int i = 0; i < n; ++i) {
            ggml_backend_tensor_set_async(ub->backend, ub->tensors[i], ub->data[i], 0, ub->sizes[i]);
        }
    }
    ub->backend = NULL;
    ub->tensors.clear();
    ub->data.clear();
    ub->sizes.clear();
}

static bool ggml_backend_sched_upload_input_async(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input, bool & synced, bool & relaxed) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];

    // GGML_SCHED_ASYNC_UPLOAD: unset = the CUDA family only (as the overlap and the prefetch: other backends have not been validated;
    // Metal wraps every async upload in a new MTLBuffer, measured 2x slower decode on macOS 15 than the sync copy), 1 = every backend
    // with set_tensor_async, 0 = never (every host input takes the synchronous copy below, the upstream path)
    static const int async_upload = getenv("GGML_SCHED_ASYNC_UPLOAD") != NULL ? atoi(getenv("GGML_SCHED_ASYNC_UPLOAD")) : -1;
    if (async_upload == 0) {
        return false;
    }
    if (async_upload < 0) {
        ggml_backend_dev_t dev = ggml_backend_get_device(split_backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";
        if (strcmp(reg_name, "CUDA") != 0 && strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0) {
            return false;
        }
    }
    if (sched->events[split_backend_id][sched->cur_copy] != NULL || split_backend->iface.set_tensor_async == NULL) {
        return false;
    }
    if (input->buffer == NULL || !ggml_backend_buffer_is_host(input->buffer)) {
        return false;
    }
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
    if (input_cpy->buffer == NULL || ggml_backend_buffer_is_host(input_cpy->buffer)) {
        return false;
    }
    // a meta (tensor parallel) buffer splits the copy by its split state, its async set only handles some of them: the sync copy below
    if (ggml_backend_buffer_is_meta(input_cpy->buffer)) {
        return false;
    }

    // k13-pp: graph inputs (leaves set by the caller before the compute, e.g. positions, masks, token ids) take the same relaxed
    // upload: it is queued on the compute stream behind the previous graph (CUDA family: relaxed_ok), its event is waited
    // before this compute returns (ov_wait_uploads at the end of compute_splits), so the caller cannot overwrite the host data
    // before the copy has read it. The host then queues this graph while the previous one still runs instead of waiting for
    // it before the first upload. GGML_SCHED_INPUT_RELAXED=0: graph inputs synchronize the device first (as before).
    static const bool input_relaxed = getenv("GGML_SCHED_INPUT_RELAXED") == NULL || atoi(getenv("GGML_SCHED_INPUT_RELAXED")) != 0;
    const bool ov_ok = sched->ov_active && (sched->ov_parts & 4) && ggml_backend_sched_ov_backend_ok(sched, split_backend_id) &&
        sched->ov[split_backend_id].relaxed_ok &&
        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    bool ov = ov_ok && ggml_backend_sched_ov_root(input)->op != GGML_OP_NONE;
    for (const struct ggml_tensor * t = input; t != NULL && ov; t = t->view_src) {
        ov = (t->flags & GGML_TENSOR_FLAG_INPUT) == 0;
    }
    // a graph input, or a view of one (e.g. the recurrent-state copy indices rs_s_copy): the copy reads the caller's host data
    const struct ggml_tensor * in_root = ggml_backend_sched_ov_root(input);
    if (!ov && ov_ok && input_relaxed && sched->input_relax_ok && in_root->op == GGML_OP_NONE && (in_root->flags & GGML_TENSOR_FLAG_INPUT) &&
            in_root->buffer == input->buffer) {
        ov = true;
        sched->n_input_relaxed++;
    }
    // GGML_SCHED_INPUT_RELAXED_LOG=N: the first N uploads that still synchronize the device are logged (debug)
    static const int relaxed_log = getenv("GGML_SCHED_INPUT_RELAXED_LOG") ? atoi(getenv("GGML_SCHED_INPUT_RELAXED_LOG")) : 0;
    static int n_logged = 0;
    if (!ov && relaxed_log && n_logged < relaxed_log) {
        n_logged++;
        GGML_LOG_INFO("%s: synchronized upload of %s (op %s, view %d, flags %d, usage %d, ov_active %d, ov_ok %d)\n", __func__, input->name,
            ggml_op_name(input->op), input->view_src != NULL, input->flags, (int) ggml_backend_buffer_get_usage(input->buffer), sched->ov_active, ov_ok);
    }

    if (!synced && !ov && !(sched->hsm_active && ggml_backend_sched_hsm_backend_ok(split_backend))) {
        // wait for the split backend to finish using its input copies before overwriting them
        // (k15 hsm: not needed on a CUDA-family backend, the upload is queued on its stream behind the work that reads them)
        ggml_backend_synchronize(split_backend);
        synced = true;
    }
    if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) {
        // produced by an earlier split
        ggml_backend_synchronize(ggml_backend_sched_get_tensor_backend(sched, input));
    }
    if (!ggml_backend_sched_upload_queue(sched, split_backend, input_cpy, input->data, ggml_nbytes(input))) {
        ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
    }
    relaxed = ov;
    return true;
}

// overlap: the number of leading nodes of split that can run before the split preceding it (on the host backend host_id) is computed
// they must read none of the input copies of the split and no tensor of host_id (e.g. a host buffer the device can read without a copy)
static int ggml_backend_sched_overlap_prefix(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split, int host_id) {
    if (split->pf_ordinal >= 0) {
        return 0;
    }
    auto is_late = [&](struct ggml_tensor * t) {
        for (int k = 0; k < split->n_inputs; k++) {
            const struct ggml_tensor * cpy = tensor_copy(split->inputs[k], split->backend_id, sched->cur_copy);
            if (t == cpy || t->view_src == cpy) {
                return true;
            }
        }
        struct ggml_tensor * base = t->view_src != NULL ? t->view_src : t;
        if (ggml_hash_contains(&sched->hash_set, base)) {
            const int id = sched->hv_tensor_backend_ids[ggml_hash_find(&sched->hash_set, base)];
            if (id != split->backend_id && (base->buffer == NULL || ggml_backend_buffer_get_usage(base->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                return true; // produced on another backend
            }
            if (id == host_id) {
                return true;
            }
        }
        return false;
    };
    for (int j = 0; j < split->graph.n_nodes; j++) {
        const struct ggml_tensor * node = split->graph.nodes[j];
        if (ggml_is_view_op(node->op)) {
            continue; // no work, its users are checked
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s] != NULL && is_late(node->src[s])) {
                return j;
            }
        }
    }
    return split->graph.n_nodes;
}

static enum ggml_status ggml_backend_sched_compute_splits_impl(ggml_backend_sched_t sched);

// while a split that reads copies placed in lent memory is queued, the lent buffers behave like compute buffers (e.g. CUDA clears the padding
// of quantized copies before reading them, as for the arena); the other splits see the user's own usage, they may read the user's tensors
// in those buffers as weights (e.g. a cache read through host lookups); the user gets them back with their own usage
static void ggml_backend_sched_lent_usage(ggml_backend_sched_t sched, int backend_id, bool compute) {
    struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_id];
    for (int i = 0; i < pf->n_lent_bufs; i++) {
        ggml_backend_buffer_set_usage(pf->lent_bufs[i], compute || pf->lent_all ? GGML_BACKEND_BUFFER_USAGE_COMPUTE : pf->lent_usage[i]);
    }
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        for (int i = 0; i < pf->n_lent_bufs; i++) {
            pf->lent_usage[i] = ggml_backend_buffer_get_usage(pf->lent_bufs[i]);
        }
        ggml_backend_sched_lent_usage(sched, b, false);
    }
    const enum ggml_status st = ggml_backend_sched_compute_splits_impl(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        for (int i = pf->n_lent_bufs - 1; i >= 0; i--) {
            ggml_backend_buffer_set_usage(pf->lent_bufs[i], pf->lent_usage[i]);
        }
    }
    return st;
}

// the host weights read by the prefetched copies of the graph in the order ggml_backend_sched_prefetch_issue issues them
// (same loop as the compute: the copies are issued before each split and after each prefetched split is queued)
static void ggml_backend_sched_prefetch_plan(ggml_backend_sched_t sched) {
    std::vector<const struct ggml_tensor *> & srcs = *sched->plan_srcs;
    srcs.clear();
    if (sched->pf_active) {
        const int ahead = std::max(sched->pf_depth - 1, 1);
        int issued[GGML_SCHED_MAX_BACKENDS]   = { 0 };
        int consumed[GGML_SCHED_MAX_BACKENDS] = { 0 };
        auto issue = [&]() {
            for (int b = 0; b < sched->n_backends; b++) {
                const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
                while (issued[b] < pf->n_bound && issued[b] < consumed[b] + ahead) {
                    const struct ggml_backend_sched_split * split = &sched->splits[pf->split_ids[issued[b]]];
                    if (split->pf_async) {
                        for (int j = 0; j < split->n_inputs; j++) {
                            if (ggml_backend_sched_prefetch_is_bound(sched, split, split->inputs[j])) {
                                srcs.push_back(split->inputs[j]);
                            }
                        }
                    }
                    issued[b]++;
                }
            }
        };
        for (int i = 0; i < sched->n_splits; i++) {
            issue();
            if (sched->splits[i].pf_ordinal >= 0) {
                consumed[sched->splits[i].backend_id]++;
                issue();
            }
        }
    }
    sched->callback_plan(srcs.data(), (int) srcs.size(), sched->callback_plan_user_data);
}

// k16-disk: disk prefetch lookahead of host experts (ported from ik_llama.cpp PR #2101, see ggml-moe-prefetch.cpp). The host expert
// tensors (host buffer, WEIGHTS usage, ne[2] > 1) that the engine has registered (expert ranges of file mappings, by llama) of the
// next GGML_MOE_PREFETCH_AHEAD (default 3, ggml_moe_prefetch_set_ahead) MoE splits of a batch of >= 32 tokens are streamed into the
// page cache while the current split runs, and the pages the streamer read for a split are MADV_COLD after it, so prompt streaming
// does not evict the decode working set. Page cache only: the copies and kernels read the same mapped bytes.
struct ggml_moe_disk_split {
    int split;
    std::vector<const ggml_tensor *> host_weights;
};

static std::atomic<int> ggml_moe_prefetch_ahead_n{-1};

extern "C" void ggml_moe_prefetch_set_ahead(int n) {
    ggml_moe_prefetch_ahead_n.store(n, std::memory_order_relaxed);
}

static int ggml_moe_prefetch_ahead() {
    const int n = ggml_moe_prefetch_ahead_n.load(std::memory_order_relaxed);
    if (n >= 0) {
        return n;
    }
    static const int def = [] {
        const char * env = getenv("GGML_MOE_PREFETCH_AHEAD");
        return env ? std::max(0, atoi(env)) : 3;
    }();
    return def;
}

static void ggml_moe_disk_splits(ggml_backend_sched_t sched, std::vector<ggml_moe_disk_split> & out) {
    out.clear();
    for (int i = 0; i < sched->n_splits; i++) {
        const struct ggml_backend_sched_split & sp = sched->splits[i];
        ggml_moe_disk_split info;
        info.split = i;
        int64_t n_tokens = 0;
        bool multi_layer = false;
        const ggml_tensor * host_ids = nullptr;
        for (int n = 0; n < sp.graph.n_nodes; ++n) {
            const ggml_tensor * node = sp.graph.nodes[n];
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            const ggml_tensor * node_ids = node->src[2];
            n_tokens = std::max<int64_t>(n_tokens, node_ids ? node_ids->ne[1] : 0);
            const ggml_tensor * w = node->src[0];
            if (w && ggml_backend_sched_is_host_weight(w) && w->ne[2] > 1 && ggml_moe_prefetch_is_registered(w)) {
                info.host_weights.push_back(w);
                // up/gate and down of a layer share one ids tensor
                multi_layer |= host_ids != nullptr && host_ids != node_ids;
                host_ids = node_ids;
            }
        }
        // in the offloaded case the node's weight srcs were rewritten to device copies; the host originals arrive as split inputs
        for (int j = 0; j < sp.n_inputs; ++j) {
            const ggml_tensor * input = sp.inputs[j];
            if (input->ne[2] > 1 && ggml_backend_sched_is_host_weight(input) && ggml_moe_prefetch_is_registered(input)) {
                info.host_weights.push_back(input);
            }
        }
        // batch graphs touch most experts; multi-layer splits (e.g. a CPU-only graph) would fetch everything at once: the kernel hook
        // streams those
        if (!info.host_weights.empty() && n_tokens >= 32 && !multi_layer) {
            out.push_back(std::move(info));
        }
    }
}

static enum ggml_status ggml_backend_sched_compute_splits_impl(ggml_backend_sched_t sched) {
    struct ggml_backend_sched_split * splits = sched->splits;

    sched->pre_split = -1;
    sched->pre_nodes = 0;

    // k13-pp: graph inputs take the relaxed upload only when no host split follows a device split. A later host split (e.g. MoE
    // experts computed on the host) would wait for the relaxed upload at its start, i.e. for the previous graph, which moves
    // that wait into the host split's own time (the CPU-miss calibration times host splits). Graphs whose only host split is
    // the leading input split (token embeddings on the host, everything else on the device) keep the lever.
    {
        bool seen_device = false;
        sched->input_relax_ok = true;
        for (int i = 0; i < sched->n_splits; i++) {
            if (ggml_backend_buft_is_host(sched->bufts[sched->splits[i].backend_id])) {
                if (seen_device) {
                    sched->input_relax_ok = false;
                    break;
                }
            } else {
                seen_device = true;
            }
        }
    }

    int prev_backend_id = -1;

    sched->hsm_active = sched->hsm_mode != 0 && !sched->ov_active && sched->n_copies == 1 && sched->callback_eval == NULL;
    sched->hsm_pending = -1;

    ggml_backend_sched_prefetch_adapt(sched);
    sched->pf_runs++;
    bool timing[GGML_SCHED_MAX_BACKENDS] = { false };
    if (sched->callback_plan != NULL) {
        ggml_backend_sched_prefetch_plan(sched);
    }

    for (int b = 0; b < sched->n_backends; b++) {
        struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        // copies armed after the previous computation of this binding are already queued
        pf->issued   = sched->pf_active && sched->pf_cross && pf->armed > 0 && pf->armed_gen == sched->pf_gen ? pf->armed : 0;
        sched->pf_n_arm_used += pf->issued;
        pf->armed    = 0;
        pf->consumed = 0;
        pf->n_timed  = 0;
        // time the waits of this graph while adapting the depth (only a graph bound at the depth being evaluated)
        timing[b] = sched->pf_active && (sched->pf_adapt || ggml_backend_sched_prefetch_stats()) && pf->n_async > 0 && (pf->depth == sched->pf_depth || ggml_backend_sched_prefetch_stats()) && sched->pf_timer_new != NULL &&
            ggml_backend_sched_prefetch_timers(sched, b);
        // the copies of this graph run after the work queued on the backend before it (e.g. earlier graphs writing memory a copy callback reads)
        if (sched->pf_active && pf->n_async > 0 && pf->start != NULL) {
            ggml_backend_event_record(pf->start, sched->backends[b]);
            ggml_backend_event_wait(pf->copy_backend, pf->start);
            pf->pending = true;
        }
    }

    if (sched->ov_active) {
        for (int i = 0; i < sched->n_splits; i++) {
            splits[i].ov_issued = false;
        }
    }

    // k16-disk: disk prefetch lookahead (see ggml_moe_disk_splits)
    std::vector<ggml_moe_disk_split> moe_disk;
    size_t moe_next = 0; // first entry with split >= current split
    size_t moe_enq  = 0; // entries already enqueued
    size_t moe_cold = 0; // entries already given MADV_COLD
    const size_t moe_ahead = ggml_moe_prefetch_enabled() ? (size_t) ggml_moe_prefetch_ahead() : 0;
    if (moe_ahead > 0) {
        ggml_moe_prefetch_new_epoch();
        ggml_moe_disk_splits(sched, moe_disk);
    }

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        if (!moe_disk.empty()) {
            // the pages streamed for the splits that ran are one-shot prompt traffic
            for (; moe_cold < moe_disk.size() && moe_disk[moe_cold].split < split_id; ++moe_cold) {
                for (const ggml_tensor * w : moe_disk[moe_cold].host_weights) {
                    ggml_moe_prefetch_cold(w);
                }
            }
            while (moe_next < moe_disk.size() && moe_disk[moe_next].split < split_id) {
                moe_next++;
            }
            // keep the next moe_ahead MoE splits streaming in
            const size_t want_end = std::min(moe_next + moe_ahead, moe_disk.size());
            for (size_t k = std::max(moe_enq, moe_next); k < want_end; ++k) {
                for (const ggml_tensor * w : moe_disk[k].host_weights) {
                    ggml_moe_prefetch_tensor(w);
                }
                moe_enq = k + 1;
            }
        }

        if (sched->pf_active) {
            ggml_backend_sched_prefetch_issue_ex(sched, true);
        }

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // host split: its device inputs arrive with async copies and one event, the device keeps computing the work queued after them
        const bool ov_copies = sched->ov_active && split->ov_src >= 0;
        if (sched->ov_active && ggml_backend_sched_ov_is_host(sched, split_backend_id)) {
            if (ov_copies) {
                if (!split->ov_issued) {
                    ggml_backend_sched_ov_issue(sched, split_id);
                }
                ggml_backend_event_synchronize(sched->ov[split->ov_src].ready);
                sched->ov[split->ov_src].ready_pending = false;
            }
            // the host work of this split may overwrite host memory that pending uploads read
            ggml_backend_sched_ov_wait_uploads(sched, -1);
        }

        // k15 hsm: a host split reads its device inputs with async copies on their backends and one sync of each (which also waits for
        // an upload left pending), instead of a backend sync and a synchronous read for every input
        bool hsm_read[GGML_SCHED_MAX_SPLIT_INPUTS] = { false };
        if (sched->hsm_active && ggml_backend_sched_ov_is_host(sched, split_backend_id) && !sched->pf_active) {
            bool queued[GGML_SCHED_MAX_BACKENDS] = { false };
            for (int input_id = 0; input_id < split->n_inputs && input_id < GGML_SCHED_MAX_SPLIT_INPUTS; input_id++) {
                struct ggml_tensor * input = split->inputs[input_id];
                if (ggml_backend_sched_is_host_weight(input) || (input->flags & GGML_TENSOR_FLAG_INPUT)) {
                    continue;
                }
                ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
                const int ib = input_backend != NULL ? ggml_backend_sched_backend_id(sched, input_backend) : -1;
                struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
                if (ib < 0 || ib == split_backend_id || !ggml_backend_sched_hsm_backend_ok(input_backend) || input_backend->iface.get_tensor_async == NULL ||
                        input->buffer == NULL || ggml_backend_buffer_is_host(input->buffer) || input_cpy->buffer == NULL ||
                        !ggml_backend_buffer_is_host(input_cpy->buffer) || ggml_nbytes(input_cpy) != ggml_nbytes(input)) {
                    continue;
                }
                ggml_backend_tensor_get_async(input_backend, input, input_cpy->data, 0, ggml_nbytes(input));
                queued[ib] = true;
                hsm_read[input_id] = true;
                sched->hsm_n_reads++;
            }
            for (int b = 0; b < sched->n_backends; b++) {
                if (queued[b]) {
                    ggml_backend_synchronize(sched->backends[b]);
                    if (sched->hsm_pending == b) {
                        sched->hsm_pending = -1;
                    }
                }
            }
            ggml_backend_sched_hsm_wait(sched);
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        bool upload_synced  = false;
        bool upload_pending = false;
        bool upload_relaxed = false;
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                if (ov_copies && ggml_backend_sched_ov_early_input(sched, split, split->inputs[input_id])) {
                    continue; // copied with the event waited above
                }
                if (input_id < GGML_SCHED_MAX_SPLIT_INPUTS && hsm_read[input_id]) {
                    continue; // read above
                }
                bool relaxed = false;
                if (ggml_backend_sched_upload_input_async(sched, split, split->inputs[input_id], upload_synced, relaxed)) {
                    upload_relaxed |= relaxed;
                    upload_pending |= !relaxed;
                    continue;
                }
                ggml_backend_sched_upload_flush(sched); // a copy below may synchronize the stream: the queued uploads go first
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }
        ggml_backend_sched_upload_flush(sched);
        if (upload_pending && sched->hsm_active && ggml_backend_sched_hsm_backend_ok(split_backend)) {
            // k15 hsm: the host data the upload reads is next written by a later host split, which syncs this backend first, or by the
            // caller after this graph, which waits for the event below
            if (sched->hsm_pending >= 0 && sched->hsm_pending != split_backend_id) {
                ggml_backend_sched_hsm_wait(sched);
            }
            if (!sched->hsm_event_tried[split_backend_id]) {
                sched->hsm_event_tried[split_backend_id] = true;
                sched->hsm_event[split_backend_id] = ggml_backend_event_new(ggml_backend_get_device(split_backend));
            }
            if (sched->hsm_event[split_backend_id] != NULL) {
                ggml_backend_event_record(sched->hsm_event[split_backend_id], split_backend);
            }
            sched->hsm_pending = split_backend_id;
            sched->hsm_n_defer++;
        } else if (upload_pending) {
            // queued before the wait on the prefetched copies of this split, so it never waits for the copy queue
            ggml_backend_synchronize(split_backend);
        } else if (upload_relaxed) {
            ggml_backend_event_record(sched->ov[split_backend_id].upload, split_backend);
            sched->ov[split_backend_id].upload_pending = true;
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                if (split->pf_async && ggml_backend_sched_prefetch_is_bound(sched, split, split->inputs[input_id])) {
                    continue; // already issued on the copy queue
                }
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }

        // the inputs of a later host split are complete: copy them now, the work of this split overlaps the host split
        if (sched->ov_active && split->ov_issue_for >= 0) {
            ggml_backend_sched_ov_issue(sched, split->ov_issue_for);
        }

        // overlap: the inputs of this CPU split are ready, start the next split up to its first node that needs them or this split
        // (hybrid graphs have MoE cache lookups, so the CPU/GPU split overlap above is never active with it)
        if (sched->overlap && !sched->ov_active && !sched->callback_eval && sched->n_copies == 1 && !sched->pf_active && split_id + 1 < sched->n_splits &&
                ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            struct ggml_backend_sched_split * next = &splits[split_id + 1];
            ggml_backend_t next_backend = sched->backends[next->backend_id];
            if (next->backend_id != split_backend_id && ggml_backend_dev_type(ggml_backend_get_device(next_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                if (next->ov_prefix < 0) {
                    next->ov_prefix = ggml_backend_sched_overlap_prefix(sched, next, split_backend_id);
                }
                const int n_pre = next->ov_prefix;
                if (n_pre > 0) {
                    struct ggml_cgraph gv = ggml_graph_view(&next->graph, 0, n_pre);
                    enum ggml_status ec = ggml_backend_graph_compute_async(next_backend, &gv);
                    if (ec != GGML_STATUS_SUCCESS) {
                        return ec;
                    }
                    sched->pre_split = split_id + 1;
                    sched->pre_nodes = n_pre;
                }
            }
        }

        const bool pf_split = sched->pf_active && split->pf_ordinal >= 0;
        const int  pf_slot  = pf_split ? split->pf_ordinal % sched->pf[split_backend_id].depth : -1;
        if (sched->pf[split_backend_id].n_lent_bufs > 0) {
            ggml_backend_sched_lent_usage(sched, split_backend_id, split->pf_lent);
        }
        if (pf_split) {
            GGML_ASSERT(sched->pf[split_backend_id].consumed == split->pf_ordinal);
            GGML_ASSERT(sched->pf[split_backend_id].issued    > split->pf_ordinal);
            if (split->pf_async) {
                struct ggml_backend_sched_prefetch * pf = &sched->pf[split_backend_id];
                const bool timed = timing[split_backend_id];
                if (timed) {
                    ggml_backend_event_record(pf->timers[2*pf->n_timed], split_backend);
                }
                ggml_backend_event_wait(split_backend, pf->ready[pf_slot]);
                if (timed) {
                    ggml_backend_event_record(pf->timers[2*pf->n_timed + 1], split_backend);
                    pf->n_timed++;
                }
            }
        }

        if (sched->pre_split == split_id) {
            // the leading nodes were launched with the previous split
            if (sched->pre_nodes < split->graph.n_nodes) {
                struct ggml_cgraph gv = ggml_graph_view(&split->graph, sched->pre_nodes, split->graph.n_nodes);
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
            }
            sched->pre_split = -1;
        } else if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                ggml_backend_sched_prefetch_drain(sched);
                ggml_backend_sched_ov_drain(sched);
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    ggml_backend_sched_prefetch_drain(sched);
                    ggml_backend_sched_ov_drain(sched);
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                // (a range that ends without a node the user asked for needs no synchronization)
                if (need && !sched->callback_eval_nosync) {
                    ggml_backend_synchronize(split_backend);
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        // the slot can be refilled once this split has been computed
        if (pf_split) {
            struct ggml_backend_sched_prefetch * pf = &sched->pf[split_backend_id];
            ggml_backend_event_record(pf->retire[pf_slot], split_backend);
            pf->consumed++;
            if (timing[split_backend_id] && pf->consumed == pf->n_bound && pf->n_timed > 0) {
                ggml_backend_event_record(pf->timers[2*pf->n_timed], split_backend);
                pf->timed = true;
            }
            ggml_backend_sched_prefetch_issue(sched);
        }

        prev_backend_id = split_backend_id;
    }

    // the caller may write graph inputs before the next sched call (llama reuses graphs), and gallocr can place a
    // freed input at the host memory a pending upload still reads: the uploads finish before returning
    // (all work is queued by now, so this waits at most for the device work queued before the last upload)
    if (sched->ov_active) {
        ggml_backend_sched_ov_wait_uploads(sched, -1);
    }
    ggml_backend_sched_hsm_wait(sched);

    if (sched->pf_active && (sched->pf_cross == 2 || (sched->pf_cross == 1 && sched->pf_runs < sched->pf_prev_runs))) {
        ggml_backend_sched_prefetch_arm(sched);
    }

    for (; moe_cold < moe_disk.size(); ++moe_cold) {
        for (const ggml_tensor * w : moe_disk[moe_cold].host_weights) {
            ggml_moe_prefetch_cold(w);
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));
    sched->plan_srcs = new std::vector<const struct ggml_tensor *>();

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hash_size_full  = sched->hash_set.size;
    sched->hash_size_small = 0;
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    // calloc: a large table (sized for graphs of several ubatches) is not touched beyond what the graphs use (a zeroing memset of it was
    // resident in every such process: q4x + MTP 12 GB, n_batch 16384, RssAnon +100 MiB at load for the trunk and MTP contexts)
    sched->hv_tensor_copies      = (ggml_tensor **) calloc(sched->hash_set.size * sched->n_backends * sched->n_copies, sizeof(struct ggml_tensor *));
    // the sparse reset of a large table keeps the entries that are not in use at -1 / NULL (see ggml_backend_sched_reset)
    memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    // allocated by the first ggml_backend_sched_split_graph, sized for its graph (at most graph_size nodes)
    GGML_UNUSED(ggml_sched_max_splits);
    sched->context_graph_size  = graph_size;
    sched->context_buffer_size = 0;
    sched->context_buffer      = NULL;

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    {
        const char * GGML_SCHED_PREFETCH           = getenv("GGML_SCHED_PREFETCH");
        const char * GGML_SCHED_PREFETCH_DEPTH     = getenv("GGML_SCHED_PREFETCH_DEPTH");
        sched->pf_mode      = GGML_SCHED_PREFETCH           ? atoi(GGML_SCHED_PREFETCH)           : 1;
        sched->pf_depth     = GGML_SCHED_PREFETCH_DEPTH     ? atoi(GGML_SCHED_PREFETCH_DEPTH)     : 2;
        sched->pf_depth     = std::min(std::max(sched->pf_depth, 1), GGML_SCHED_MAX_PREFETCH_DEPTH);
        sched->pf_depth0    = sched->pf_depth;
        const char * GGML_SCHED_PREFETCH_HEADROOM_MIB = getenv("GGML_SCHED_PREFETCH_HEADROOM_MIB");
        sched->pf_headroom  = (size_t) (GGML_SCHED_PREFETCH_HEADROOM_MIB ? std::max(0, atoi(GGML_SCHED_PREFETCH_HEADROOM_MIB)) : 512)*1024*1024;
        // GGML_SCHED_LEND_SYNC=0: bindings with only SYNC copies are not offered to the lender (their copies go to the graph allocator
        // when no arena exists, see ggml_backend_sched_prefetch_bind); 1 (default) only if the graph allocator cannot grow by them; 2 always
        const char * GGML_SCHED_LEND_SYNC = getenv("GGML_SCHED_LEND_SYNC");
        sched->lend_sync    = GGML_SCHED_LEND_SYNC ? atoi(GGML_SCHED_LEND_SYNC) : 1;
        const char * GGML_SCHED_PREFETCH_WINDOW = getenv("GGML_SCHED_PREFETCH_WINDOW");
        const char * GGML_SCHED_PREFETCH_CROSS  = getenv("GGML_SCHED_PREFETCH_CROSS");
        const char * GGML_SCHED_PREFETCH_ADAPT  = getenv("GGML_SCHED_PREFETCH_ADAPT");
        // off by default: the byte ring only differs from uniform slots for splits of different sizes, and the cross-graph arm
        // did not change prompt speed where it engaged (copy-bound medium ubatches)
        sched->pf_window    = GGML_SCHED_PREFETCH_WINDOW ? atoi(GGML_SCHED_PREFETCH_WINDOW) != 0 : false;
        sched->pf_cross     = GGML_SCHED_PREFETCH_CROSS  ? atoi(GGML_SCHED_PREFETCH_CROSS)       : 0;
        // an explicit depth is kept as is
        sched->pf_adapt     = (GGML_SCHED_PREFETCH_ADAPT ? atoi(GGML_SCHED_PREFETCH_ADAPT) != 0 : true) && GGML_SCHED_PREFETCH_DEPTH == NULL && sched->pf_mode != 0;
        sched->pf_adapt_max = std::min(4, GGML_SCHED_MAX_PREFETCH_DEPTH);
        const char * GGML_SCHED_PREFETCH_ADAPT_GAIN = getenv("GGML_SCHED_PREFETCH_ADAPT_GAIN");
        sched->pf_adapt_gain = GGML_SCHED_PREFETCH_ADAPT_GAIN ? atof(GGML_SCHED_PREFETCH_ADAPT_GAIN) : 0.01;
        const char * GGML_SCHED_PREFETCH_ADAPT_MIN_WAIT = getenv("GGML_SCHED_PREFETCH_ADAPT_MIN_WAIT");
        sched->pf_adapt_min_wait = GGML_SCHED_PREFETCH_ADAPT_MIN_WAIT ? atof(GGML_SCHED_PREFETCH_ADAPT_MIN_WAIT) : 0.05;

        const char * GGML_SCHED_OVERLAP = getenv("GGML_SCHED_OVERLAP");
        sched->ov_mode = GGML_SCHED_OVERLAP ? atoi(GGML_SCHED_OVERLAP) : 1;
        const char * GGML_SCHED_HOST_SYNC_MERGE = getenv("GGML_SCHED_HOST_SYNC_MERGE");
        sched->hsm_mode    = GGML_SCHED_HOST_SYNC_MERGE ? atoi(GGML_SCHED_HOST_SYNC_MERGE) : 1;
        sched->hsm_pending = -1;
        const char * GGML_SCHED_OVERLAP_PARTS = getenv("GGML_SCHED_OVERLAP_PARTS");
        sched->ov_parts = GGML_SCHED_OVERLAP_PARTS ? atoi(GGML_SCHED_OVERLAP_PARTS) : 6;
    }

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched != NULL) {
        delete sched->upload_batch;
        sched->upload_batch = NULL;
    }
    if (sched == NULL) {
        return;
    }
    if (sched->n_input_relaxed > 0) {
        GGML_LOG_INFO("%s: %lld graph-input uploads queued behind the previous graph without a device sync (GGML_SCHED_INPUT_RELAXED=0: off)\n",
            __func__, (long long) sched->n_input_relaxed);
    }
    if (sched->hsm_n_reads > 0 || sched->hsm_n_defer > 0) {
        GGML_LOG_INFO("%s: host splits: %lld device inputs read with one sync per split, %lld uploads waited for later (GGML_SCHED_HOST_SYNC_MERGE=0: off)\n",
            __func__, (long long) sched->hsm_n_reads, (long long) sched->hsm_n_defer);
    }
    for (int b = 0; b < GGML_SCHED_MAX_BACKENDS; b++) {
        if (sched->hsm_event[b] != NULL) {
            ggml_backend_event_free(sched->hsm_event[b]);
        }
    }
    if (sched->pf_n_armed > 0) {
        GGML_LOG_INFO("%s: host weight prefetch: cross-graph arm: %lld splits armed, %lld used\n", __func__,
            (long long) sched->pf_n_armed, (long long) sched->pf_n_arm_used);
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    for (int b = 0; b < sched->n_backends; b++) {
        // the backends may already be freed: the events are released by their devices
        ggml_backend_event_free(sched->ov[b].ready);
        ggml_backend_event_free(sched->ov[b].upload);
        if (sched->pf[b].copy_backend != NULL) {
            ggml_backend_synchronize(sched->pf[b].copy_backend);
        }
        ggml_backend_sched_prefetch_release(sched, b);
        free(sched->pf[b].split_ids);
        free(sched->pf[b].lend_copies);
        free((void *) sched->pf[b].lent_copies);
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    delete sched->plan_srcs;
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        static const bool sparse_env = getenv("GGML_SCHED_SPARSE_RESET") == nullptr || atoi(getenv("GGML_SCHED_SPARSE_RESET")) != 0;
        if (sparse_env && sched->hash_set.size > (1u << 15)) {
            // a large table (sized for graphs of several ubatches, see ggml_backend_sched_set_reuse_weight_copies) clears only the entries
            // in use: every entry of the per-tensor arrays is written through hash_id, which marks it used
            const size_t n_words = ggml_bitset_size(sched->hash_set.size);
            const size_t per     = (size_t) sched->n_backends * sched->n_copies;
            for (size_t w = 0; w < n_words; ++w) {
                ggml_bitset_t bits = sched->hash_set.used[w];
                while (bits) {
                    const size_t i = (w << BITSET_SHR) + (size_t) ggml_bitset_ctz(bits);
                    bits &= bits - 1;
                    sched->hv_tensor_backend_ids[i] = -1;
                    memset(sched->hv_tensor_copies + i*per, 0, per * sizeof(struct ggml_tensor *));
                }
            }
            ggml_hash_set_reset(&sched->hash_set);
        } else {
            ggml_hash_set_reset(&sched->hash_set);
            memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
            memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        }
        sched->is_reset = true;
    }
    // the table is clean: the usual graphs use the small size again
    if (sched->hash_size_small > 0) {
        sched->hash_set.size = sched->hash_size_small;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_set_hash_size_small(ggml_backend_sched_t sched, size_t graph_size) {
    GGML_ASSERT(sched);
    ggml_backend_sched_reset(sched);
    const size_t s = graph_size > 0 ? ggml_hash_size(graph_size) : 0;
    sched->hash_size_small = s > 0 && s < sched->hash_size_full ? s : 0;
    sched->hash_set.size   = sched->hash_size_small > 0 ? sched->hash_size_small : sched->hash_size_full;
}

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
static void ggml_sched_pages_free(void * base, size_t from, size_t to) {
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    uintptr_t b = (uintptr_t) base + from;
    uintptr_t e = (uintptr_t) base + to;
    b = (b + page - 1) & ~(page - 1);
    e = e & ~(page - 1);
    if (e > b) {
        madvise((void *) b, e - b, MADV_DONTNEED);
    }
}
#endif

void ggml_backend_sched_trim(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
#if defined(__linux__)
    // on a clean table only: the entries above the small size are NULL (copies) or unread (keys), so zero pages are the same
    if (!sched->is_reset || sched->hash_size_small == 0 || sched->hash_size_small >= sched->hash_size_full) {
        return;
    }
    const size_t per = (size_t) sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *);
    ggml_sched_pages_free(sched->hv_tensor_copies, sched->hash_size_small*per, sched->hash_size_full*per);
    ggml_sched_pages_free(sched->hash_set.keys, sched->hash_size_small*sizeof(struct ggml_tensor *), sched->hash_size_full*sizeof(struct ggml_tensor *));
#endif
}

void ggml_backend_sched_use_full_hash(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // only on a clean table (nothing inserted since the reset): the entries keep their index
    GGML_ASSERT(sched->is_reset && "ggml_backend_sched_use_full_hash: call it right after ggml_backend_sched_reset");
    sched->hash_set.size = sched->hash_size_full;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_backend_sched_prefetch_bind(sched, /*dry =*/ true);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);

    // the staging slots of the prefetched weights
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->pf[b].n_bound > 0) {
            sizes[b] += sched->pf[b].staging;
        }
    }

    // like ggml_backend_sched_reserve: the copies of the measured graph hold placeholder addresses
    ggml_backend_sched_reset(sched);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_backend_sched_prefetch_bind(sched, /*dry =*/ false);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        // retry without the staging memory
        if (!ggml_backend_sched_prefetch_disable(sched) ||
            !ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            return false;
        }
    } else if (!ggml_backend_sched_prefetch_headroom_ok(sched)) {
        // the arena would take the memory the backends allocate while computing: drop it
        if (ggml_backend_sched_prefetch_disable(sched) &&
            !ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            return false;
        }
    } else if (ggml_backend_sched_prefetch_depth_init_short(sched)) {
        // the starting depth raised by ggml_backend_sched_set_prefetch_depth_init leaves less than the headroom: start at the default
        // depth, whose smaller arena is what the memory fit measured
        sched->pf_depth = sched->pf_depth0;
        ggml_backend_sched_prefetch_sync(sched);
        for (int b = 0; b < sched->n_backends; b++) {
            struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
            if (pf->arena != NULL) {
                ggml_backend_buffer_free(pf->arena);
                pf->arena      = NULL;
                pf->arena_size = 0;
            }
        }
        // split and bind again from fresh tensor copies (the bound copies of this graph hold arena addresses); the depth is now the
        // default, so this does not come back here
        sched->pf_check = false;
        ggml_backend_sched_reset(sched);
        return ggml_backend_sched_reserve(sched, measure_graph);
    }
    sched->pf_check = false;

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    // the allocation may move host compute memory that pending copies read or write
    ggml_backend_sched_ov_drain(sched);

    const int cur_copy_prev = sched->cur_copy;
    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    // k15-fn: GGML_SCHED_ALLOC_TIMING=1 logs where the host time of allocating a graph of more than 1000 nodes goes (diagnostic)
    static const bool alloc_timing = getenv("GGML_SCHED_ALLOC_TIMING") != nullptr && atoi(getenv("GGML_SCHED_ALLOC_TIMING")) != 0;
    const bool at_on = alloc_timing && graph->n_nodes > 1000;
    const int64_t at0 = at_on ? ggml_time_us() : 0;
    ggml_backend_sched_split_graph(sched, graph);
    const int64_t at1 = at_on ? ggml_time_us() : 0;
    struct at_log {
        bool on; int64_t t0, t1; int n_nodes; int n_splits;
        ~at_log() {
            if (on) {
                const int64_t t2 = ggml_time_us();
                GGML_LOG_INFO("ggml_backend_sched_alloc_graph: %d nodes, %d splits: split %.3f ms, measure + bind + alloc %.3f ms\n", n_nodes, n_splits,
                    (t1 - t0)/1000.0, (t2 - t1)/1000.0);
            }
        }
    } at_l = { at_on, at0, at1, graph->n_nodes, sched->n_splits };

    sched->alloc_over = 0;
    size_t lend_need[GGML_SCHED_MAX_BACKENDS] = { 0 };
    // k15-fn (GGML_SCHED_LEND_GUARD=0 off): a graph the lender may take is measured like one under the limit, so that a growth neither
    // lent nor fitting next to the staging arena is refused below (alloc_over, prefetch kept) instead of failing in cudaMalloc, which
    // disabled the host-weight prefetch for the rest of the process (G4 fn4 16 GB, 8-ubatch group: 2761 MiB cudaMalloc with the
    // 1275 MiB staging placed in device memory, the 16k prompt 3.84 -> 9.08 s)
    const bool lend_guard = ggml_backend_sched_lend_guard_on();
    // a merged graph (decode_layer_major sets the limit or the lend scope around it) can be split by its caller
    const bool guard_lend = lend_guard && (sched->alloc_limit_on || (sched->alloc_lend_graph && sched->callback_lend != NULL));
    size_t grow_dev[GGML_SCHED_MAX_BACKENDS] = { 0 };
    if (sched->alloc_limit_on || guard_lend) {
        // the planned compute buffers of the graph (the current ones and their allocation are kept): refused, nothing allocated and the
        // prefetch kept, if a device buffer would grow by more than the limit or leave less than the prefetch headroom (512 MiB at least)
        // free. Without the check a large graph that allocates drops the prefetch for good (ggml_backend_sched_alloc_splits).
        // Measured with the prefetched copies bound (dry: placeholders, so that the graph allocator skips them), then unbound again
        ggml_backend_sched_prefetch_bind(sched, /*dry =*/ true);
        size_t sizes[GGML_SCHED_MAX_BACKENDS] = { 0 };
        ggml_gallocr_measure_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input_cpy = tensor_copy(split->inputs[j], split->backend_id, sched->cur_copy);
                if (input_cpy != NULL && input_cpy->data == &ggml_backend_sched_dry_placeholder) {
                    input_cpy->data   = NULL;
                    input_cpy->buffer = NULL;
                }
            }
        }
        // the free memory left: the prefetch headroom (at least 512 MiB), as ggml_backend_sched_alloc_splits checks after a reallocation;
        // with a lender set that check is skipped (the lender keeps its own reserve): GGML_SCHED_ALLOC_LIMIT_FREE_MIB (64;
        // the memory fit already left its target free before: Qwen3.6 UD-Q3_K_M on G4 at 8 GB, server with MTP, grew 16 ubatches by 306 MiB)
        static const size_t lend_free = (size_t) (getenv("GGML_SCHED_ALLOC_LIMIT_FREE_MIB") ? std::max(0, atoi(getenv("GGML_SCHED_ALLOC_LIMIT_FREE_MIB"))) : 64) << 20;
        const size_t headroom = sched->callback_lend != NULL ? lend_free : std::max<size_t>(sched->pf_headroom, (size_t) 512 << 20);
        size_t over = 0;
        for (int b = 0; b < sched->n_backends; b++) {
            lend_need[b] = 0;
            ggml_backend_dev_t dev = sched->backends[b]->device;
            if (dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            const size_t cur = ggml_gallocr_get_buffer_size(sched->galloc, b);
            if (sizes[b] <= cur) {
                continue;
            }
            const size_t grow = sizes[b] - cur;
            grow_dev[b] = grow;
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            const size_t limit = sched->alloc_limit_on ? sched->alloc_limit : SIZE_MAX;
            const size_t allow = std::min(limit, free > headroom ? free - headroom : 0);
            if (grow > allow) {
                if (sched->alloc_lend_graph && sched->callback_lend != NULL) {
                    lend_need[b] = grow; // decided after the lender placed the graph (below)
                    continue;
                }
                over = std::max(over, grow);
            }
        }
        if (over > 0) {
            sched->alloc_over = over;
            sched->next_copy  = sched->cur_copy;
            sched->cur_copy   = cur_copy_prev;
            return false;
        }
    }

    ggml_backend_sched_prefetch_bind(sched, /*dry =*/ false);

    // k11: a growth past the free memory that the lender did not take is refused as by the limit (the caller splits the graph)
    {
        size_t over = 0;
        for (int b = 0; b < sched->n_backends; b++) {
            if (lend_need[b] > 0 && !sched->pf[b].lent_all) {
                over = std::max(over, lend_need[b]);
            }
            if (guard_lend && sched->pf[b].ensure_refused) {
                over = std::max(over, std::max<size_t>(grow_dev[b], 1));
            }
            // k15-fn: the binding may have allocated the staging arena in device memory (the lender took nothing): a growth that no longer
            // fits next to it is refused too
            if (guard_lend && over == 0 && grow_dev[b] > 0 && !sched->pf[b].lent_all) {
                ggml_backend_dev_t dev = sched->backends[b]->device;
                size_t free  = 0;
                size_t total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                static const size_t guard_free = (size_t) (getenv("GGML_SCHED_ALLOC_LIMIT_FREE_MIB") ? std::max(0, atoi(getenv("GGML_SCHED_ALLOC_LIMIT_FREE_MIB"))) : 64) << 20;
                if (grow_dev[b] + guard_free > free) {
                    over = std::max(over, grow_dev[b]);
                }
            }
        }
        if (over > 0) {
            // as a failed allocation below: the caller resets the scheduler, which binds the copies again for the next graph
            sched->alloc_over = over;
            sched->next_copy  = sched->cur_copy;
            sched->cur_copy   = cur_copy_prev;
            return false;
        }
    }

    if (!ggml_backend_sched_alloc_splits(sched)) {
        if (guard_lend) {
            // k15-fn: a graph the caller can split (it set the limit or the lend scope) is refused, the prefetch stays on
            sched->alloc_over = 1;
            sched->next_copy  = sched->cur_copy;
            sched->cur_copy   = cur_copy_prev;
            return false;
        }
        // retry without the staging memory
        if (!ggml_backend_sched_prefetch_disable(sched) || !ggml_backend_sched_alloc_splits(sched)) {
            return false;
        }
    }

    sched->is_alloc = true;

    return true;
}

// ---- k15-fn: saved split + allocation of one graph (see ggml_backend_sched_plan_save) -----------------------------------------------
struct ggml_backend_sched_plan {
    ggml_backend_sched_t sched;
    uint64_t gen;              // ggml_gallocr_get_generation at save
    char *   ctx_base;         // the split context's buffer and its used bytes (the input copies and dependency nodes live there)
    size_t   ctx_size;
    std::vector<char> ctx_bytes;
    size_t   hash_size;
    std::vector<uint32_t>             h_idx;     // used hash slots and their keys, backend ids and copies
    std::vector<struct ggml_tensor *> h_key;
    std::vector<int>                  h_bid;
    std::vector<struct ggml_tensor *> h_cpy;
    int n_nodes, n_leafs;
    std::vector<struct ggml_tensor *> nodes, leafs;
    uint64_t graph_uid;
    std::vector<int> node_bid, leaf_bid, prev_node_bid, prev_leaf_bid;
    int graph_size, prev_graph_size;
    std::vector<struct ggml_backend_sched_split>     splits;
    std::vector<std::vector<struct ggml_tensor *>>   split_inputs;
    std::vector<struct ggml_tensor *>                graph_inputs;
    int  pre_split, pre_nodes;
    bool input_relax_ok;
    bool ov_active;            // the host-split overlap plan is in the splits (ov_src, ov_issue_at, ...)
    size_t bytes;
};

static bool ggml_backend_sched_plan_debug() {
    static const bool on = getenv("GGML_SCHED_PLAN_DEBUG") != nullptr && atoi(getenv("GGML_SCHED_PLAN_DEBUG")) != 0;
    return on;
}

ggml_backend_sched_plan_t ggml_backend_sched_plan_save(ggml_backend_sched_t sched) {
    const char * why = sched == NULL ? "no scheduler" : !sched->is_alloc ? "not allocated" : sched->n_copies != 1 ? "pipeline copies" :
        sched->pf_active ? "prefetch" : sched->ctx == NULL ? "no split context" :
        ggml_gallocr_has_external(sched->galloc) ? "lent memory" : nullptr;
    for (int b = 0; why == nullptr && b < sched->n_backends; b++) {
        if (sched->pf[b].n_lent_bufs > 0 || sched->pf[b].lent > 0) {
            why = "lent prefetch buffers";
        }
    }
    if (why != nullptr) {
        if (ggml_backend_sched_plan_debug()) {
            GGML_LOG_INFO("%s: not saved (%s), %d nodes\n", __func__, why, sched ? sched->graph.n_nodes : 0);
        }
        return NULL;
    }
    auto * p = new ggml_backend_sched_plan();
    p->sched     = sched;
    p->gen       = ggml_gallocr_get_generation(sched->galloc);
    p->ctx_base  = sched->context_buffer;
    p->ctx_size  = sched->context_buffer_size;
    const size_t used = ggml_used_mem(sched->ctx);
    p->ctx_bytes.assign(sched->context_buffer, sched->context_buffer + used);
    p->hash_size = sched->hash_set.size;
    const size_t per     = (size_t) sched->n_backends * sched->n_copies;
    const size_t n_words = ggml_bitset_size(sched->hash_set.size);
    for (size_t w = 0; w < n_words; ++w) {
        ggml_bitset_t bits = sched->hash_set.used[w];
        while (bits) {
            const size_t i = (w << BITSET_SHR) + (size_t) ggml_bitset_ctz(bits);
            bits &= bits - 1;
            p->h_idx.push_back((uint32_t) i);
            p->h_key.push_back(sched->hash_set.keys[i]);
            p->h_bid.push_back(sched->hv_tensor_backend_ids[i]);
            p->h_cpy.insert(p->h_cpy.end(), sched->hv_tensor_copies + i*per, sched->hv_tensor_copies + (i + 1)*per);
        }
    }
    p->n_nodes = sched->graph.n_nodes;
    p->n_leafs = sched->graph.n_leafs;
    p->nodes.assign(sched->graph.nodes, sched->graph.nodes + p->n_nodes);
    p->leafs.assign(sched->graph.leafs, sched->graph.leafs + p->n_leafs);
    p->graph_uid = sched->graph.uid;
    p->node_bid.assign(sched->node_backend_ids, sched->node_backend_ids + p->n_nodes);
    p->leaf_bid.assign(sched->leaf_backend_ids, sched->leaf_backend_ids + p->n_leafs);
    p->prev_node_bid.assign(sched->prev_node_backend_ids, sched->prev_node_backend_ids + p->n_nodes);
    p->prev_leaf_bid.assign(sched->prev_leaf_backend_ids, sched->prev_leaf_backend_ids + p->n_leafs);
    p->graph_size      = sched->graph_size;
    p->prev_graph_size = sched->prev_graph_size;
    p->splits.assign(sched->splits, sched->splits + sched->n_splits);
    p->split_inputs.resize(sched->n_splits);
    for (int i = 0; i < sched->n_splits; i++) {
        p->split_inputs[i].assign(sched->splits[i].inputs, sched->splits[i].inputs + sched->splits[i].n_inputs);
    }
    p->graph_inputs.assign(sched->graph_inputs, sched->graph_inputs + sched->n_graph_inputs);
    p->pre_split      = sched->pre_split;
    p->pre_nodes      = sched->pre_nodes;
    p->input_relax_ok = sched->input_relax_ok;
    p->ov_active      = sched->ov_active;
    p->bytes = sizeof(*p) + p->ctx_bytes.size() + p->h_idx.size()*(sizeof(uint32_t) + sizeof(void *) + sizeof(int) + per*sizeof(void *)) +
        (p->nodes.size() + p->leafs.size())*(sizeof(void *) + 2*sizeof(int)) + p->splits.size()*sizeof(struct ggml_backend_sched_split) +
        p->graph_inputs.size()*sizeof(void *);
    for (const auto & v : p->split_inputs) {
        p->bytes += v.size()*sizeof(void *);
    }
    return p;
}

bool ggml_backend_sched_plan_restore(ggml_backend_sched_t sched, ggml_backend_sched_plan_t p) {
    const char * why = p == NULL ? "no plan" : p->sched != sched ? "other scheduler" : p->gen != ggml_gallocr_get_generation(sched->galloc) ?
        "graph buffers replaced" : ggml_gallocr_has_external(sched->galloc) ? "lent memory" :
        p->ctx_base != sched->context_buffer || p->ctx_size != sched->context_buffer_size ? "split context replaced" :
        p->hash_size != sched->hash_set.size ? "hash size" : sched->graph.size < std::max(p->n_nodes, p->n_leafs) ? "graph size" :
        sched->n_copies != 1 ? "pipeline copies" : sched->is_alloc ? "allocated" : nullptr;
    if (why != nullptr) {
        if (ggml_backend_sched_plan_debug()) {
            GGML_LOG_INFO("%s: not restored (%s)\n", __func__, why);
        }
        return false;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->pf[b].n_lent_bufs > 0 || sched->pf[b].lent > 0) {
            return false;
        }
    }
    // the allocation may move host compute memory that pending copies read or write (as ggml_backend_sched_alloc_graph)
    ggml_backend_sched_ov_drain(sched);
    // the split context: the next split frees and re-creates it on the same buffer, the bytes of this graph's copies go back in place
    memcpy(sched->context_buffer, p->ctx_bytes.data(), p->ctx_bytes.size());
    const size_t per = (size_t) sched->n_backends * sched->n_copies;
    for (size_t k = 0; k < p->h_idx.size(); ++k) {
        const size_t i = p->h_idx[k];
        ggml_bitset_set(sched->hash_set.used, i);
        sched->hash_set.keys[i]          = p->h_key[k];
        sched->hv_tensor_backend_ids[i]  = p->h_bid[k];
        memcpy(sched->hv_tensor_copies + i*per, p->h_cpy.data() + k*per, per*sizeof(struct ggml_tensor *));
    }
    sched->graph.n_nodes = p->n_nodes;
    sched->graph.n_leafs = p->n_leafs;
    memcpy(sched->graph.nodes, p->nodes.data(), p->nodes.size()*sizeof(struct ggml_tensor *));
    memcpy(sched->graph.leafs, p->leafs.data(), p->leafs.size()*sizeof(struct ggml_tensor *));
    sched->graph.uid = p->graph_uid;
    memcpy(sched->node_backend_ids,      p->node_bid.data(),      p->node_bid.size()*sizeof(int));
    memcpy(sched->leaf_backend_ids,      p->leaf_bid.data(),      p->leaf_bid.size()*sizeof(int));
    memcpy(sched->prev_node_backend_ids, p->prev_node_bid.data(), p->prev_node_bid.size()*sizeof(int));
    memcpy(sched->prev_leaf_backend_ids, p->prev_leaf_bid.data(), p->prev_leaf_bid.size()*sizeof(int));
    sched->graph_size      = p->graph_size;
    sched->prev_graph_size = p->prev_graph_size;
    const int n_splits = (int) p->splits.size();
    if (sched->splits_capacity < n_splits) {
        const int old_cap = sched->splits_capacity;
        sched->splits = (ggml_backend_sched_split *) realloc(sched->splits, n_splits*sizeof(struct ggml_backend_sched_split));
        GGML_ASSERT(sched->splits != NULL);
        memset(sched->splits + old_cap, 0, (n_splits - old_cap)*sizeof(struct ggml_backend_sched_split));
        sched->splits_capacity = n_splits;
    }
    for (int i = 0; i < n_splits; i++) {
        struct ggml_backend_sched_split * sp = &sched->splits[i];
        struct ggml_tensor ** inputs = sp->inputs;
        int cap = sp->inputs_capacity;
        const int n_in = (int) p->split_inputs[i].size();
        if (cap < n_in || inputs == NULL) {
            cap    = std::max(n_in, 16);
            inputs = (struct ggml_tensor **) realloc(inputs, cap*sizeof(struct ggml_tensor *));
            GGML_ASSERT(inputs != NULL);
        }
        *sp = p->splits[i];
        sp->inputs          = inputs;
        sp->inputs_capacity = cap;
        if (n_in > 0) {
            memcpy(sp->inputs, p->split_inputs[i].data(), n_in*sizeof(struct ggml_tensor *));
        }
        sp->n_inputs = n_in;
    }
    sched->n_splits = n_splits;
    while (sched->graph_inputs_capacity < (int) p->graph_inputs.size()) {
        ggml_backend_sched_graph_inputs_grow(sched);
    }
    if (!p->graph_inputs.empty()) {
        memcpy(sched->graph_inputs, p->graph_inputs.data(), p->graph_inputs.size()*sizeof(struct ggml_tensor *));
    }
    sched->n_graph_inputs = (int) p->graph_inputs.size();
    sched->pre_split      = p->pre_split;
    sched->pre_nodes      = p->pre_nodes;
    sched->input_relax_ok = p->input_relax_ok;
    sched->pf_active      = false;
    sched->ov_active      = p->ov_active;
    for (int i = 0; i < n_splits; i++) {
        sched->splits[i].ov_issued = false;
    }
    sched->is_reset       = false;
    sched->is_alloc       = true;
    return true;
}

void ggml_backend_sched_plan_free(ggml_backend_sched_plan_t plan) {
    delete plan;
}

size_t ggml_backend_sched_plan_bytes(ggml_backend_sched_plan_t plan) {
    return plan ? plan->bytes : 0;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
        sched->ov[i].upload_pending = false;
        sched->ov[i].ready_pending  = false;
    }
    ggml_backend_sched_prefetch_drain(sched);
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
    sched->callback_eval_nosync = false;
}

void ggml_backend_sched_set_eval_callback_sync(ggml_backend_sched_t sched, bool sync) {
    GGML_ASSERT(sched);
    sched->callback_eval_nosync = !sync;
}

// the copies of the allocated graph were selected with the old callbacks: the regular path fills their slots until the next allocation
static void ggml_backend_sched_prefetch_invalidate(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_splits; i++) {
        sched->splits[i].pf_async = false;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        sched->pf[b].armed = 0;
    }
}

void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    if (sched->callback_copy != callback || sched->callback_copy_user_data != user_data) {
        ggml_backend_sched_prefetch_invalidate(sched);
    }
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

void ggml_backend_sched_set_overlap(ggml_backend_sched_t sched, bool enable) {
    GGML_ASSERT(sched);
    sched->overlap = enable;
}

void ggml_backend_sched_set_reuse_weight_copies(ggml_backend_sched_t sched, bool enable) {
    GGML_ASSERT(sched);
    sched->reuse_weight_copies = enable;
}

void ggml_backend_sched_set_alloc_limit(ggml_backend_sched_t sched, bool enable, size_t max_grow) {
    GGML_ASSERT(sched);
    sched->alloc_limit_on = enable;
    sched->alloc_limit    = max_grow;
    if (!enable) {
        sched->alloc_lend_graph = false;
    }
}

void ggml_backend_sched_set_alloc_lend_graph(ggml_backend_sched_t sched, bool enable) {
    sched->alloc_lend_graph = enable;
}

void ggml_backend_sched_free_device_buffers(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_backend_sched_synchronize(sched);
    ggml_backend_sched_reset(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_dev_t dev = sched->backends[b]->device;
        if (dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (sched->pf[b].ext_buf != NULL) {
            ggml_gallocr_set_external(sched->galloc, b, NULL, 0, 0);
            sched->pf[b].ext_buf = NULL;
        }
        ggml_gallocr_free_buffer(sched->galloc, b);
    }
}

size_t ggml_backend_sched_get_alloc_over(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->alloc_over;
}

bool ggml_backend_sched_tensor_is_staged(ggml_backend_sched_t sched, const struct ggml_tensor * t) {
    GGML_ASSERT(sched);
    if (t == NULL || t->buffer == NULL || !sched->pf_active) {
        return false;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        const struct ggml_backend_sched_prefetch * pf = &sched->pf[b];
        if ((pf->arena != NULL && t->buffer == pf->arena) || ggml_backend_sched_is_lent(pf, t)) {
            return true;
        }
    }
    return false;
}

void ggml_backend_sched_set_lend_callback(ggml_backend_sched_t sched, ggml_backend_sched_lend_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    if (sched->callback_lend != callback || sched->callback_lend_user_data != user_data) {
        ggml_backend_sched_synchronize(sched);
        for (int b = 0; b < sched->n_backends; b++) {
            ggml_backend_sched_lend_end(sched, b);
        }
    }
    sched->callback_lend = callback;
    sched->callback_lend_user_data = user_data;
}

void ggml_backend_sched_set_prefetch_plan_callback(ggml_backend_sched_t sched, ggml_backend_sched_prefetch_plan_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_plan           = callback;
    sched->callback_plan_user_data = user_data;
}

void ggml_backend_sched_set_prefetch_source_callback(ggml_backend_sched_t sched, ggml_backend_sched_prefetch_source_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_source           = callback;
    sched->callback_source_user_data = user_data;
}

void ggml_backend_sched_set_prefetch_callback(ggml_backend_sched_t sched, ggml_backend_sched_prefetch_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    if (sched->callback_prefetch != callback || sched->callback_prefetch_user_data != user_data) {
        ggml_backend_sched_prefetch_invalidate(sched);
    }
    sched->callback_prefetch = callback;
    sched->callback_prefetch_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    size_t size = ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
    if (sched->pf[backend_index].arena != NULL) {
        size += ggml_backend_buffer_get_size(sched->pf[backend_index].arena);
    }
    return size;
}

void ggml_backend_sched_set_prefetch_depth_init(ggml_backend_sched_t sched, int depth) {
    if (getenv("GGML_SCHED_PREFETCH_DEPTH") != NULL || sched->pf_mode == 0 || depth <= sched->pf_depth) {
        return;
    }
    sched->pf_depth = std::min(depth, GGML_SCHED_MAX_PREFETCH_DEPTH);
}

void ggml_backend_sched_set_prefetch_headroom(ggml_backend_sched_t sched, size_t headroom) {
    GGML_ASSERT(sched);
    sched->pf_headroom = headroom;
}

size_t ggml_backend_sched_get_prefetch_headroom(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->pf_headroom;
}

size_t ggml_backend_sched_get_staging_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    const struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_index];
    return pf->n_bound > 0 ? pf->staging : 0;
}

size_t ggml_backend_sched_get_staging_split(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    const struct ggml_backend_sched_prefetch * pf = &sched->pf[backend_index];
    return pf->n_bound > 0 ? std::min(pf->need, pf->staging) : 0;
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
