#include <cxcu/cxcu.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#if CXCU_HAVE_CUDA
#include <cuda.h>
#endif

#if CXCU_HAVE_NVRTC
#include <nvrtc.h>
#endif

void cxcu_error_clear(cxcu_error* err) {
    if (!err) return;
    err->status = CXCU_STATUS_OK;
    err->cuda_code = 0;
    err->message[0] = '\0';
}

const char* cxcu_status_name(cxcu_status status) {
    switch (status) {
        case CXCU_STATUS_OK: return "ok";
        case CXCU_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case CXCU_STATUS_UNAVAILABLE: return "unavailable";
        case CXCU_STATUS_CUDA_ERROR: return "cuda_error";
        case CXCU_STATUS_OUT_OF_MEMORY: return "out_of_memory";
        case CXCU_STATUS_COMPILE_ERROR: return "compile_error";
        default: return "unknown";
    }
}

void cxcu_error_format(const char* prefix, const cxcu_error* err, char* buf, size_t size) {
    if (!buf || size == 0u) return;
    if (!prefix || prefix[0] == '\0') prefix = "CUDA error";
    if (!err || err->message[0] == '\0') {
        (void)snprintf(buf, size, "%s", prefix);
    } else {
        (void)snprintf(buf, size, "%s: %s", prefix, err->message);
    }
    buf[size - 1u] = '\0';
}

static void cxcu_set_error(
    cxcu_error* err,
    cxcu_status status,
    int cuda_code,
    const char* fmt,
    ...) {
    va_list ap;

    if (!err) return;
    err->status = status;
    err->cuda_code = cuda_code;
    err->message[0] = '\0';
    if (!fmt) return;

    va_start(ap, fmt);
    (void)vsnprintf(err->message, sizeof(err->message), fmt, ap);
    va_end(ap);
    err->message[sizeof(err->message) - 1u] = '\0';
}

char* cxcu_source_concat(const unsigned char* const* sources, const size_t* sizes, size_t count) {
    char* out;
    size_t total = 0u;
    size_t pos = 0u;
    size_t i;

    if (!sources || !sizes || count == 0u) return NULL;
    for (i = 0u; i < count; ++i) {
        size_t n;
        if (!sources[i] || sizes[i] == 0u) return NULL;
        n = sizes[i];
        if (sources[i][n - 1u] == '\0') --n;
        total += n + 1u;
    }

    out = (char*)calloc(total + 1u, 1u);
    if (!out) return NULL;
    for (i = 0u; i < count; ++i) {
        size_t n = sizes[i];
        if (sources[i][n - 1u] == '\0') --n;
        if (n > 0u) {
            memcpy(out + pos, sources[i], n);
            pos += n;
        }
        if (pos == 0u || out[pos - 1u] != '\n') out[pos++] = '\n';
    }
    out[pos] = '\0';
    return out;
}

static uint64_t cxcu_fnv1a_update(uint64_t hash, const void* data, size_t size) {
    const unsigned char* bytes = (const unsigned char*)data;
    size_t i;
    if (!bytes) return hash;
    for (i = 0u; i < size; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t cxcu_fnv1a_update_string(uint64_t hash, const char* text) {
    const size_t len = text ? strlen(text) : 0u;
    hash = cxcu_fnv1a_update(hash, &len, sizeof(len));
    return cxcu_fnv1a_update(hash, text ? text : "", len);
}

static uint64_t cxcu_fnv1a_update_string_with_suffix(uint64_t hash, const char* text, const char* suffix) {
    const size_t text_len = text ? strlen(text) : 0u;
    const size_t suffix_len = suffix ? strlen(suffix) : 0u;
    const size_t len = text_len + suffix_len;
    hash = cxcu_fnv1a_update(hash, &len, sizeof(len));
    hash = cxcu_fnv1a_update(hash, text ? text : "", text_len);
    return cxcu_fnv1a_update(hash, suffix ? suffix : "", suffix_len);
}

static int cxcu_mkdir_one(const char* path) {
    struct stat st;
    if (!path || path[0] == '\0') return 0;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 1 : 0;
    if (mkdir(path, 0755) == 0) return 1;
    if (errno == EEXIST && stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 1;
    return 0;
}

static int cxcu_mkdir_p(const char* path) {
    char tmp[PATH_MAX];
    size_t len;
    size_t i;

    if (!path || path[0] == '\0') return 0;
    len = strlen(path);
    if (len >= sizeof(tmp)) return 0;
    memcpy(tmp, path, len + 1u);
    for (i = 1u; i < len; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (tmp[0] != '\0' && !cxcu_mkdir_one(tmp)) return 0;
            tmp[i] = '/';
        }
    }
    return cxcu_mkdir_one(tmp);
}

static int cxcu_file_exists(const char* path) {
    struct stat st;
    if (!path || path[0] == '\0') return 0;
    if (stat(path, &st) != 0) return 0;
    return S_ISREG(st.st_mode) && st.st_size > 0;
}

#if CXCU_HAVE_CUDA

static CUcontext g_cxcu_context = NULL;
static int g_cxcu_context_ready = 0;

static void cxcu_set_cuda_error(
    cxcu_error* err,
    cxcu_status status,
    CUresult result,
    const char* operation) {
    const char* name = NULL;
    const char* text = NULL;

    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &text);
    cxcu_set_error(
        err,
        status,
        (int)result,
        "%s failed: %s%s%s",
        operation ? operation : "CUDA call",
        name ? name : "CUDA_ERROR_UNKNOWN",
        text ? " - " : "",
        text ? text : "");
}

static int cxcu_init_driver(cxcu_error* err) {
    CUresult result = cuInit(0u);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_UNAVAILABLE, result, "cuInit");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

static int cxcu_ensure_context(cxcu_error* err) {
    CUdevice device;
    CUresult result;
    int count = 0;

    if (g_cxcu_context_ready && g_cxcu_context) {
        result = cuCtxSetCurrent(g_cxcu_context);
        if (result != CUDA_SUCCESS) {
            cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuCtxSetCurrent");
            return 0;
        }
        cxcu_error_clear(err);
        return 1;
    }

    if (!cxcu_init_driver(err)) return 0;

    result = cuDeviceGetCount(&count);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_UNAVAILABLE, result, "cuDeviceGetCount");
        return 0;
    }
    if (count <= 0) {
        cxcu_set_error(err, CXCU_STATUS_UNAVAILABLE, 0, "no CUDA devices found");
        return 0;
    }

    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDeviceGet");
        return 0;
    }
    result = cuDevicePrimaryCtxRetain(&g_cxcu_context, device);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDevicePrimaryCtxRetain");
        return 0;
    }
    result = cuCtxSetCurrent(g_cxcu_context);
    if (result != CUDA_SUCCESS) {
        (void)cuDevicePrimaryCtxRelease(device);
        g_cxcu_context = NULL;
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuCtxSetCurrent");
        return 0;
    }

    g_cxcu_context_ready = 1;
    cxcu_error_clear(err);
    return 1;
}

int cxcu_available(cxcu_error* err) {
    int count = 0;
    CUresult result;

    if (!cxcu_init_driver(err)) return 0;
    result = cuDeviceGetCount(&count);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_UNAVAILABLE, result, "cuDeviceGetCount");
        return 0;
    }
    if (count <= 0) {
        cxcu_set_error(err, CXCU_STATUS_UNAVAILABLE, 0, "no CUDA devices found");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

int cxcu_device_count(int* out_count, cxcu_error* err) {
    CUresult result;
    int count = 0;

    if (!out_count) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "out_count is required");
        return 0;
    }
    *out_count = 0;
    if (!cxcu_init_driver(err)) return 0;
    result = cuDeviceGetCount(&count);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_UNAVAILABLE, result, "cuDeviceGetCount");
        return 0;
    }
    *out_count = count;
    cxcu_error_clear(err);
    return 1;
}

int cxcu_device_info_get(int ordinal, cxcu_device_info* out_info, cxcu_error* err) {
    CUdevice device;
    CUresult result;
    int count = 0;
    size_t total_memory = 0u;

    if (!out_info) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "out_info is required");
        return 0;
    }
    memset(out_info, 0, sizeof(*out_info));
    if (!cxcu_device_count(&count, err)) return 0;
    if (ordinal < 0 || ordinal >= count) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "device ordinal is out of range");
        return 0;
    }

    result = cuDeviceGet(&device, ordinal);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDeviceGet");
        return 0;
    }
    result = cuDeviceGetName(out_info->name, (int)sizeof(out_info->name), device);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDeviceGetName");
        return 0;
    }
    result = cuDeviceGetAttribute(
        &out_info->compute_capability_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDeviceGetAttribute major");
        return 0;
    }
    result = cuDeviceGetAttribute(
        &out_info->compute_capability_minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDeviceGetAttribute minor");
        return 0;
    }
    result = cuDeviceTotalMem(&total_memory, device);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuDeviceTotalMem");
        return 0;
    }

    out_info->ordinal = ordinal;
    out_info->total_memory = total_memory;
    cxcu_error_clear(err);
    return 1;
}

int cxcu_stream_create(cxcu_stream* out_stream, cxcu_error* err) {
    CUstream stream = NULL;
    CUresult result;

    if (!out_stream) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "out_stream is required");
        return 0;
    }
    memset(out_stream, 0, sizeof(*out_stream));
    if (!cxcu_ensure_context(err)) return 0;
    result = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuStreamCreate");
        return 0;
    }
    out_stream->handle = (uintptr_t)stream;
    cxcu_error_clear(err);
    return 1;
}

void cxcu_stream_destroy(cxcu_stream* stream) {
    if (!stream || !stream->handle) return;
    if (cxcu_ensure_context(NULL)) {
        (void)cuStreamDestroy((CUstream)stream->handle);
    }
    stream->handle = 0u;
}

int cxcu_buffer_alloc(cxcu_buffer* out_buffer, size_t bytes, cxcu_error* err) {
    CUdeviceptr ptr = 0;
    CUresult result;

    if (!out_buffer || bytes == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "non-empty output buffer is required");
        return 0;
    }
    memset(out_buffer, 0, sizeof(*out_buffer));
    if (!cxcu_ensure_context(err)) return 0;
    result = cuMemAlloc(&ptr, bytes);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_OUT_OF_MEMORY, result, "cuMemAlloc");
        return 0;
    }
    out_buffer->device_ptr = (uintptr_t)ptr;
    out_buffer->bytes = bytes;
    cxcu_error_clear(err);
    return 1;
}

int cxcu_buffer_alloc_upload(cxcu_buffer* out_buffer, const void* host, size_t bytes, cxcu_error* err) {
    if (!out_buffer || !host || bytes == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "valid output buffer and host data are required");
        if (out_buffer) memset(out_buffer, 0, sizeof(*out_buffer));
        return 0;
    }
    if (!cxcu_buffer_alloc(out_buffer, bytes, err)) return 0;
    if (!cxcu_memcpy_h2d(out_buffer, host, bytes, err)) {
        cxcu_buffer_free(out_buffer);
        return 0;
    }
    return 1;
}

void cxcu_buffer_free(cxcu_buffer* buffer) {
    if (!buffer || !buffer->device_ptr) return;
    if (cxcu_ensure_context(NULL)) {
        (void)cuMemFree((CUdeviceptr)buffer->device_ptr);
    }
    buffer->device_ptr = 0u;
    buffer->bytes = 0u;
}

void cxcu_buffer_group_init(cxcu_buffer_group* group, cxcu_buffer** storage, size_t capacity) {
    if (!group) return;
    group->buffers = storage;
    group->count = 0u;
    group->capacity = storage ? capacity : 0u;
}

int cxcu_buffer_group_add(cxcu_buffer_group* group, cxcu_buffer* buffer, cxcu_error* err) {
    if (!group || !group->buffers || !buffer || group->count >= group->capacity) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "buffer group capacity exceeded");
        return 0;
    }
    group->buffers[group->count++] = buffer;
    cxcu_error_clear(err);
    return 1;
}

int cxcu_buffer_group_alloc(cxcu_buffer_group* group, cxcu_buffer* out_buffer, size_t bytes, cxcu_error* err) {
    if (!cxcu_buffer_alloc(out_buffer, bytes, err)) return 0;
    if (!cxcu_buffer_group_add(group, out_buffer, err)) {
        cxcu_buffer_free(out_buffer);
        return 0;
    }
    return 1;
}

int cxcu_buffer_group_alloc_upload(
    cxcu_buffer_group* group,
    cxcu_buffer* out_buffer,
    const void* host,
    size_t bytes,
    cxcu_error* err) {
    if (!cxcu_buffer_alloc_upload(out_buffer, host, bytes, err)) return 0;
    if (!cxcu_buffer_group_add(group, out_buffer, err)) {
        cxcu_buffer_free(out_buffer);
        return 0;
    }
    return 1;
}

void cxcu_buffer_group_free_all(cxcu_buffer_group* group) {
    size_t i;
    if (!group || !group->buffers) return;
    for (i = group->count; i > 0u; --i) {
        cxcu_buffer_free(group->buffers[i - 1u]);
    }
    group->count = 0u;
}

int cxcu_memcpy_h2d(cxcu_buffer* dst, const void* src, size_t bytes, cxcu_error* err) {
    CUresult result;

    if (!dst || !dst->device_ptr || !src || bytes == 0u || bytes > dst->bytes) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "invalid host-to-device copy arguments");
        return 0;
    }
    if (!cxcu_ensure_context(err)) return 0;
    result = cuMemcpyHtoD((CUdeviceptr)dst->device_ptr, src, bytes);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuMemcpyHtoD");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

int cxcu_memcpy_d2h(void* dst, const cxcu_buffer* src, size_t bytes, cxcu_error* err) {
    CUresult result;

    if (!dst || !src || !src->device_ptr || bytes == 0u || bytes > src->bytes) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "invalid device-to-host copy arguments");
        return 0;
    }
    if (!cxcu_ensure_context(err)) return 0;
    result = cuMemcpyDtoH(dst, (CUdeviceptr)src->device_ptr, bytes);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuMemcpyDtoH");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

int cxcu_module_load_data(cxcu_module* out_module, const void* image, size_t image_size, cxcu_error* err) {
    CUmodule module = NULL;
    CUresult result;

    if (!out_module || !image || image_size == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "module image is required");
        return 0;
    }
    memset(out_module, 0, sizeof(*out_module));
    if (!cxcu_ensure_context(err)) return 0;
    result = cuModuleLoadData(&module, image);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuModuleLoadData");
        return 0;
    }
    out_module->handle = (uintptr_t)module;
    cxcu_error_clear(err);
    return 1;
}

void cxcu_module_unload(cxcu_module* module) {
    if (!module || !module->handle) return;
    if (cxcu_ensure_context(NULL)) {
        (void)cuModuleUnload((CUmodule)module->handle);
    }
    module->handle = 0u;
}

#if CXCU_HAVE_NVRTC

static void cxcu_set_nvrtc_error(
    cxcu_error* err,
    nvrtcResult result,
    const char* operation,
    const char* log) {
    const char* text = nvrtcGetErrorString(result);
    if (log && log[0] != '\0') {
        cxcu_set_error(
            err,
            CXCU_STATUS_COMPILE_ERROR,
            (int)result,
            "%s failed: %s: %.180s",
            operation ? operation : "NVRTC call",
            text ? text : "NVRTC_ERROR_UNKNOWN",
            log);
        return;
    }
    cxcu_set_error(
        err,
        CXCU_STATUS_COMPILE_ERROR,
        (int)result,
        "%s failed: %s",
        operation ? operation : "NVRTC call",
        text ? text : "NVRTC_ERROR_UNKNOWN");
}

int cxcu_compile_ptx(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    cxcu_ptx* out_ptx,
    cxcu_error* err) {
    nvrtcProgram program = NULL;
    nvrtcResult result;
    char* ptx = NULL;
    char* log = NULL;
    size_t log_size = 0u;
    size_t ptx_size = 0u;
    int ok = 0;

    if (!cuda_source || !out_ptx) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "CUDA source and out_ptx are required");
        return 0;
    }
    memset(out_ptx, 0, sizeof(*out_ptx));

    result = nvrtcCreateProgram(
        &program,
        cuda_source,
        program_name && program_name[0] != '\0' ? program_name : "cxcu_program.cu",
        0,
        NULL,
        NULL);
    if (result != NVRTC_SUCCESS) {
        cxcu_set_nvrtc_error(err, result, "nvrtcCreateProgram", NULL);
        return 0;
    }

    result = nvrtcCompileProgram(program, (int)option_count, options);
    (void)nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1u) {
        log = (char*)malloc(log_size);
        if (log) (void)nvrtcGetProgramLog(program, log);
    }
    if (result != NVRTC_SUCCESS) {
        cxcu_set_nvrtc_error(err, result, "nvrtcCompileProgram", log);
        goto cleanup;
    }

    result = nvrtcGetPTXSize(program, &ptx_size);
    if (result != NVRTC_SUCCESS || ptx_size == 0u) {
        cxcu_set_nvrtc_error(err, result, "nvrtcGetPTXSize", log);
        goto cleanup;
    }
    ptx = (char*)malloc(ptx_size);
    if (!ptx) {
        cxcu_set_error(err, CXCU_STATUS_OUT_OF_MEMORY, 0, "failed to allocate PTX buffer");
        goto cleanup;
    }
    result = nvrtcGetPTX(program, ptx);
    if (result != NVRTC_SUCCESS) {
        cxcu_set_nvrtc_error(err, result, "nvrtcGetPTX", log);
        goto cleanup;
    }

    out_ptx->data = ptx;
    out_ptx->size = ptx_size;
    ptx = NULL;
    cxcu_error_clear(err);
    ok = 1;

cleanup:
    free(ptx);
    free(log);
    if (program) (void)nvrtcDestroyProgram(&program);
    return ok;
}

static int cxcu_nvrtc_option_is_arch_key(const char* option) {
    return option &&
           (strcmp(option, "--gpu-architecture") == 0 ||
            strcmp(option, "-arch") == 0);
}

static int cxcu_nvrtc_option_is_arch_assignment(const char* option) {
    return option &&
           (strncmp(option, "--gpu-architecture=", 19u) == 0 ||
            strncmp(option, "-arch=", 6u) == 0);
}

static int cxcu_build_sm_arch_option(int device_ordinal,
                                     char* out,
                                     size_t out_size,
                                     cxcu_error* err) {
    cxcu_device_info info;

    if (!out || out_size == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "architecture buffer is required");
        return 0;
    }
    if (device_ordinal < 0) device_ordinal = 0;
    if (!cxcu_device_info_get(device_ordinal, &info, err)) return 0;
    if (info.compute_capability_major <= 0 || info.compute_capability_minor < 0) {
        cxcu_set_error(err, CXCU_STATUS_UNAVAILABLE, 0, "invalid CUDA compute capability");
        return 0;
    }
    (void)snprintf(
        out,
        out_size,
        "--gpu-architecture=sm_%d%d",
        info.compute_capability_major,
        info.compute_capability_minor);
    out[out_size - 1u] = '\0';
    return 1;
}

static const char** cxcu_build_cubin_options(const char* const* options,
                                             size_t option_count,
                                             const char* arch_option,
                                             size_t* out_count,
                                             cxcu_error* err) {
    const char** cubin_options;
    size_t count = 0u;
    size_t i;
    int skip_arch_value = 0;

    if (out_count) *out_count = 0u;
    if (!arch_option || !out_count) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "architecture option is required");
        return NULL;
    }
    cubin_options = (const char**)calloc(option_count + 1u, sizeof(*cubin_options));
    if (!cubin_options) {
        cxcu_set_error(err, CXCU_STATUS_OUT_OF_MEMORY, 0, "failed to allocate NVRTC options");
        return NULL;
    }

    for (i = 0u; i < option_count; ++i) {
        const char* option = options ? options[i] : NULL;
        if (!option) continue;
        if (skip_arch_value) {
            skip_arch_value = 0;
            continue;
        }
        if (cxcu_nvrtc_option_is_arch_key(option)) {
            skip_arch_value = 1;
            continue;
        }
        if (cxcu_nvrtc_option_is_arch_assignment(option)) continue;
        cubin_options[count++] = option;
    }
    cubin_options[count++] = arch_option;
    *out_count = count;
    return cubin_options;
}

static int cxcu_compile_cubin_for_device(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    int device_ordinal,
    cxcu_module_image* out_image,
    cxcu_error* err) {
    char arch_option[64];
    const char** cubin_options = NULL;
    size_t cubin_option_count = 0u;
    nvrtcProgram program = NULL;
    nvrtcResult result;
    char* cubin = NULL;
    char* log = NULL;
    size_t log_size = 0u;
    size_t cubin_size = 0u;
    int ok = 0;

    if (!cxcu_build_sm_arch_option(device_ordinal, arch_option, sizeof(arch_option), err)) {
        return 0;
    }
    cubin_options = cxcu_build_cubin_options(
        options,
        option_count,
        arch_option,
        &cubin_option_count,
        err);
    if (!cubin_options) return 0;

    result = nvrtcCreateProgram(
        &program,
        cuda_source,
        program_name && program_name[0] != '\0' ? program_name : "cxcu_program.cu",
        0,
        NULL,
        NULL);
    if (result != NVRTC_SUCCESS) {
        cxcu_set_nvrtc_error(err, result, "nvrtcCreateProgram", NULL);
        goto cleanup;
    }

    result = nvrtcCompileProgram(program, (int)cubin_option_count, cubin_options);
    (void)nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1u) {
        log = (char*)malloc(log_size);
        if (log) (void)nvrtcGetProgramLog(program, log);
    }
    if (result != NVRTC_SUCCESS) {
        cxcu_set_nvrtc_error(err, result, "nvrtcCompileProgram", log);
        goto cleanup;
    }

    result = nvrtcGetCUBINSize(program, &cubin_size);
    if (result != NVRTC_SUCCESS || cubin_size == 0u) {
        cxcu_set_nvrtc_error(err, result, "nvrtcGetCUBINSize", log);
        goto cleanup;
    }
    cubin = (char*)malloc(cubin_size);
    if (!cubin) {
        cxcu_set_error(err, CXCU_STATUS_OUT_OF_MEMORY, 0, "failed to allocate CUBIN buffer");
        goto cleanup;
    }
    result = nvrtcGetCUBIN(program, cubin);
    if (result != NVRTC_SUCCESS) {
        cxcu_set_nvrtc_error(err, result, "nvrtcGetCUBIN", log);
        goto cleanup;
    }

    out_image->data = cubin;
    out_image->size = cubin_size;
    out_image->kind = CXCU_MODULE_IMAGE_CUBIN;
    (void)snprintf(
        out_image->architecture,
        sizeof(out_image->architecture),
        "%.31s",
        arch_option);
    out_image->architecture[sizeof(out_image->architecture) - 1u] = '\0';
    cubin = NULL;
    cxcu_error_clear(err);
    ok = 1;

cleanup:
    free(cubin);
    free(log);
    free(cubin_options);
    if (program) (void)nvrtcDestroyProgram(&program);
    return ok;
}

int cxcu_compile_module_image_for_device(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    int device_ordinal,
    cxcu_module_image* out_image,
    cxcu_error* err) {
    cxcu_ptx ptx = {0};
    cxcu_error cubin_err = {0};

    if (!cuda_source || !out_image) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "CUDA source and out_image are required");
        return 0;
    }
    memset(out_image, 0, sizeof(*out_image));

    if (cxcu_compile_cubin_for_device(
            cuda_source,
            program_name,
            options,
            option_count,
            device_ordinal,
            out_image,
            &cubin_err)) {
        cxcu_error_clear(err);
        return 1;
    }

    if (!cxcu_compile_ptx(cuda_source, program_name, options, option_count, &ptx, err)) {
        if (err && err->message[0] == '\0') *err = cubin_err;
        return 0;
    }
    out_image->data = ptx.data;
    out_image->size = ptx.size;
    out_image->kind = CXCU_MODULE_IMAGE_PTX;
    (void)snprintf(out_image->architecture, sizeof(out_image->architecture), "%s", "ptx");
    out_image->architecture[sizeof(out_image->architecture) - 1u] = '\0';
    ptx.data = NULL;
    ptx.size = 0u;
    cxcu_error_clear(err);
    return 1;
}

#else

int cxcu_compile_ptx(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    cxcu_ptx* out_ptx,
    cxcu_error* err) {
    (void)cuda_source;
    (void)program_name;
    (void)options;
    (void)option_count;
    if (out_ptx) memset(out_ptx, 0, sizeof(*out_ptx));
    cxcu_set_error(err, CXCU_STATUS_UNAVAILABLE, 0, "cxcu was built without NVRTC support");
    return 0;
}

int cxcu_compile_module_image_for_device(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    int device_ordinal,
    cxcu_module_image* out_image,
    cxcu_error* err) {
    (void)cuda_source;
    (void)program_name;
    (void)options;
    (void)option_count;
    (void)device_ordinal;
    if (out_image) memset(out_image, 0, sizeof(*out_image));
    cxcu_set_error(err, CXCU_STATUS_UNAVAILABLE, 0, "cxcu was built without NVRTC support");
    return 0;
}

#endif

int cxcu_module_cache_path(
    const cxcu_module_cache* cache,
    const char* cuda_source,
    const char* const* options,
    size_t option_count,
    char* out_path,
    size_t out_path_size,
    char* out_key,
    size_t out_key_size,
    cxcu_error* err) {
    cxcu_device_info info;
    cxcu_error device_err = {0};
    char device_key[384];
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 1099511628211ULL;
    const char* prefix;
    const char* extension;
    int written;
    size_t i;

    if (out_path && out_path_size > 0u) out_path[0] = '\0';
    if (out_key && out_key_size > 0u) out_key[0] = '\0';
    if (!cache || !cache->enabled || !cache->directory || cache->directory[0] == '\0' ||
        !cache->namespace_tag || cache->namespace_tag[0] == '\0' || !cuda_source ||
        !out_path || out_path_size == 0u || !out_key || out_key_size == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "invalid module cache arguments");
        return 0;
    }

    memset(&info, 0, sizeof(info));
    if (!cxcu_device_info_get(cache->device_ordinal, &info, &device_err)) {
        (void)snprintf(device_key, sizeof(device_key), "device=unknown");
    } else {
        (void)snprintf(
            device_key,
            sizeof(device_key),
            "sm_%d%d:%s",
            info.compute_capability_major,
            info.compute_capability_minor,
            info.name);
    }
    device_key[sizeof(device_key) - 1u] = '\0';

    h1 = cxcu_fnv1a_update_string(h1, cache->namespace_tag);
    h1 = cxcu_fnv1a_update_string(h1, device_key);
    h1 = cxcu_fnv1a_update_string(h1, cuda_source);
    h1 = cxcu_fnv1a_update(h1, &option_count, sizeof(option_count));
    h2 = cxcu_fnv1a_update_string_with_suffix(h2, cache->namespace_tag, "b");
    h2 = cxcu_fnv1a_update_string(h2, device_key);
    h2 = cxcu_fnv1a_update_string(h2, cuda_source);
    h2 = cxcu_fnv1a_update(h2, &option_count, sizeof(option_count));
    for (i = 0u; i < option_count; ++i) {
        h1 = cxcu_fnv1a_update_string(h1, options ? options[i] : "");
        h2 = cxcu_fnv1a_update_string(h2, options ? options[i] : "");
    }

    written = snprintf(
        out_key,
        out_key_size,
        "%016llx%016llx",
        (unsigned long long)h1,
        (unsigned long long)h2);
    if (written <= 0 || (size_t)written >= out_key_size) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "module cache key buffer is too small");
        return 0;
    }
    prefix = cache->file_prefix ? cache->file_prefix : "";
    extension = cache->file_extension ? cache->file_extension : "";
    written = snprintf(out_path, out_path_size, "%s/%s%s%s", cache->directory, prefix, out_key, extension);
    if (written <= 0 || (size_t)written >= out_path_size) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "module cache path buffer is too small");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

int cxcu_module_cache_read_image(const char* path, cxcu_module_image* out_image) {
    FILE* in;
    long size;
    void* data = NULL;
    size_t nread;

    if (out_image) memset(out_image, 0, sizeof(*out_image));
    if (!path || !out_image) return 0;
    in = fopen(path, "rb");
    if (!in) return 0;
    if (fseek(in, 0, SEEK_END) != 0) {
        fclose(in);
        return 0;
    }
    size = ftell(in);
    if (size <= 0 || fseek(in, 0, SEEK_SET) != 0) {
        fclose(in);
        return 0;
    }
    data = malloc((size_t)size);
    if (!data) {
        fclose(in);
        return 0;
    }
    nread = fread(data, 1u, (size_t)size, in);
    fclose(in);
    if (nread != (size_t)size) {
        free(data);
        return 0;
    }
    out_image->data = data;
    out_image->size = (size_t)size;
    out_image->kind = CXCU_MODULE_IMAGE_CUBIN;
    (void)snprintf(out_image->architecture, sizeof(out_image->architecture), "%s", "cache");
    out_image->architecture[sizeof(out_image->architecture) - 1u] = '\0';
    return 1;
}

int cxcu_module_cache_write_image(const char* path, const void* data, size_t size) {
    char tmp_path[PATH_MAX];
    FILE* out;
    const char* slash;
    char dir[PATH_MAX];
    size_t dir_len;
    int written;
    int ok = 0;

    if (!path || !data || size == 0u) return 0;
    slash = strrchr(path, '/');
    if (slash) {
        dir_len = (size_t)(slash - path);
        if (dir_len == 0u || dir_len >= sizeof(dir)) return 0;
        memcpy(dir, path, dir_len);
        dir[dir_len] = '\0';
        if (!cxcu_mkdir_p(dir)) return 0;
    }
    written = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (written <= 0 || (size_t)written >= sizeof(tmp_path)) return 0;

    out = fopen(tmp_path, "wb");
    if (!out) return 0;
    if (fwrite(data, 1u, size, out) == size && fflush(out) == 0) ok = 1;
    if (fclose(out) != 0) ok = 0;
    if (!ok) {
        (void)remove(tmp_path);
        return 0;
    }
    if (rename(tmp_path, path) != 0) {
        (void)remove(tmp_path);
        return 0;
    }
    return 1;
}

int cxcu_module_cache_image_exists(
    const cxcu_module_cache* cache,
    const char* cuda_source,
    const char* const* options,
    size_t option_count) {
    char path[PATH_MAX];
    char key[65];
    if (!cxcu_module_cache_path(
            cache,
            cuda_source,
            options,
            option_count,
            path,
            sizeof(path),
            key,
            sizeof(key),
            NULL)) {
        return 0;
    }
    return cxcu_file_exists(path);
}

int cxcu_compile_module_image_cached(
    const cxcu_module_cache* cache,
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    cxcu_module_image* out_image,
    cxcu_error* err) {
    char path[PATH_MAX];
    char key[65];
    int have_path = 0;

    if (!cuda_source || !out_image) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "CUDA source and out_image are required");
        return 0;
    }
    memset(out_image, 0, sizeof(*out_image));
    if (cache && cache->enabled) {
        have_path = cxcu_module_cache_path(
            cache,
            cuda_source,
            options,
            option_count,
            path,
            sizeof(path),
            key,
            sizeof(key),
            NULL);
        if (have_path && cxcu_module_cache_read_image(path, out_image)) {
            cxcu_error_clear(err);
            return 1;
        }
    }
    if (!cxcu_compile_module_image_for_device(
            cuda_source,
            program_name,
            options,
            option_count,
            cache ? cache->device_ordinal : 0,
            out_image,
            err)) {
        return 0;
    }
    if (have_path) {
        (void)cxcu_module_cache_write_image(path, out_image->data, out_image->size);
    }
    return 1;
}

void cxcu_ptx_free(cxcu_ptx* ptx) {
    if (!ptx) return;
    free(ptx->data);
    memset(ptx, 0, sizeof(*ptx));
}

void cxcu_module_image_free(cxcu_module_image* image) {
    if (!image) return;
    free(image->data);
    memset(image, 0, sizeof(*image));
}

int cxcu_launch(
    cxcu_module* module,
    const char* kernel_name,
    const cxcu_launch_config* cfg,
    void** args,
    cxcu_error* err) {
    CUfunction fn = NULL;
    CUstream stream = NULL;
    CUresult result;

    if (!module || !module->handle || !kernel_name || !cfg ||
        cfg->grid_x == 0u || cfg->grid_y == 0u || cfg->grid_z == 0u ||
        cfg->block_x == 0u || cfg->block_y == 0u || cfg->block_z == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "invalid kernel launch arguments");
        return 0;
    }
    if (!cxcu_ensure_context(err)) return 0;
    result = cuModuleGetFunction(&fn, (CUmodule)module->handle, kernel_name);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuModuleGetFunction");
        return 0;
    }
    if (cfg->stream && cfg->stream->handle) {
        stream = (CUstream)cfg->stream->handle;
    }
    result = cuLaunchKernel(
        fn,
        cfg->grid_x,
        cfg->grid_y,
        cfg->grid_z,
        cfg->block_x,
        cfg->block_y,
        cfg->block_z,
        cfg->shared_mem_bytes,
        stream,
        args,
        NULL);
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuLaunchKernel");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

int cxcu_synchronize(cxcu_error* err) {
    CUresult result;
    if (!cxcu_ensure_context(err)) return 0;
    result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        cxcu_set_cuda_error(err, CXCU_STATUS_CUDA_ERROR, result, "cuCtxSynchronize");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

void cxcu_shutdown(void) {
    CUcontext ctx = g_cxcu_context;
    CUdevice device;
    if (!g_cxcu_context_ready || !ctx) return;
    if (cuCtxGetDevice(&device) == CUDA_SUCCESS) {
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    g_cxcu_context = NULL;
    g_cxcu_context_ready = 0;
}

#else

static int cxcu_stub_unavailable(cxcu_error* err) {
    cxcu_set_error(
        err,
        CXCU_STATUS_UNAVAILABLE,
        0,
        "cxcu was built without CUDA Driver API support");
    return 0;
}

int cxcu_available(cxcu_error* err) {
    return cxcu_stub_unavailable(err);
}

int cxcu_device_count(int* out_count, cxcu_error* err) {
    if (out_count) *out_count = 0;
    return cxcu_stub_unavailable(err);
}

int cxcu_device_info_get(int ordinal, cxcu_device_info* out_info, cxcu_error* err) {
    (void)ordinal;
    if (out_info) memset(out_info, 0, sizeof(*out_info));
    return cxcu_stub_unavailable(err);
}

int cxcu_stream_create(cxcu_stream* out_stream, cxcu_error* err) {
    if (out_stream) memset(out_stream, 0, sizeof(*out_stream));
    return cxcu_stub_unavailable(err);
}

void cxcu_stream_destroy(cxcu_stream* stream) {
    if (stream) memset(stream, 0, sizeof(*stream));
}

int cxcu_buffer_alloc(cxcu_buffer* out_buffer, size_t bytes, cxcu_error* err) {
    (void)bytes;
    if (out_buffer) memset(out_buffer, 0, sizeof(*out_buffer));
    return cxcu_stub_unavailable(err);
}

void cxcu_buffer_free(cxcu_buffer* buffer) {
    if (buffer) memset(buffer, 0, sizeof(*buffer));
}

int cxcu_memcpy_h2d(cxcu_buffer* dst, const void* src, size_t bytes, cxcu_error* err) {
    (void)dst;
    (void)src;
    (void)bytes;
    return cxcu_stub_unavailable(err);
}

int cxcu_memcpy_d2h(void* dst, const cxcu_buffer* src, size_t bytes, cxcu_error* err) {
    (void)dst;
    (void)src;
    (void)bytes;
    return cxcu_stub_unavailable(err);
}

int cxcu_module_load_data(cxcu_module* out_module, const void* image, size_t image_size, cxcu_error* err) {
    (void)image;
    (void)image_size;
    if (out_module) memset(out_module, 0, sizeof(*out_module));
    return cxcu_stub_unavailable(err);
}

void cxcu_module_unload(cxcu_module* module) {
    if (module) memset(module, 0, sizeof(*module));
}

int cxcu_compile_ptx(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    cxcu_ptx* out_ptx,
    cxcu_error* err) {
    (void)cuda_source;
    (void)program_name;
    (void)options;
    (void)option_count;
    if (out_ptx) memset(out_ptx, 0, sizeof(*out_ptx));
    return cxcu_stub_unavailable(err);
}

int cxcu_compile_module_image_for_device(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    int device_ordinal,
    cxcu_module_image* out_image,
    cxcu_error* err) {
    (void)cuda_source;
    (void)program_name;
    (void)options;
    (void)option_count;
    (void)device_ordinal;
    if (out_image) memset(out_image, 0, sizeof(*out_image));
    return cxcu_stub_unavailable(err);
}

int cxcu_buffer_alloc_upload(cxcu_buffer* out_buffer, const void* host, size_t bytes, cxcu_error* err) {
    if (!out_buffer || !host || bytes == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "valid output buffer and host data are required");
        if (out_buffer) memset(out_buffer, 0, sizeof(*out_buffer));
        return 0;
    }
    if (out_buffer) memset(out_buffer, 0, sizeof(*out_buffer));
    return cxcu_stub_unavailable(err);
}

void cxcu_buffer_group_init(cxcu_buffer_group* group, cxcu_buffer** storage, size_t capacity) {
    if (!group) return;
    group->buffers = storage;
    group->count = 0u;
    group->capacity = storage ? capacity : 0u;
}

int cxcu_buffer_group_add(cxcu_buffer_group* group, cxcu_buffer* buffer, cxcu_error* err) {
    if (!group || !group->buffers || !buffer || group->count >= group->capacity) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "buffer group capacity exceeded");
        return 0;
    }
    group->buffers[group->count++] = buffer;
    cxcu_error_clear(err);
    return 1;
}

int cxcu_buffer_group_alloc(cxcu_buffer_group* group, cxcu_buffer* out_buffer, size_t bytes, cxcu_error* err) {
    if (!cxcu_buffer_alloc(out_buffer, bytes, err)) return 0;
    if (!cxcu_buffer_group_add(group, out_buffer, err)) {
        cxcu_buffer_free(out_buffer);
        return 0;
    }
    return 1;
}

int cxcu_buffer_group_alloc_upload(
    cxcu_buffer_group* group,
    cxcu_buffer* out_buffer,
    const void* host,
    size_t bytes,
    cxcu_error* err) {
    if (!cxcu_buffer_alloc_upload(out_buffer, host, bytes, err)) return 0;
    if (!cxcu_buffer_group_add(group, out_buffer, err)) {
        cxcu_buffer_free(out_buffer);
        return 0;
    }
    return 1;
}

void cxcu_buffer_group_free_all(cxcu_buffer_group* group) {
    size_t i;
    if (!group || !group->buffers) return;
    for (i = group->count; i > 0u; --i) {
        cxcu_buffer_free(group->buffers[i - 1u]);
    }
    group->count = 0u;
}

int cxcu_module_cache_path(
    const cxcu_module_cache* cache,
    const char* cuda_source,
    const char* const* options,
    size_t option_count,
    char* out_path,
    size_t out_path_size,
    char* out_key,
    size_t out_key_size,
    cxcu_error* err) {
    char device_key[384];
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 1099511628211ULL;
    const char* prefix;
    const char* extension;
    int written;
    size_t i;

    if (out_path && out_path_size > 0u) out_path[0] = '\0';
    if (out_key && out_key_size > 0u) out_key[0] = '\0';
    if (!cache || !cache->enabled || !cache->directory || cache->directory[0] == '\0' ||
        !cache->namespace_tag || cache->namespace_tag[0] == '\0' || !cuda_source ||
        !out_path || out_path_size == 0u || !out_key || out_key_size == 0u) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "invalid module cache arguments");
        return 0;
    }

    (void)snprintf(device_key, sizeof(device_key), "device=unknown");
    device_key[sizeof(device_key) - 1u] = '\0';

    h1 = cxcu_fnv1a_update_string(h1, cache->namespace_tag);
    h1 = cxcu_fnv1a_update_string(h1, device_key);
    h1 = cxcu_fnv1a_update_string(h1, cuda_source);
    h1 = cxcu_fnv1a_update(h1, &option_count, sizeof(option_count));
    h2 = cxcu_fnv1a_update_string_with_suffix(h2, cache->namespace_tag, "b");
    h2 = cxcu_fnv1a_update_string(h2, device_key);
    h2 = cxcu_fnv1a_update_string(h2, cuda_source);
    h2 = cxcu_fnv1a_update(h2, &option_count, sizeof(option_count));
    for (i = 0u; i < option_count; ++i) {
        h1 = cxcu_fnv1a_update_string(h1, options ? options[i] : "");
        h2 = cxcu_fnv1a_update_string(h2, options ? options[i] : "");
    }

    written = snprintf(
        out_key,
        out_key_size,
        "%016llx%016llx",
        (unsigned long long)h1,
        (unsigned long long)h2);
    if (written <= 0 || (size_t)written >= out_key_size) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "module cache key buffer is too small");
        return 0;
    }
    prefix = cache->file_prefix ? cache->file_prefix : "";
    extension = cache->file_extension ? cache->file_extension : "";
    written = snprintf(out_path, out_path_size, "%s/%s%s%s", cache->directory, prefix, out_key, extension);
    if (written <= 0 || (size_t)written >= out_path_size) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "module cache path buffer is too small");
        return 0;
    }
    cxcu_error_clear(err);
    return 1;
}

int cxcu_module_cache_read_image(const char* path, cxcu_module_image* out_image) {
    FILE* in;
    long size;
    void* data = NULL;
    size_t nread;

    if (out_image) memset(out_image, 0, sizeof(*out_image));
    if (!path || !out_image) return 0;
    in = fopen(path, "rb");
    if (!in) return 0;
    if (fseek(in, 0, SEEK_END) != 0) {
        fclose(in);
        return 0;
    }
    size = ftell(in);
    if (size <= 0 || fseek(in, 0, SEEK_SET) != 0) {
        fclose(in);
        return 0;
    }
    data = malloc((size_t)size);
    if (!data) {
        fclose(in);
        return 0;
    }
    nread = fread(data, 1u, (size_t)size, in);
    fclose(in);
    if (nread != (size_t)size) {
        free(data);
        return 0;
    }
    out_image->data = data;
    out_image->size = (size_t)size;
    out_image->kind = CXCU_MODULE_IMAGE_CUBIN;
    (void)snprintf(out_image->architecture, sizeof(out_image->architecture), "%s", "cache");
    out_image->architecture[sizeof(out_image->architecture) - 1u] = '\0';
    return 1;
}

int cxcu_module_cache_write_image(const char* path, const void* data, size_t size) {
    char tmp_path[PATH_MAX];
    FILE* out;
    const char* slash;
    char dir[PATH_MAX];
    size_t dir_len;
    int written;
    int ok = 0;

    if (!path || !data || size == 0u) return 0;
    slash = strrchr(path, '/');
    if (slash) {
        dir_len = (size_t)(slash - path);
        if (dir_len == 0u || dir_len >= sizeof(dir)) return 0;
        memcpy(dir, path, dir_len);
        dir[dir_len] = '\0';
        if (!cxcu_mkdir_p(dir)) return 0;
    }
    written = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (written <= 0 || (size_t)written >= sizeof(tmp_path)) return 0;

    out = fopen(tmp_path, "wb");
    if (!out) return 0;
    if (fwrite(data, 1u, size, out) == size && fflush(out) == 0) ok = 1;
    if (fclose(out) != 0) ok = 0;
    if (!ok) {
        (void)remove(tmp_path);
        return 0;
    }
    if (rename(tmp_path, path) != 0) {
        (void)remove(tmp_path);
        return 0;
    }
    return 1;
}

int cxcu_module_cache_image_exists(
    const cxcu_module_cache* cache,
    const char* cuda_source,
    const char* const* options,
    size_t option_count) {
    char path[PATH_MAX];
    char key[65];
    if (!cxcu_module_cache_path(
            cache,
            cuda_source,
            options,
            option_count,
            path,
            sizeof(path),
            key,
            sizeof(key),
            NULL)) {
        return 0;
    }
    return cxcu_file_exists(path);
}

int cxcu_compile_module_image_cached(
    const cxcu_module_cache* cache,
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    cxcu_module_image* out_image,
    cxcu_error* err) {
    char path[PATH_MAX];
    char key[65];
    int have_path = 0;

    (void)program_name;
    if (!cuda_source || !out_image) {
        cxcu_set_error(err, CXCU_STATUS_INVALID_ARGUMENT, 0, "CUDA source and out_image are required");
        return 0;
    }
    memset(out_image, 0, sizeof(*out_image));
    if (cache && cache->enabled) {
        have_path = cxcu_module_cache_path(
            cache,
            cuda_source,
            options,
            option_count,
            path,
            sizeof(path),
            key,
            sizeof(key),
            NULL);
        if (have_path && cxcu_module_cache_read_image(path, out_image)) {
            cxcu_error_clear(err);
            return 1;
        }
    }
    return cxcu_stub_unavailable(err);
}

void cxcu_ptx_free(cxcu_ptx* ptx) {
    if (ptx) memset(ptx, 0, sizeof(*ptx));
}

void cxcu_module_image_free(cxcu_module_image* image) {
    if (image) memset(image, 0, sizeof(*image));
}

int cxcu_launch(
    cxcu_module* module,
    const char* kernel_name,
    const cxcu_launch_config* cfg,
    void** args,
    cxcu_error* err) {
    (void)module;
    (void)kernel_name;
    (void)cfg;
    (void)args;
    return cxcu_stub_unavailable(err);
}

int cxcu_synchronize(cxcu_error* err) {
    return cxcu_stub_unavailable(err);
}

void cxcu_shutdown(void) {
}

#endif
