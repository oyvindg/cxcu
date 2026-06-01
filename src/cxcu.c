#include <cxcu/cxcu.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void cxcu_buffer_free(cxcu_buffer* buffer) {
    if (!buffer || !buffer->device_ptr) return;
    if (cxcu_ensure_context(NULL)) {
        (void)cuMemFree((CUdeviceptr)buffer->device_ptr);
    }
    buffer->device_ptr = 0u;
    buffer->bytes = 0u;
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
