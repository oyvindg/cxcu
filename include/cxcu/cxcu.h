#ifndef CXCU_CXCU_H
#define CXCU_CXCU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file cxcu.h
 * @brief Small C wrapper around the CUDA Driver API.
 *
 * The API stores CUDA handles as integer-sized opaque values so callers do not
 * need to include CUDA headers directly. Functions returning int use 1 for
 * success and 0 for failure. When an error pointer is supplied, failure details
 * are written to it and successful calls clear it.
 */

/** @brief Major version of the cxcu API. */
#define CXCU_VERSION_MAJOR 0
/** @brief Minor version of the cxcu API. */
#define CXCU_VERSION_MINOR 1
/** @brief Patch version of the cxcu API. */
#define CXCU_VERSION_PATCH 0

/** @brief Status codes reported by cxcu operations. */
typedef enum cxcu_status {
    /** Operation completed successfully. */
    CXCU_STATUS_OK = 0,
    /** A required argument was missing or invalid. */
    CXCU_STATUS_INVALID_ARGUMENT,
    /** CUDA support, the CUDA driver, or a CUDA device is unavailable. */
    CXCU_STATUS_UNAVAILABLE,
    /** A CUDA Driver API call failed. */
    CXCU_STATUS_CUDA_ERROR,
    /** Device memory allocation failed. */
    CXCU_STATUS_OUT_OF_MEMORY,
    /** Runtime CUDA source compilation failed. */
    CXCU_STATUS_COMPILE_ERROR
} cxcu_status;

/** @brief Detailed error information for failed cxcu operations. */
typedef struct cxcu_error {
    /** cxcu status classification. */
    cxcu_status status;
    /** Raw CUDA error code when available, otherwise 0. */
    int cuda_code;
    /** Human-readable, null-terminated error message. */
    char message[256];
} cxcu_error;

/** @brief Basic information about a CUDA device. */
typedef struct cxcu_device_info {
    /** CUDA device ordinal. */
    int ordinal;
    /** Compute capability major version. */
    int compute_capability_major;
    /** Compute capability minor version. */
    int compute_capability_minor;
    /** Total device memory in bytes. */
    size_t total_memory;
    /** Null-terminated CUDA device name. */
    char name[256];
} cxcu_device_info;

/** @brief Opaque CUDA device memory allocation. */
typedef struct cxcu_buffer {
    /** Opaque CUDA device pointer value. */
    uintptr_t device_ptr;
    /** Allocation size in bytes. */
    size_t bytes;
} cxcu_buffer;

/** @brief Opaque loaded CUDA module. */
typedef struct cxcu_module {
    /** Opaque CUDA module handle value. */
    uintptr_t handle;
} cxcu_module;

/** @brief Owned PTX image produced by NVRTC. */
typedef struct cxcu_ptx {
    /** Null-terminated PTX text. */
    char* data;
    /** Size in bytes, including the trailing null byte when present. */
    size_t size;
} cxcu_ptx;

/** @brief Type of module image produced by NVRTC. */
typedef enum cxcu_module_image_kind {
    /** No image is present. */
    CXCU_MODULE_IMAGE_NONE = 0,
    /** PTX text image. */
    CXCU_MODULE_IMAGE_PTX,
    /** Device-specific CUBIN binary image. */
    CXCU_MODULE_IMAGE_CUBIN
} cxcu_module_image_kind;

/** @brief Owned loadable module image produced by NVRTC. */
typedef struct cxcu_module_image {
    /** Module image bytes. PTX is text; CUBIN is binary. */
    void* data;
    /** Size in bytes. */
    size_t size;
    /** Image kind. */
    cxcu_module_image_kind kind;
    /** NVRTC architecture option used for the image, if known. */
    char architecture[32];
} cxcu_module_image;

/** @brief Opaque CUDA stream. */
typedef struct cxcu_stream {
    /** Opaque CUDA stream handle value. */
    uintptr_t handle;
} cxcu_stream;

/** @brief Kernel launch dimensions and optional stream. */
typedef struct cxcu_launch_config {
    /** Grid size in the X dimension. Must be greater than 0. */
    unsigned int grid_x;
    /** Grid size in the Y dimension. Must be greater than 0. */
    unsigned int grid_y;
    /** Grid size in the Z dimension. Must be greater than 0. */
    unsigned int grid_z;
    /** Block size in the X dimension. Must be greater than 0. */
    unsigned int block_x;
    /** Block size in the Y dimension. Must be greater than 0. */
    unsigned int block_y;
    /** Block size in the Z dimension. Must be greater than 0. */
    unsigned int block_z;
    /** Dynamic shared memory size in bytes. */
    unsigned int shared_mem_bytes;
    /** Optional stream for the launch. NULL uses the default stream. */
    const cxcu_stream* stream;
} cxcu_launch_config;

/**
 * @brief Reset an error object to CXCU_STATUS_OK.
 * @param err Error object to reset. May be NULL.
 */
void cxcu_error_clear(cxcu_error* err);

/**
 * @brief Return a stable string name for a status code.
 * @param status Status code to name.
 * @return String literal for the status, or "unknown" for unrecognized values.
 */
const char* cxcu_status_name(cxcu_status status);

/**
 * @brief Check whether CUDA Driver API support and at least one device are available.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 when CUDA is available, otherwise 0.
 */
int cxcu_available(cxcu_error* err);

/**
 * @brief Query the number of CUDA devices.
 * @param out_count Receives the device count. Required.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_device_count(int* out_count, cxcu_error* err);

/**
 * @brief Query information for one CUDA device.
 * @param ordinal CUDA device ordinal to query.
 * @param out_info Receives device information. Required.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_device_info_get(int ordinal, cxcu_device_info* out_info, cxcu_error* err);

/**
 * @brief Create a CUDA stream in the current cxcu context.
 * @param out_stream Receives the stream handle. Required.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_stream_create(cxcu_stream* out_stream, cxcu_error* err);

/**
 * @brief Destroy a CUDA stream and clear the handle.
 * @param stream Stream to destroy. May be NULL.
 */
void cxcu_stream_destroy(cxcu_stream* stream);

/**
 * @brief Allocate CUDA device memory.
 * @param out_buffer Receives the device allocation. Required.
 * @param bytes Allocation size in bytes. Must be greater than 0.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_buffer_alloc(cxcu_buffer* out_buffer, size_t bytes, cxcu_error* err);

/**
 * @brief Free CUDA device memory and clear the buffer.
 * @param buffer Buffer to free. May be NULL.
 */
void cxcu_buffer_free(cxcu_buffer* buffer);

/**
 * @brief Copy bytes from host memory to a CUDA device buffer.
 * @param dst Destination device buffer. Required.
 * @param src Source host memory. Required.
 * @param bytes Number of bytes to copy. Must be greater than 0 and no larger than dst->bytes.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_memcpy_h2d(cxcu_buffer* dst, const void* src, size_t bytes, cxcu_error* err);

/**
 * @brief Copy bytes from a CUDA device buffer to host memory.
 * @param dst Destination host memory. Required.
 * @param src Source device buffer. Required.
 * @param bytes Number of bytes to copy. Must be greater than 0 and no larger than src->bytes.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_memcpy_d2h(void* dst, const cxcu_buffer* src, size_t bytes, cxcu_error* err);

/**
 * @brief Load a CUDA module from an in-memory image such as PTX or cubin data.
 * @param out_module Receives the loaded module handle. Required.
 * @param image Module image bytes. Required.
 * @param image_size Size of image in bytes. Must be greater than 0.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_module_load_data(cxcu_module* out_module, const void* image, size_t image_size, cxcu_error* err);

/**
 * @brief Compile CUDA C source to PTX with NVRTC.
 * @param cuda_source Null-terminated CUDA C source. Required.
 * @param program_name Optional diagnostic program name.
 * @param options Optional NVRTC option strings, such as "--gpu-architecture=compute_50".
 * @param option_count Number of option strings.
 * @param out_ptx Receives owned PTX data. Release with cxcu_ptx_free().
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_compile_ptx(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    cxcu_ptx* out_ptx,
    cxcu_error* err);

/**
 * @brief Compile CUDA C source to a module image loadable by cxcu_module_load_data().
 *
 * The implementation prefers a device-specific CUBIN for @p device_ordinal,
 * because it avoids driver-side PTX JIT and is more tolerant of newer NVRTC
 * toolchains paired with older drivers. If CUBIN is unavailable, it falls back
 * to PTX using the supplied options.
 *
 * Any caller-supplied `--gpu-architecture` or `-arch` option is replaced for
 * the CUBIN attempt. The original options are preserved for the PTX fallback.
 *
 * @param cuda_source Null-terminated CUDA C source. Required.
 * @param program_name Optional diagnostic program name.
 * @param options Optional NVRTC option strings.
 * @param option_count Number of option strings.
 * @param device_ordinal CUDA device ordinal used to select `sm_XX`.
 * @param out_image Receives owned image data. Release with cxcu_module_image_free().
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_compile_module_image_for_device(
    const char* cuda_source,
    const char* program_name,
    const char* const* options,
    size_t option_count,
    int device_ordinal,
    cxcu_module_image* out_image,
    cxcu_error* err);

/**
 * @brief Release PTX data produced by cxcu_compile_ptx().
 * @param ptx PTX image to clear. May be NULL.
 */
void cxcu_ptx_free(cxcu_ptx* ptx);

/**
 * @brief Release image data produced by cxcu_compile_module_image_for_device().
 * @param image Module image to clear. May be NULL.
 */
void cxcu_module_image_free(cxcu_module_image* image);

/**
 * @brief Unload a CUDA module and clear the handle.
 * @param module Module to unload. May be NULL.
 */
void cxcu_module_unload(cxcu_module* module);

/**
 * @brief Launch a kernel from a loaded CUDA module.
 * @param module Loaded CUDA module. Required.
 * @param kernel_name Name of the kernel function in the module. Required.
 * @param cfg Launch configuration with non-zero grid and block dimensions. Required.
 * @param args Kernel argument pointer array passed to cuLaunchKernel. May be NULL for kernels with no parameters.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_launch(cxcu_module* module, const char* kernel_name, const cxcu_launch_config* cfg, void** args, cxcu_error* err);

/**
 * @brief Synchronize the current cxcu CUDA context.
 * @param err Optional error object populated on failure and cleared on success.
 * @return 1 on success, otherwise 0.
 */
int cxcu_synchronize(cxcu_error* err);

/**
 * @brief Release the retained CUDA primary context used by cxcu.
 */
void cxcu_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* CXCU_CXCU_H */
