#ifndef CXCU_DEVICE_VIEW_CUH
#define CXCU_DEVICE_VIEW_CUH

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define CXCU_DEVICE_INLINE static __device__ __forceinline__
#else
#define CXCU_DEVICE_INLINE static inline
#endif

typedef struct cxcu_device_view_kernel_descriptor {
    unsigned long long device_ptr;
    unsigned long long generation;
    unsigned long long shape[4];
    unsigned long long element_strides[4];
    unsigned int element_type;
    unsigned int rank;
    unsigned int flags;
    unsigned int reserved;
} cxcu_device_view_kernel_descriptor;

typedef struct cxcu_device_view_kernel_status {
    int code;
    unsigned int slot;
    unsigned int dimension;
    unsigned long long index;
    unsigned long long bound;
} cxcu_device_view_kernel_status;

CXCU_DEVICE_INLINE int cxcu_device_view_offset_checked(
    const cxcu_device_view_kernel_descriptor* view,
    const unsigned long long* indices,
    unsigned int slot,
    unsigned long long* out_offset,
    cxcu_device_view_kernel_status* status) {
    unsigned long long offset = 0;
    unsigned int dimension;
    if (!view || !indices || !out_offset || view->rank == 0 || view->rank > 4) return 0;
    for (dimension = 0; dimension < view->rank; ++dimension) {
        if (indices[dimension] >= view->shape[dimension]) {
#if defined(__CUDA_ARCH__)
            if (status && atomicCAS(&status->code, 0, 1) == 0) {
#else
            if (status && status->code == 0 && (status->code = 1)) {
#endif
                status->slot = slot;
                status->dimension = dimension;
                status->index = indices[dimension];
                status->bound = view->shape[dimension];
            }
            return 0;
        }
        offset += indices[dimension] * view->element_strides[dimension];
    }
    *out_offset = offset;
    return 1;
}

CXCU_DEVICE_INLINE double cxcu_device_view_number_or(
    const cxcu_device_view_kernel_descriptor* view,
    const unsigned long long* indices,
    unsigned int slot,
    double fallback,
    cxcu_device_view_kernel_status* status) {
    unsigned long long offset;
    const double* values;
    if (!cxcu_device_view_offset_checked(view, indices, slot, &offset, status)) return fallback;
    values = (const double*)view->device_ptr;
    return values[offset];
}

CXCU_DEVICE_INLINE double cxcu_device_view_number_clamped(
    const cxcu_device_view_kernel_descriptor* view,
    const unsigned long long* indices) {
    unsigned long long clamped[4] = {0, 0, 0, 0};
    unsigned long long offset = 0;
    unsigned int dimension;
    const double* values;
    if (!view || !indices || view->rank == 0 || view->rank > 4) return 0.0;
    for (dimension = 0; dimension < view->rank; ++dimension) {
        clamped[dimension] = indices[dimension] < view->shape[dimension]
            ? indices[dimension] : view->shape[dimension] - 1u;
        offset += clamped[dimension] * view->element_strides[dimension];
    }
    values = (const double*)view->device_ptr;
    return values[offset];
}

CXCU_DEVICE_INLINE unsigned int cxcu_device_view_uint32_at(
    const cxcu_device_view_kernel_descriptor* view,
    unsigned long long index) {
    const unsigned int* values = (const unsigned int*)view->device_ptr;
    return values[index * view->element_strides[0]];
}

#undef CXCU_DEVICE_INLINE
#endif
