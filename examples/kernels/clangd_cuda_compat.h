/*
 * Tooling-only CUDA shims for clangd.
 *
 * cxcu examples load .cu files through NVRTC at runtime, so they are not normal
 * CMake build units. .clangd includes this header so editor parsing works
 * without requiring CUDA runtime headers.
 */
#ifndef CXCU_EXAMPLE_CLANGD_CUDA_COMPAT_H
#define CXCU_EXAMPLE_CLANGD_CUDA_COMPAT_H

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __constant__
#define __constant__
#endif

#ifndef __managed__
#define __managed__
#endif

#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

typedef struct cxcu_clangd_cuda_dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} cxcu_clangd_cuda_dim3;

extern const cxcu_clangd_cuda_dim3 threadIdx;
extern const cxcu_clangd_cuda_dim3 blockIdx;
extern const cxcu_clangd_cuda_dim3 blockDim;
extern const cxcu_clangd_cuda_dim3 gridDim;

#endif /* CXCU_EXAMPLE_CLANGD_CUDA_COMPAT_H */
