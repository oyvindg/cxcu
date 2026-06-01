/*
 * Standalone CUDA kernel source loaded by the C example at runtime.
 *
 * This file is intentionally host agnostic. The C host code owns allocation,
 * module loading, launch dimensions, synchronization, and CPU verification.
 */

#include "vector_math.h"

extern "C" __global__ void cxcu_vector_scale(
    const float* input,
    float* output,
    unsigned int count,
    float scale,
    float bias) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = cxcu_example_scale_bias(input[index], scale, bias);
}
