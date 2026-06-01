#ifndef CXCU_EXAMPLE_VECTOR_MATH_H
#define CXCU_EXAMPLE_VECTOR_MATH_H

static __device__ float cxcu_example_scale_bias(float value, float scale, float bias) {
    return value * scale + bias;
}

#endif /* CXCU_EXAMPLE_VECTOR_MATH_H */
