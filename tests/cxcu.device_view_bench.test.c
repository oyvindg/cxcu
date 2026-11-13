#include <cxcu/cxcu.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static const char* kernel_source =
    "#include <cxcu/device_view.cuh>\n"
    "extern \"C\" __global__ void view_scale(const cxcu_device_view_kernel_descriptor* views, unsigned long long count) {\n"
    "  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;\n"
    "  if(i>=count)return;\n"
    "  unsigned long long index[1]={i};\n"
    "  double* out=(double*)views[1].device_ptr;\n"
    "  out[i*views[1].element_strides[0]]=cxcu_device_view_number_or(&views[0],index,0,0.0,0)*2.0;\n"
    "}\n";

static int bind_views(
    cxcu_device_view_plan* plan,
    const double* input,
    size_t count,
    uint64_t generation,
    cxcu_device_view_descriptor descriptors[2],
    cxcu_error* err) {
    const cxcu_device_view_binding bindings[2] = {
        {.host_data = input, .element_type = CXCU_DEVICE_VIEW_NUMBER, .rank = 1u,
         .shape = {count}, .capacity_bytes = count * sizeof(double), .generation = generation,
         .flags = CXCU_DEVICE_VIEW_READ_ONLY | CXCU_DEVICE_VIEW_CONTIGUOUS},
        {.host_data = NULL, .element_type = CXCU_DEVICE_VIEW_NUMBER, .rank = 1u,
         .shape = {count}, .capacity_bytes = count * sizeof(double), .generation = generation,
         .flags = CXCU_DEVICE_VIEW_WRITABLE | CXCU_DEVICE_VIEW_CONTIGUOUS},
    };
    return cxcu_device_view_plan_bind(plan, bindings, 2u, descriptors, err);
}

static int run_kernel(
    cxcu_module* module,
    const cxcu_device_view_descriptor descriptors[2],
    size_t count,
    cxcu_buffer* descriptor_buffer,
    cxcu_error* err) {
    uint64_t descriptor_ptr;
    uint64_t count_arg = count;
    void* args[2];
    cxcu_launch_config cfg = {
        (unsigned int)((count + 255u) / 256u), 1u, 1u, 256u, 1u, 1u, 0u, NULL};
    if (!descriptor_buffer->device_ptr) {
        if (!cxcu_buffer_alloc_upload(
                descriptor_buffer, descriptors, 2u * sizeof(descriptors[0]), err)) return 0;
    } else if (!cxcu_memcpy_h2d(
                   descriptor_buffer, descriptors, 2u * sizeof(descriptors[0]), err)) return 0;
    descriptor_ptr = descriptor_buffer->device_ptr;
    args[0] = &descriptor_ptr;
    args[1] = &count_arg;
    return cxcu_launch(module, "view_scale", &cfg, args, err) && cxcu_synchronize(err);
}

int main(int argc, char** argv) {
    const size_t count = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 1048576u;
    const size_t iterations = argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : 7u;
    cxcu_device_view_capability capability = {0};
    cxcu_module_image image = {0};
    cxcu_module module = {0};
    cxcu_error err = {0};
    double* input = NULL;
    double* output = NULL;
    double cpu_ms = 0.0;
    double transient_ms = 0.0;
    double resident_cold_ms = 0.0;
    double resident_warm_ms = 0.0;
    double started;
    size_t i;
    size_t run;

    assert(count > 0u && iterations > 1u);
    assert(cxcu_device_view_capability_get(0, &capability, &err));
    if (!capability.device_available) {
        cxcu_error probe = {0};
        (void)cxcu_available(&probe);
        printf("{\"schema\":\"cxcu.device-view-benchmark.v1\",\"gpu\":\"skipped\","
               "\"status\":\"%s\",\"cudaCode\":%d,\"reason\":\"%s\"}\n",
               cxcu_status_name(probe.status), probe.cuda_code,
               probe.message[0] ? probe.message : "CUDA device unavailable");
        return 0;
    }
    input = (double*)malloc(count * sizeof(*input));
    output = (double*)malloc(count * sizeof(*output));
    assert(input && output);
    for (i = 0u; i < count; ++i) input[i] = (double)(i % 1009u) / 17.0;

    started = now_ms();
    for (run = 0u; run < iterations; ++run)
        for (i = 0u; i < count; ++i) output[i] = input[i] * 2.0;
    cpu_ms = (now_ms() - started) / (double)iterations;

    {
        const char* options[] = {CXCU_TEST_INCLUDE_OPTION};
        assert(cxcu_compile_module_image_for_device(
            kernel_source, "cxcu_device_view_bench.cu", options, 1u, 0, &image, &err));
    }
    assert(cxcu_module_load_data(&module, image.data, image.size, &err));

    started = now_ms();
    for (run = 0u; run < iterations; ++run) {
        cxcu_device_view_plan plan;
        cxcu_device_view_descriptor descriptors[2];
        cxcu_buffer descriptor_buffer = {0};
        cxcu_device_view_plan_init(&plan, 0);
        assert(bind_views(&plan, input, count, run + 1u, descriptors, &err));
        assert(run_kernel(&module, descriptors, count, &descriptor_buffer, &err));
        assert(cxcu_memcpy_d2h(output, &plan.entries[1].buffer, count * sizeof(*output), &err));
        cxcu_buffer_free(&descriptor_buffer);
        cxcu_device_view_plan_free(&plan);
    }
    transient_ms = (now_ms() - started) / (double)iterations;

    {
        cxcu_device_view_plan plan;
        cxcu_device_view_descriptor descriptors[2];
        cxcu_buffer descriptor_buffer = {0};
        cxcu_device_view_plan_init(&plan, 0);
        started = now_ms();
        assert(bind_views(&plan, input, count, 1u, descriptors, &err));
        assert(run_kernel(&module, descriptors, count, &descriptor_buffer, &err));
        assert(cxcu_memcpy_d2h(output, &plan.entries[1].buffer, count * sizeof(*output), &err));
        resident_cold_ms = now_ms() - started;
        started = now_ms();
        for (run = 1u; run < iterations; ++run) {
            assert(bind_views(&plan, input, count, 1u, descriptors, &err));
            assert(run_kernel(&module, descriptors, count, &descriptor_buffer, &err));
            assert(cxcu_memcpy_d2h(output, &plan.entries[1].buffer, count * sizeof(*output), &err));
        }
        resident_warm_ms = (now_ms() - started) / (double)(iterations - 1u);
        assert(plan.upload_count == 2u);
        assert(plan.reuse_count == 2u * (iterations - 1u));
        cxcu_buffer_free(&descriptor_buffer);
        cxcu_device_view_plan_free(&plan);
    }
    for (i = 0u; i < count; ++i) assert(fabs(output[i] - input[i] * 2.0) < 1e-12);
    printf("{\"schema\":\"cxcu.device-view-benchmark.v1\",\"elements\":%zu,"
           "\"iterations\":%zu,\"cpuMs\":%.6f,\"gpuTransientMs\":%.6f,"
           "\"gpuResidentColdMs\":%.6f,\"gpuResidentWarmMs\":%.6f,"
           "\"parity\":true}\n",
           count, iterations, cpu_ms, transient_ms, resident_cold_ms, resident_warm_ms);
    cxcu_module_unload(&module);
    cxcu_module_image_free(&image);
    cxcu_shutdown();
    free(input);
    free(output);
    return 0;
}
