#include <cxcu/cxcu.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CXCU_EXAMPLE_KERNEL_PATH
#define CXCU_EXAMPLE_KERNEL_PATH "kernels/vector_scale.cu"
#endif

static int read_text_file(const char* path, char** out_data, size_t* out_size) {
    FILE* file = NULL;
    char* data = NULL;
    long size = 0;
    size_t read_size = 0u;

    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0u;
    if (!path || !out_data) return 0;

    file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0) goto fail;
    size = ftell(file);
    if (size < 0) goto fail;
    if (fseek(file, 0, SEEK_SET) != 0) goto fail;

    data = (char*)malloc((size_t)size + 1u);
    if (!data) goto fail;
    read_size = fread(data, 1u, (size_t)size, file);
    if (read_size != (size_t)size) goto fail;
    data[read_size] = '\0';

    if (out_size) *out_size = read_size + 1u;
    *out_data = data;
    fclose(file);
    return 1;

fail:
    free(data);
    fclose(file);
    return 0;
}

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

static char* make_include_option(const char* kernel_path) {
    static const char prefix[] = "--include-path=";
    const char* slash = NULL;
    const char* backslash = NULL;
    const char* end = NULL;
    size_t prefix_len = sizeof(prefix) - 1u;
    size_t dir_len = 1u;
    char* option = NULL;

    if (!kernel_path || kernel_path[0] == '\0') kernel_path = ".";
    slash = strrchr(kernel_path, '/');
    backslash = strrchr(kernel_path, '\\');
    if (slash && backslash) {
        end = slash > backslash ? slash : backslash;
    } else {
        end = slash ? slash : backslash;
    }
    if (end) {
        dir_len = (size_t)(end - kernel_path);
        if (dir_len == 0u) dir_len = 1u;
    }

    option = (char*)malloc(prefix_len + dir_len + 1u);
    if (!option) return NULL;
    memcpy(option, prefix, prefix_len);
    if (end) {
        memcpy(option + prefix_len, kernel_path, dir_len);
    } else {
        option[prefix_len] = '.';
    }
    option[prefix_len + dir_len] = '\0';
    return option;
}

int main(int argc, char** argv) {
    const char* compile_options[4];
    size_t compile_option_count = 0u;
    enum { count = 16 };
    const char* kernel_path = argc > 1 ? argv[1] : CXCU_EXAMPLE_KERNEL_PATH;
    const float scale = 1.75f;
    const float bias = -2.0f;
    cxcu_error err = {0};
    cxcu_module_image image = {0};
    cxcu_module module = {0};
    cxcu_buffer input_buffer = {0};
    cxcu_buffer output_buffer = {0};
    cxcu_launch_config cfg;
    char* kernel_source = NULL;
    char* include_option = NULL;
    float input[count];
    float output[count];
    uint64_t input_ptr = 0u;
    uint64_t output_ptr = 0u;
    uint32_t count_arg = count;
    float scale_arg = scale;
    float bias_arg = bias;
    void* args[5];
    int exit_code = 1;

    if (!cxcu_available(&err)) {
        printf("CUDA unavailable: %s\n", err.message);
        return 0;
    }
    if (!read_text_file(kernel_path, &kernel_source, NULL)) {
        fprintf(stderr, "failed to read CUDA source: %s\n", kernel_path);
        return 1;
    }
    include_option = make_include_option(kernel_path);
    if (!include_option) {
        fprintf(stderr, "failed to allocate NVRTC include option\n");
        goto cleanup;
    }

    compile_options[compile_option_count++] = "--std=c++11";
    compile_options[compile_option_count++] = "--gpu-architecture=compute_50";
    compile_options[compile_option_count++] = "--fmad=false";
    compile_options[compile_option_count++] = include_option;

    if (!cxcu_compile_module_image_for_device(
            kernel_source,
            "vector_scale.cu",
            compile_options,
            compile_option_count,
            0,
            &image,
            &err)) {
        fprintf(
            err.status == CXCU_STATUS_UNAVAILABLE ? stdout : stderr,
            "runtime CUDA compilation unavailable: %s\n",
            err.message);
        exit_code = err.status == CXCU_STATUS_UNAVAILABLE ? 0 : 1;
        goto cleanup;
    }
    if (!cxcu_module_load_data(&module, image.data, image.size, &err)) {
        fprintf(stderr, "failed to load CUDA image: %s\n", err.message);
        goto cleanup;
    }

    for (uint32_t i = 0u; i < count; ++i) {
        input[i] = (float)i + 0.5f;
        output[i] = 0.0f;
    }

    if (!cxcu_buffer_alloc(&input_buffer, sizeof(input), &err)) goto cuda_fail;
    if (!cxcu_buffer_alloc(&output_buffer, sizeof(output), &err)) goto cuda_fail;
    if (!cxcu_memcpy_h2d(&input_buffer, input, sizeof(input), &err)) goto cuda_fail;

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid_x = 1u;
    cfg.grid_y = 1u;
    cfg.grid_z = 1u;
    cfg.block_x = count;
    cfg.block_y = 1u;
    cfg.block_z = 1u;

    input_ptr = (uint64_t)input_buffer.device_ptr;
    output_ptr = (uint64_t)output_buffer.device_ptr;
    args[0] = &input_ptr;
    args[1] = &output_ptr;
    args[2] = &count_arg;
    args[3] = &scale_arg;
    args[4] = &bias_arg;

    if (!cxcu_launch(&module, "cxcu_vector_scale", &cfg, args, &err)) goto cuda_fail;
    if (!cxcu_synchronize(&err)) goto cuda_fail;
    if (!cxcu_memcpy_d2h(output, &output_buffer, sizeof(output), &err)) goto cuda_fail;

    for (uint32_t i = 0u; i < count; ++i) {
        const float expected = input[i] * scale + bias;
        if (abs_f32(output[i] - expected) > 0.0001f) {
            fprintf(
                stderr,
                "mismatch at %u: got %.6f, expected %.6f\n",
                i,
                output[i],
                expected);
            goto cleanup;
        }
    }

    printf(
        "CUDA .cu example verified: count=%u, scale=%.2f, bias=%.2f, first=%.3f, last=%.3f\n",
        count_arg,
        scale_arg,
        bias_arg,
        output[0],
        output[count - 1u]);
    exit_code = 0;
    goto cleanup;

cuda_fail:
    fprintf(stderr, "CUDA vector scale failed: %s\n", err.message);

cleanup:
    cxcu_buffer_free(&output_buffer);
    cxcu_buffer_free(&input_buffer);
    cxcu_module_unload(&module);
    cxcu_module_image_free(&image);
    cxcu_shutdown();
    free(include_option);
    free(kernel_source);
    return exit_code;
}
