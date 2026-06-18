#include <cxcu/cxcu.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_status_names(void) {
    assert(strcmp(cxcu_status_name(CXCU_STATUS_OK), "ok") == 0);
    assert(strcmp(cxcu_status_name(CXCU_STATUS_INVALID_ARGUMENT), "invalid_argument") == 0);
    assert(strcmp(cxcu_status_name(CXCU_STATUS_UNAVAILABLE), "unavailable") == 0);
    assert(strcmp(cxcu_status_name(CXCU_STATUS_CUDA_ERROR), "cuda_error") == 0);
    assert(strcmp(cxcu_status_name(CXCU_STATUS_OUT_OF_MEMORY), "out_of_memory") == 0);
    assert(strcmp(cxcu_status_name(CXCU_STATUS_COMPILE_ERROR), "compile_error") == 0);
    printf("  ok test_status_names\n");
}

static void test_error_clear(void) {
    cxcu_error err;
    err.status = CXCU_STATUS_CUDA_ERROR;
    err.cuda_code = 123;
    strcpy(err.message, "example");

    cxcu_error_clear(&err);
    assert(err.status == CXCU_STATUS_OK);
    assert(err.cuda_code == 0);
    assert(err.message[0] == '\0');
    printf("  ok test_error_clear\n");
}

static void test_error_format(void) {
    cxcu_error err;
    char buf[64];

    cxcu_error_clear(&err);
    cxcu_error_format("prefix", &err, buf, sizeof(buf));
    assert(strcmp(buf, "prefix") == 0);
    strcpy(err.message, "detail");
    cxcu_error_format("prefix", &err, buf, sizeof(buf));
    assert(strcmp(buf, "prefix: detail") == 0);
    cxcu_error_format("prefix", &err, buf, 8u);
    assert(strcmp(buf, "prefix:") == 0);
    printf("  ok test_error_format\n");
}

static void test_source_concat(void) {
    static const unsigned char one[] = "alpha";
    static const unsigned char two[] = {'b', 'e', 't', 'a'};
    const unsigned char* sources[2];
    size_t sizes[2];
    char* out;

    assert(cxcu_source_concat(NULL, NULL, 0u) == NULL);
    sources[0] = one;
    sizes[0] = sizeof(one);
    sources[1] = two;
    sizes[1] = sizeof(two);
    out = cxcu_source_concat(sources, sizes, 2u);
    assert(out != NULL);
    assert(strcmp(out, "alpha\nbeta\n") == 0);
    free(out);
    printf("  ok test_source_concat\n");
}

static void test_module_cache_path_and_file_io(void) {
    cxcu_module_cache cache;
    char path_a[512];
    char path_b[512];
    char key_a[65];
    char key_b[65];
    cxcu_error err = {0};
    cxcu_module_image image = {0};
    const char payload[] = "cached image";
    FILE* out;

    memset(&cache, 0, sizeof(cache));
    cache.enabled = 1;
    cache.directory = "/tmp/cxcu_cache_test";
    cache.namespace_tag = "cxcu-test-cache-v1";
    cache.file_prefix = "unit_";
    cache.file_extension = ".cubin";
    cache.device_ordinal = 0;

    assert(cxcu_module_cache_path(&cache, "source", NULL, 0u, path_a, sizeof(path_a), key_a, sizeof(key_a), &err));
    assert(strstr(path_a, "/tmp/cxcu_cache_test/unit_") == path_a);
    assert(strstr(path_a, ".cubin") != NULL);
    {
        static const char* options[] = {"--std=c++11"};
        assert(cxcu_module_cache_path(&cache, "source", options, 1u, path_b, sizeof(path_b), key_b, sizeof(key_b), &err));
    }
    assert(strcmp(key_a, key_b) != 0);
    cache.namespace_tag = "cxcu-test-cache-v2";
    assert(cxcu_module_cache_path(&cache, "source", NULL, 0u, path_b, sizeof(path_b), key_b, sizeof(key_b), &err));
    assert(strcmp(key_a, key_b) != 0);

    cache.namespace_tag = "cxcu-test-cache-v1";
    assert(cxcu_module_cache_write_image(path_a, payload, sizeof(payload)));
    assert(cxcu_module_cache_read_image(path_a, &image));
    assert(image.data != NULL);
    assert(image.size == sizeof(payload));
    assert(memcmp(image.data, payload, sizeof(payload)) == 0);
    cxcu_module_image_free(&image);
    assert(cxcu_compile_module_image_cached(&cache, "source", "bad.cu", NULL, 0u, &image, &err));
    assert(image.data != NULL);
    assert(image.size == sizeof(payload));
    assert(memcmp(image.data, payload, sizeof(payload)) == 0);
    cxcu_module_image_free(&image);

    out = fopen(path_a, "wb");
    assert(out != NULL);
    assert(fclose(out) == 0);
    assert(!cxcu_module_cache_read_image(path_a, &image));
    assert(image.data == NULL);
    (void)remove(path_a);
    printf("  ok test_module_cache_path_and_file_io\n");
}

static void test_buffer_group_helpers(void) {
    cxcu_error err = {0};
    cxcu_buffer a = {0};
    cxcu_buffer b = {0};
    cxcu_buffer* storage[1];
    cxcu_buffer_group group;

    assert(!cxcu_buffer_alloc_upload(&a, NULL, 4u, &err));
    assert(a.device_ptr == 0u);
    assert(a.bytes == 0u);
    assert(err.status == CXCU_STATUS_INVALID_ARGUMENT);

    cxcu_buffer_group_init(&group, storage, 1u);
    assert(cxcu_buffer_group_add(&group, &a, &err));
    assert(group.count == 1u);
    assert(!cxcu_buffer_group_add(&group, &b, &err));
    assert(err.status == CXCU_STATUS_INVALID_ARGUMENT);
    cxcu_buffer_group_free_all(&group);
    assert(group.count == 0u);
    printf("  ok test_buffer_group_helpers\n");
}

static void print_device_info(int count, cxcu_error* err) {
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cxcu_device_info info;
        double total_gib = 0.0;

        assert(cxcu_device_info_get(ordinal, &info, err));
        total_gib = (double)info.total_memory / (1024.0 * 1024.0 * 1024.0);
        printf(
            "  gpu %d: %s, compute capability %d.%d, memory %.2f GiB (%zu bytes)\n",
            info.ordinal,
            info.name,
            info.compute_capability_major,
            info.compute_capability_minor,
            total_gib,
            info.total_memory);
    }
}

static void test_availability_and_optional_copy(void) {
    cxcu_error err = {0};
    int available = cxcu_available(&err);

    if (!available) {
        assert(err.status == CXCU_STATUS_UNAVAILABLE || err.status == CXCU_STATUS_CUDA_ERROR);
        assert(err.message[0] != '\0');
        printf("  ok test_availability_unavailable: %s\n", err.message);
        return;
    }

    int count = 0;
    cxcu_device_info info;
    cxcu_buffer buffer = {0};
    const double input[4] = {1.0, 2.0, 3.0, 4.0};
    double output[4] = {0.0, 0.0, 0.0, 0.0};

    assert(cxcu_device_count(&count, &err));
    assert(count > 0);
    assert(cxcu_device_info_get(0, &info, &err));
    assert(info.name[0] != '\0');
    print_device_info(count, &err);
    assert(cxcu_buffer_alloc_upload(&buffer, input, sizeof(input), &err));
    assert(buffer.device_ptr != 0u);
    assert(buffer.bytes == sizeof(input));
    assert(cxcu_memcpy_d2h(output, &buffer, sizeof(output), &err));
    assert(memcmp(input, output, sizeof(input)) == 0);
    cxcu_buffer_free(&buffer);
    cxcu_shutdown();
    printf("  ok test_availability_and_optional_copy\n");
}

static const char* cxcu_smoke_fill_ptx =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry cxcu_smoke_fill(\n"
    "    .param .u64 out_ptr,\n"
    "    .param .u32 count\n"
    ")\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    .reg .b32 %r<6>;\n"
    "    .reg .b64 %rd<5>;\n"
    "\n"
    "    ld.param.u64 %rd1, [out_ptr];\n"
    "    ld.param.u32 %r1, [count];\n"
    "    mov.u32 %r2, %tid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mov.u32 %r4, %ntid.x;\n"
    "    mad.lo.s32 %r5, %r3, %r4, %r2;\n"
    "    setp.ge.u32 %p, %r5, %r1;\n"
    "    @%p bra DONE;\n"
    "    mul.wide.u32 %rd2, %r5, 4;\n"
    "    add.s64 %rd3, %rd1, %rd2;\n"
    "    add.u32 %r5, %r5, 7;\n"
    "    st.global.u32 [%rd3], %r5;\n"
    "DONE:\n"
    "    ret;\n"
    "}\n";

static void test_optional_kernel_launch_smoke(void) {
    cxcu_error err = {0};
    if (!cxcu_available(&err)) {
        printf("  ok test_optional_kernel_launch_smoke skipped: %s\n", err.message);
        return;
    }

    enum { count = 8 };
    uint32_t output[count] = {0};
    uint64_t device_ptr = 0u;
    uint32_t count_arg = count;
    void* args[2];
    cxcu_buffer buffer = {0};
    cxcu_module module = {0};
    cxcu_launch_config cfg;

    assert(cxcu_buffer_alloc(&buffer, sizeof(output), &err));
    assert(cxcu_module_load_data(
        &module,
        cxcu_smoke_fill_ptx,
        strlen(cxcu_smoke_fill_ptx) + 1u,
        &err));

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid_x = 1u;
    cfg.grid_y = 1u;
    cfg.grid_z = 1u;
    cfg.block_x = count;
    cfg.block_y = 1u;
    cfg.block_z = 1u;

    device_ptr = (uint64_t)buffer.device_ptr;
    args[0] = &device_ptr;
    args[1] = &count_arg;
    assert(cxcu_launch(&module, "cxcu_smoke_fill", &cfg, args, &err));
    assert(cxcu_synchronize(&err));
    assert(cxcu_memcpy_d2h(output, &buffer, sizeof(output), &err));

    for (uint32_t i = 0; i < count; ++i) {
        assert(output[i] == i + 7u);
    }

    cxcu_module_unload(&module);
    cxcu_buffer_free(&buffer);
    cxcu_shutdown();
    printf("  ok test_optional_kernel_launch_smoke\n");
}

static void test_optional_nvrtc_compile_smoke(void) {
    static const char* source =
        "extern \"C\" __global__ void cxcu_nvrtc_fill(unsigned int* out, unsigned int count) {\n"
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < count) out[i] = i + 11u;\n"
        "}\n";
    static const char* options[] = {
        "--std=c++11",
        "--gpu-architecture=compute_50"
    };
    cxcu_error err = {0};
    cxcu_module_image image = {0};
    cxcu_module_image cached_image = {0};
    cxcu_module_cache cache;
    cxcu_module module = {0};
    cxcu_buffer buffer = {0};
    cxcu_launch_config cfg;
    enum { count = 8 };
    uint32_t output[count] = {0};
    uint64_t device_ptr = 0u;
    uint32_t count_arg = count;
    void* args[2];

    if (!cxcu_available(&err)) {
        printf("  ok test_optional_nvrtc_compile_smoke skipped: %s\n", err.message);
        return;
    }
    memset(&cache, 0, sizeof(cache));
    cache.enabled = 1;
    cache.directory = "/tmp/cxcu_nvrtc_cache_test";
    cache.namespace_tag = "cxcu-nvrtc-smoke-v1";
    cache.file_prefix = "nvrtc_";
    cache.file_extension = ".cubin";
    cache.device_ordinal = 0;
    if (!cxcu_compile_module_image_cached(
            &cache,
            source,
            "cxcu_nvrtc_fill.cu",
            options,
            2u,
            &image,
            &err)) {
        printf("  ok test_optional_nvrtc_compile_smoke skipped: %s\n", err.message);
        return;
    }
    assert(image.data != NULL);
    assert(image.size > 0u);
    assert(image.kind == CXCU_MODULE_IMAGE_CUBIN || image.kind == CXCU_MODULE_IMAGE_PTX);
    assert(cxcu_compile_module_image_cached(
        &cache,
        source,
        "cxcu_nvrtc_fill.cu",
        options,
        2u,
        &cached_image,
        &err));
    assert(cached_image.data != NULL);
    assert(cached_image.size == image.size);
    cxcu_module_image_free(&cached_image);
    assert(cxcu_module_load_data(&module, image.data, image.size, &err));
    assert(cxcu_buffer_alloc(&buffer, sizeof(output), &err));

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid_x = 1u;
    cfg.grid_y = 1u;
    cfg.grid_z = 1u;
    cfg.block_x = count;
    cfg.block_y = 1u;
    cfg.block_z = 1u;

    device_ptr = (uint64_t)buffer.device_ptr;
    args[0] = &device_ptr;
    args[1] = &count_arg;
    assert(cxcu_launch(&module, "cxcu_nvrtc_fill", &cfg, args, &err));
    assert(cxcu_synchronize(&err));
    assert(cxcu_memcpy_d2h(output, &buffer, sizeof(output), &err));
    for (uint32_t i = 0; i < count; ++i) {
        assert(output[i] == i + 11u);
    }

    cxcu_buffer_free(&buffer);
    cxcu_module_unload(&module);
    cxcu_module_image_free(&image);
    cxcu_shutdown();
    printf("  ok test_optional_nvrtc_compile_smoke\n");
}

int main(void) {
    test_status_names();
    test_error_clear();
    test_error_format();
    test_source_concat();
    test_module_cache_path_and_file_io();
    test_buffer_group_helpers();
    test_availability_and_optional_copy();
    test_optional_kernel_launch_smoke();
    test_optional_nvrtc_compile_smoke();
    return 0;
}
