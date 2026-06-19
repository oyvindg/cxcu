# cxcu

[![CI](https://github.com/oyvindg/cxcu/actions/workflows/ci.yml/badge.svg)](https://github.com/oyvindg/cxcu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`cxcu` is a small, standalone **C11 bridge for optional CUDA integration**. It
isolates CUDA runtime and driver plumbing from domain libraries, so a project
can detect and use NVIDIA GPUs without making its core C code depend on
`cuda.h`, CUDA libraries, or `nvcc`.

Link against `cxcu` and you can still fall back cleanly to a CPU path when no
CUDA-capable GPU, driver, or runtime is present.

## What it does

- Detect at runtime whether CUDA is usable, with clear driver/library mismatch
  errors (`cxcu_available`)
- Enumerate devices and basic capabilities (`cxcu_device_count`,
  `cxcu_device_info_get`)
- Manage contexts and streams
- Allocate/free device buffers, copy host↔device, allocate-and-upload in one
  call, and track buffer groups for one-shot cleanup
- Load PTX/CUBIN modules, and compile CUDA source at runtime — preferring
  device-specific CUBIN and falling back to PTX (`cxcu_compile_module_image_for_device`)
- Optionally cache compiled module images on disk
- Launch kernels through a narrow C API (`cxcu_launch`)

The public header uses opaque integer handles instead of CUDA types. **Consumers
never include `cuda.h`.** Functions return `1` on success and `0` on failure,
writing details to an optional `cxcu_error*`.

## Quick start

Minimal CMake consumer:

```cmake
cmake_minimum_required(VERSION 3.20)
project(cxcu_quick_start LANGUAGES C)

find_package(cxcu CONFIG REQUIRED)

add_executable(quick_start main.c)
target_link_libraries(quick_start PRIVATE cxcu::cxcu)
```

`main.c`:

```c
#include <cxcu/cxcu.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* CUDA source compiled at runtime. */
static const char* kernel_source =
    "extern \"C\" __global__ void scale(const float* in, float* out,\n"
    "                                   unsigned int n, float k) {\n"
    "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) out[i] = in[i] * k;\n"
    "}\n";

static int fail(const char* operation, const cxcu_error* err) {
    fprintf(stderr, "%s: %s\n", operation, err && err->message[0] ? err->message : "failed");
    return 1;
}

int main(void) {
    enum { item_count = 16 };
    const char* options[] = {"--std=c++11"};
    cxcu_error err = {0};
    cxcu_module_image image = {0};
    cxcu_module module = {0};
    cxcu_buffer input_buffer = {0};
    cxcu_buffer output_buffer = {0};
    cxcu_buffer* buffers[2];
    cxcu_buffer_group group = {0};
    float input[item_count];
    float output[item_count];
    unsigned int n = item_count;
    float scale = 1.75f;
    uintptr_t input_ptr;
    uintptr_t output_ptr;
    void* args[4];
    cxcu_launch_config cfg;
    int rc = 1;
    unsigned int i;

    /* 1. Probe. When unavailable, fall back to a CPU path instead of failing. */
    if (!cxcu_available(&err)) {
        printf("CUDA unavailable: %s\n", err.message);
        return 0;
    }

    /* 2. Compile to a loadable image (CUBIN if possible, else PTX). */
    if (!cxcu_compile_module_image_for_device(
            kernel_source, "scale.cu", options, 1, 0, &image, &err)) {
        return fail("compile", &err);
    }
    if (!cxcu_module_load_data(&module, image.data, image.size, &err)) {
        rc = fail("module load", &err);
        goto cleanup;
    }

    /* 3. Upload input, allocate output. A buffer group frees everything later. */
    for (i = 0; i < item_count; ++i) input[i] = (float)i;

    cxcu_buffer_group_init(&group, buffers, 2);
    if (!cxcu_buffer_group_alloc_upload(&group, &input_buffer, input, sizeof(input), &err)) {
        rc = fail("input upload", &err);
        goto cleanup;
    }
    if (!cxcu_buffer_group_alloc(&group, &output_buffer, sizeof(output), &err)) {
        rc = fail("output allocation", &err);
        goto cleanup;
    }

    /* 4. Launch: one thread per element. */
    input_ptr = input_buffer.device_ptr;
    output_ptr = output_buffer.device_ptr;
    args[0] = &input_ptr;
    args[1] = &output_ptr;
    args[2] = &n;
    args[3] = &scale;

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid_x = cfg.grid_y = cfg.grid_z = 1;
    cfg.block_x = item_count;
    cfg.block_y = cfg.block_z = 1;

    if (!cxcu_launch(&module, "scale", &cfg, args, &err)) {
        rc = fail("kernel launch", &err);
        goto cleanup;
    }
    if (!cxcu_synchronize(&err)) {
        rc = fail("synchronize", &err);
        goto cleanup;
    }

    /* 5. Copy back. */
    if (!cxcu_memcpy_d2h(output, &output_buffer, sizeof(output), &err)) {
        rc = fail("copy result", &err);
        goto cleanup;
    }
    printf("output[0]=%.2f output[%u]=%.2f\n", output[0], item_count - 1, output[item_count - 1]);
    rc = 0;

cleanup:
    cxcu_buffer_group_free_all(&group);
    cxcu_module_unload(&module);
    cxcu_module_image_free(&image);
    cxcu_shutdown();
    return rc;
}
```

Install `cxcu` first, then build the consumer with the install prefix:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/tmp/cxcu-install
cmake --build build -j4
cmake --install build

cmake -S quick-start -B quick-start/build -DCMAKE_PREFIX_PATH=/tmp/cxcu-install
cmake --build quick-start/build -j4
quick-start/build/quick_start
```

See **[`examples/`](examples/)** for complete, runnable programs:

- [`device_query.c`](examples/device_query.c) — availability probe and device
  enumeration
- [`nvrtc_vector_scale.c`](examples/nvrtc_vector_scale.c) — loads a `.cu` file
  from disk, derives the NVRTC `--include-path`, compiles, launches, and
  verifies on the CPU

## Build

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

CUDA support is enabled by default when CMake finds the CUDA Driver API. Force
it on or off explicitly:

```bash
cmake -S . -B build -DCXCU_ENABLE_CUDA=ON    # require CUDA
cmake -S . -B build -DCXCU_ENABLE_CUDA=OFF   # portable unavailable stub
```

Build the examples and tests:

```bash
cmake -S . -B build -DCXCU_BUILD_EXAMPLES=ON -DCXCU_BUILD_TESTS=ON
cmake --build build -j4
build/examples/cxcu_example_device_query
```

When NVRTC is available, prefer `cxcu_compile_module_image_for_device()` over
direct PTX compilation: it targets the current device's `sm_XX` and returns
CUBIN when possible, avoiding driver-side PTX JIT and tolerating a CUDA toolkit
newer than the installed driver. It falls back to PTX automatically.

## Design

`cxcu` provides **generic GPU infrastructure only**. It does not know about
application-specific data models, configuration formats, metrics, or business
rules. The intended layering is:

```text
host application or domain library
optional CUDA backend adapter
cxcu (generic CUDA bridge, standalone C)
```

`cxcu` owns generic CUDA mechanics; a host-specific adapter owns the meaning of
its data, algorithms, metrics, and CPU-versus-GPU parity rules. This keeps CUDA
support optional and prevents CUDA-specific build constraints from leaking into
portable C code.

By design, `cxcu` does **not**:

- implement application or optimizer logic
- depend on host application libraries
- parse host-specific configuration or IR formats
- own CPU fallback behavior for application algorithms
- require applications to use CUDA when it is linked
- hide numerical differences between CPU and GPU implementations

## Benchmarks

Two standalone benchmarks estimate where a grid-shaped workload starts paying
off on the GPU. Neither depends on a host application; both run a deterministic
synthetic per-combo/per-item loop on CPU and GPU and verify the results match.

`test_cxcu_batched_parameter_sweep` reports a CPU/GPU comparison for a synthetic
batched workload:

```bash
build/tests/test_cxcu_batched_parameter_sweep 65536 4096
```

Output includes `cpu_ms`, `gpu_ms` (h2d + kernel + d2h), the `kernel_ms` /
`h2d_ms` / `d2h_ms` breakdown, `compile_ms` (runtime compile + module load), and
`speedup` (`cpu_ms / gpu_ms`). When CUDA is unavailable it still prints the CPU
result and exits successfully.

| combos | items | cpu_ms | gpu_ms | kernel_ms | h2d_ms | d2h_ms | compile_ms | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 512 | 11.059 | 0.298 | 0.183 | 0.086 | 0.030 | 116.977 | 37.07x |
| 4096 | 1024 | 88.546 | 0.472 | 0.297 | 0.096 | 0.078 | 90.034 | 187.63x |
| 8192 | 2048 | 351.628 | 0.898 | 0.629 | 0.125 | 0.144 | 93.386 | 391.38x |
| 16384 | 4096 | 1434.084 | 2.212 | 1.751 | 0.184 | 0.277 | 106.276 | 648.35x |

`test_cxcu_grid_bench` is a smoke benchmark over a range of grid sizes; its
`gpu_ms` excludes PTX module load and a warmup run precedes the table:

```bash
build/tests/test_cxcu_grid_bench --smoke
build/tests/test_cxcu_grid_bench --full
build/tests/test_cxcu_grid_bench --case 65536 4096
```

## Versioning

The API is small and still pre-1.0 (`CXCU_VERSION_MAJOR.MINOR.PATCH` in
[`include/cxcu/cxcu.h`](include/cxcu/cxcu.h)). See that header for the full,
documented API surface.
