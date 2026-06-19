# cxcu

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

```c
#include <cxcu/cxcu.h>
#include <stdint.h>
#include <stdio.h>

/* CUDA source compiled at runtime. */
static const char* kernel_source =
    "extern \"C\" __global__ void scale(const float* in, float* out,\n"
    "                                   unsigned int n, float k) {\n"
    "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) out[i] = in[i] * k;\n"
    "}\n";

int main(void) {
    cxcu_error err = {0};

    /* 1. Probe. When unavailable, fall back to a CPU path instead of failing. */
    if (!cxcu_available(&err)) {
        printf("CUDA unavailable: %s\n", err.message);
        return 0;
    }

    /* 2. Compile to a loadable image (CUBIN if possible, else PTX). */
    const char* opts[] = {"--std=c++11"};
    cxcu_module_image image = {0};
    if (!cxcu_compile_module_image_for_device(
            kernel_source, "scale.cu", opts, 1, /*device*/ 0, &image, &err)) {
        fprintf(stderr, "compile failed: %s\n", err.message);
        return 1;
    }

    cxcu_module module = {0};
    cxcu_module_load_data(&module, image.data, image.size, &err);

    /* 3. Upload input, allocate output. A buffer group frees everything later. */
    enum { N = 16 };
    float in[N], out[N];
    for (unsigned i = 0; i < N; ++i) in[i] = (float)i;

    cxcu_buffer in_buf = {0}, out_buf = {0};
    cxcu_buffer* storage[2];
    cxcu_buffer_group group;
    cxcu_buffer_group_init(&group, storage, 2);
    cxcu_buffer_group_alloc_upload(&group, &in_buf, in, sizeof(in), &err);
    cxcu_buffer_group_alloc(&group, &out_buf, sizeof(out), &err);

    /* 4. Launch: one thread per element. */
    uint64_t in_ptr = in_buf.device_ptr, out_ptr = out_buf.device_ptr;
    uint32_t n = N;
    float k = 1.75f;
    void* args[] = {&in_ptr, &out_ptr, &n, &k};

    cxcu_launch_config cfg = {0};
    cfg.grid_x = cfg.grid_y = cfg.grid_z = 1;
    cfg.block_x = N; cfg.block_y = cfg.block_z = 1;

    cxcu_launch(&module, "scale", &cfg, args, &err);
    cxcu_synchronize(&err);

    /* 5. Copy back. */
    cxcu_memcpy_d2h(out, &out_buf, sizeof(out), &err);
    printf("out[0]=%.2f out[%d]=%.2f\n", out[0], N - 1, out[N - 1]);

    /* 6. Clean up. */
    cxcu_buffer_group_free_all(&group);
    cxcu_module_unload(&module);
    cxcu_module_image_free(&image);
    cxcu_shutdown();
    return 0;
}
```

Error handling is omitted on a few calls above for brevity; in real code check
each return value and inspect `err.message` (or `err.status ==
CXCU_STATUS_UNAVAILABLE` to treat CUDA as simply absent).

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
