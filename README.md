# cxcu

`cxcu` is intended to be a small standalone C11 bridge for optional CUDA
integration. Its purpose is to isolate CUDA runtime and driver plumbing from
domain libraries, so projects can detect and use NVIDIA GPUs without making
their core C code depend directly on CUDA headers, CUDA libraries, or `nvcc`.

The library should provide generic GPU infrastructure only. It should not know
about application-specific data models, configuration formats, metrics, or
business rules.

## Purpose

`cxcu` exists to make optional CUDA backends practical while keeping the rest of
the codebase portable:

- detect whether CUDA is usable at runtime
- report clear driver/library mismatch errors
- enumerate CUDA devices and basic capabilities
- manage CUDA contexts and streams
- allocate and free device buffers
- copy data between host and device memory
- load PTX/CUBIN modules
- compile runtime CUDA source to a loadable module image, preferring
  device-specific CUBIN and falling back to PTX
- launch kernels through a narrow C API
- expose useful CUDA error strings

Consumers should be able to link against `cxcu` and still fall back cleanly when
no CUDA-capable GPU, driver, or runtime is available.

## Intended Layering

```text
host application or domain library
optional CUDA backend adapter
cxcu                         generic CUDA bridge, standalone C
```

`cxcu` owns generic CUDA mechanics. A host-specific adapter owns the meaning of
its data, algorithms, metrics, and CPU versus GPU parity rules.

This keeps CUDA support optional and prevents CUDA-specific build constraints
from leaking into portable C libraries or applications that only need a CPU
path.

## Non-goals

`cxcu` should not:

- implement application or optimizer logic
- depend on host application libraries
- parse host-specific configuration or IR formats
- own CPU fallback behavior for application algorithms
- require applications to use CUDA when it is linked
- hide numerical differences between CPU and GPU implementations

## Build Strategy

The preferred integration model is optional:

- `cxcu` is built only when CUDA support is enabled
- projects using it keep a CPU path as the default
- runtime availability is checked before selecting a CUDA backend
- failures such as `cuInit` errors are reported as ordinary unavailable states

For strict C host code, `cxcu` can use the CUDA Driver API from `.c` files and
load PTX/CUBIN artifacts produced by CUDA tooling. Kernel sources may still be
compiled with NVIDIA tools, but that compilation stays outside the core domain
libraries.

When NVRTC is available, prefer `cxcu_compile_module_image_for_device()` over
direct PTX compilation. It targets the current device's `sm_XX` architecture and
returns CUBIN when possible, which avoids driver-side PTX JIT and is more robust
when the installed CUDA toolkit is newer than the NVIDIA driver. If CUBIN is not
available, it falls back to PTX.

## Build

```bash
cmake -S libs/cxcu -B libs/cxcu/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build libs/cxcu/build -j4
ctest --test-dir libs/cxcu/build --output-on-failure
```

CUDA support is enabled by default when CMake can find the CUDA Driver API:

```bash
cmake -S libs/cxcu -B libs/cxcu/build -DCXCU_ENABLE_CUDA=ON
```

To force a portable unavailable stub:

```bash
cmake -S libs/cxcu -B libs/cxcu/build -DCXCU_ENABLE_CUDA=OFF
```

To build the standalone examples:

```bash
cmake -S libs/cxcu -B libs/cxcu/build -DCXCU_BUILD_EXAMPLES=ON
cmake --build libs/cxcu/build -j4
```

See `examples/` for host-agnostic device discovery and file-based `.cu` runtime
compilation examples.

## CPU/GPU Benchmark Example

`test_cxcu_batched_parameter_sweep` compares a CPU implementation with an
optional CUDA implementation of the same synthetic batched workload:

```bash
cmake -S libs/cxcu -B libs/cxcu/build -DCXCU_BUILD_TESTS=ON
cmake --build libs/cxcu/build -j4
libs/cxcu/build/tests/test_cxcu_batched_parameter_sweep 65536 4096
```

The output includes:

- `cpu_ms`: CPU evaluation time
- `gpu_ms`: host-to-device copy + kernel execution + device-to-host copy
- `kernel_ms`, `h2d_ms`, `d2h_ms`: GPU timing breakdown
- `compile_ms`: runtime CUDA compilation and module load setup
- `speedup`: `cpu_ms / gpu_ms`

When CUDA is unavailable, the example still prints the CPU result and exits
successfully after reporting the CUDA availability reason.

Benchmark result:

| combos | items | cpu_ms | gpu_ms | kernel_ms | h2d_ms | d2h_ms | compile_ms | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 512 | 11.059 | 0.298 | 0.183 | 0.086 | 0.030 | 116.977 | 37.07x |
| 4096 | 1024 | 88.546 | 0.472 | 0.297 | 0.096 | 0.078 | 90.034 | 187.63x |
| 8192 | 2048 | 351.628 | 0.898 | 0.629 | 0.125 | 0.144 | 93.386 | 391.38x |
| 16384 | 4096 | 1434.084 | 2.212 | 1.751 | 0.184 | 0.277 | 106.276 | 648.35x |

A small Python helper can run several benchmark sizes and write a CSV plus a
PNG curve:

```bash
libs/cxcu/tests/run_benchmark_curve.sh
libs/cxcu/tests/run_benchmark_curve.sh 4096:1024 16384:4096 65536:4096
```

The shell helper builds `test_cxcu_batched_parameter_sweep`, runs each benchmark
case, stores raw output in `libs/cxcu/build/benchmark/cxcu_benchmark_raw.log`,
then calls `benchmark_curve.py` to write CSV and PNG outputs from that raw log.

To create an isolated Python environment for plotting:

```bash
CXCU_BENCH_USE_VENV=1 CXCU_BENCH_INSTALL_DEPS=1 libs/cxcu/tests/run_benchmark_curve.sh
```

`CXCU_BENCH_INSTALL_DEPS=1` installs `matplotlib` into
`libs/cxcu/build/benchmark/.venv`. Without it, the helper still writes CSV and
skips PNG if `matplotlib` is not already available. Set `CXCU_PYTHON` to choose
a specific Python 3 executable.

The Python plotter can also be called directly:

```bash
python3 libs/cxcu/tests/benchmark_curve.py \
  --from-log libs/cxcu/build/benchmark/cxcu_benchmark_raw.log \
  --csv /tmp/cxcu_benchmark_curve.csv \
  --png /tmp/cxcu_benchmark_curve.png
```

Plotting uses `matplotlib` when available. If it is missing, the helper still
writes the CSV and prints an install link.

## Initial API Surface

The first API is deliberately small:

```c
#include <cxcu/cxcu.h>

cxcu_error err = {0};
if (!cxcu_available(&err)) {
    /* Fall back to CPU and log err.message if useful. */
}

int count = 0;
cxcu_device_count(&count, &err);

cxcu_buffer buffer = {0};
cxcu_buffer_alloc(&buffer, 4096, &err);
cxcu_buffer_free(&buffer);
```

The public header intentionally uses opaque integer handles instead of CUDA
types. Consumers do not include `cuda.h`.

## Synthetic Grid Benchmark

`test_cxcu_grid_bench` is a standalone smoke benchmark for estimating where a
grid-shaped workload starts benefiting from GPU execution. It intentionally does
not depend on a host application or implement application semantics. Instead, it
runs a deterministic synthetic per-combo/per-item loop on both CPU and GPU,
verifies that the results match, and prints timing for different grid sizes.

The workload is shaped like a generic batched parameter sweep:

- several input arrays
- several parameter arrays
- one result struct per combo containing numeric metrics and a checksum
- one CUDA thread per grid combo, with each thread looping over all input items

```bash
cmake --build libs/cxcu/build -j4
libs/cxcu/build/tests/test_cxcu_grid_bench
```

Useful modes:

```bash
libs/cxcu/build/tests/test_cxcu_grid_bench --smoke
libs/cxcu/build/tests/test_cxcu_grid_bench --full
libs/cxcu/build/tests/test_cxcu_grid_bench --case 65536 4096
```

The printed `gpu_ms` value includes host-to-device copy, kernel execution, and
device-to-host copy. PTX module load is excluded and a warmup run is performed
before the table.
