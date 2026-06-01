# cxcu examples

This directory contains host-agnostic examples for small `cxcu` integration
tasks. More complete CPU/GPU parity and benchmark coverage lives under
`tests/`.

## Build

```bash
cmake -S libs/cxcu -B libs/cxcu/build -DCXCU_BUILD_EXAMPLES=ON
cmake --build libs/cxcu/build -j4
```

## device_query

Minimal availability and device-info example:

```bash
libs/cxcu/build/examples/cxcu_example_device_query
```

When CUDA is unavailable, the program prints the reported reason and exits
successfully so it can be used in optional backend probes.

## nvrtc_vector_scale

File-based `.cu` runtime compilation example:

```bash
libs/cxcu/build/examples/cxcu_example_nvrtc_vector_scale
```

The host program is strict C and reads `kernels/vector_scale.cu` at runtime.
That `.cu` file includes `kernels/vector_math.h`; the host derives
`--include-path` from the `.cu` file directory before compiling with NVRTC
through `cxcu`. The example then launches `cxcu_vector_scale`, copies the result
back, and verifies it on the CPU. Pass a custom `.cu` path as the first argument
to test a different kernel file, with any required headers placed beside it or
reachable from the same include directory.
