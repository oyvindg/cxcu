#include <cxcu/cxcu.h>

#include <stdio.h>

int main(void) {
    cxcu_error err = {0};
    int count = 0;

    if (!cxcu_available(&err)) {
        printf("CUDA unavailable: %s\n", err.message);
        return 0;
    }

    if (!cxcu_device_count(&count, &err)) {
        fprintf(stderr, "failed to query CUDA device count: %s\n", err.message);
        return 1;
    }

    printf("CUDA devices: %d\n", count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cxcu_device_info info;
        const double memory_gib = 1024.0 * 1024.0 * 1024.0;

        if (!cxcu_device_info_get(ordinal, &info, &err)) {
            fprintf(stderr, "failed to query CUDA device %d: %s\n", ordinal, err.message);
            cxcu_shutdown();
            return 1;
        }

        printf(
            "  %d: %s, compute capability %d.%d, memory %.2f GiB\n",
            info.ordinal,
            info.name,
            info.compute_capability_major,
            info.compute_capability_minor,
            (double)info.total_memory / memory_gib);
    }

    cxcu_shutdown();
    return 0;
}
