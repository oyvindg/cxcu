#include <cxcu/cxcu.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct example_series {
    uint32_t* input_a;
    uint32_t* input_high;
    uint32_t* input_low;
    uint32_t* input_b;
    uint32_t* weight;
} example_series;

typedef struct example_params {
    uint32_t* fast_window;
    uint32_t* slow_window;
    uint32_t* threshold;
    uint32_t* scale;
    uint32_t* floor_limit;
    uint32_t* cap_limit;
} example_params;

typedef struct example_result {
    int32_t score;
    uint32_t max_drop;
    uint32_t events;
    uint32_t positive_events;
    uint32_t negative_events;
    uint32_t checksum;
    uint32_t combo;
    uint32_t item_count;
} example_result;

typedef struct example_timing {
    double cpu_ms;
    double gpu_compile_ms;
    double gpu_h2d_ms;
    double gpu_kernel_ms;
    double gpu_d2h_ms;
} example_timing;

static const char* example_cuda_source =
    "typedef unsigned int uint32_t;\n"
    "typedef int int32_t;\n"
    "\n"
    "typedef struct example_result {\n"
    "    int32_t score;\n"
    "    uint32_t max_drop;\n"
    "    uint32_t events;\n"
    "    uint32_t positive_events;\n"
    "    uint32_t negative_events;\n"
    "    uint32_t checksum;\n"
    "    uint32_t combo;\n"
    "    uint32_t item_count;\n"
    "} example_result;\n"
    "\n"
    "extern \"C\" __global__ void cxcu_example_sweep_eval(\n"
    "    const uint32_t* input_a,\n"
    "    const uint32_t* input_high,\n"
    "    const uint32_t* input_low,\n"
    "    const uint32_t* input_b,\n"
    "    const uint32_t* weight,\n"
    "    const uint32_t* fast_window,\n"
    "    const uint32_t* slow_window,\n"
    "    const uint32_t* threshold,\n"
    "    const uint32_t* scale,\n"
    "    const uint32_t* floor_limit,\n"
    "    const uint32_t* cap_limit,\n"
    "    example_result* results,\n"
    "    uint32_t combo_count,\n"
    "    uint32_t item_count) {\n"
    "    const uint32_t combo = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (combo >= combo_count || item_count == 0u) return;\n"
    "\n"
    "    const int32_t fast = (int32_t)fast_window[combo];\n"
    "    const int32_t slow = (int32_t)slow_window[combo];\n"
    "    const int32_t gate_base = (int32_t)threshold[combo];\n"
    "    const int32_t scale_value = (int32_t)scale[combo];\n"
    "    const int32_t floor_value = (int32_t)floor_limit[combo];\n"
    "    const int32_t cap_value = (int32_t)cap_limit[combo];\n"
    "    int32_t fast_state = (int32_t)input_b[0];\n"
    "    int32_t slow_state = (int32_t)input_b[0];\n"
    "    int32_t score = 0;\n"
    "    int32_t peak = 0;\n"
    "    int32_t max_drop = 0;\n"
    "    uint32_t events = 0u;\n"
    "    uint32_t positive_events = 0u;\n"
    "    uint32_t negative_events = 0u;\n"
    "    uint32_t checksum = 0x811c9dc5u;\n"
    "\n"
    "    for (uint32_t i = 1u; i < item_count; ++i) {\n"
    "        const int32_t a = (int32_t)input_a[i];\n"
    "        const int32_t hi = (int32_t)input_high[i];\n"
    "        const int32_t lo = (int32_t)input_low[i];\n"
    "        const int32_t b = (int32_t)input_b[i];\n"
    "        const int32_t w = (int32_t)weight[i];\n"
    "        const int32_t span = hi - lo;\n"
    "        const int32_t step = b - a;\n"
    "        const int32_t sensitivity = 1 + (scale_value & 7);\n"
    "        const int32_t gate = gate_base + (span & 15);\n"
    "        int32_t signal;\n"
    "        int32_t abs_signal;\n"
    "\n"
    "        fast_state += (b - fast_state) / fast;\n"
    "        slow_state += (b - slow_state) / slow;\n"
    "        signal = (fast_state - slow_state) + step * sensitivity;\n"
    "        abs_signal = signal < 0 ? -signal : signal;\n"
    "\n"
    "        if (abs_signal > gate) {\n"
    "            const int32_t base = abs_signal - gate;\n"
    "            const int32_t cap = cap_value + (span & 31);\n"
    "            const int32_t floor = -floor_value;\n"
    "            int32_t delta = (step * scale_value) / 16;\n"
    "\n"
    "            if (signal < 0) delta = -delta;\n"
    "            delta += base;\n"
    "            delta -= floor_value;\n"
    "            if (delta > cap) delta = cap;\n"
    "            if (delta < floor) delta = floor;\n"
    "\n"
    "            score += delta;\n"
    "            events += 1u;\n"
    "            if (delta > 0) positive_events += 1u;\n"
    "            else negative_events += 1u;\n"
    "        } else {\n"
    "            score -= span & 3;\n"
    "        }\n"
    "\n"
    "        if (score > peak) peak = score;\n"
    "        if (peak - score > max_drop) max_drop = peak - score;\n"
    "        checksum ^= (uint32_t)(score + (int32_t)(events * 17u) +\n"
    "            (int32_t)(positive_events * 31u) +\n"
    "            (int32_t)(negative_events * 13u) + w + (int32_t)i);\n"
    "        checksum *= 16777619u;\n"
    "    }\n"
    "\n"
    "    results[combo].score = score;\n"
    "    results[combo].max_drop = (uint32_t)max_drop;\n"
    "    results[combo].events = events;\n"
    "    results[combo].positive_events = positive_events;\n"
    "    results[combo].negative_events = negative_events;\n"
    "    results[combo].checksum = checksum;\n"
    "    results[combo].combo = combo;\n"
    "    results[combo].item_count = item_count;\n"
    "}\n";

static uint32_t max_u32(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static double now_ms(void) {
    struct timespec ts;

    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void free_series(example_series* series) {
    if (!series) return;
    free(series->weight);
    free(series->input_b);
    free(series->input_low);
    free(series->input_high);
    free(series->input_a);
    memset(series, 0, sizeof(*series));
}

static void free_params(example_params* params) {
    if (!params) return;
    free(params->cap_limit);
    free(params->floor_limit);
    free(params->scale);
    free(params->threshold);
    free(params->slow_window);
    free(params->fast_window);
    memset(params, 0, sizeof(*params));
}

static int allocate_series(example_series* series, uint32_t item_count) {
    const size_t bytes = (size_t)item_count * sizeof(uint32_t);

    memset(series, 0, sizeof(*series));
    series->input_a = (uint32_t*)malloc(bytes);
    series->input_high = (uint32_t*)malloc(bytes);
    series->input_low = (uint32_t*)malloc(bytes);
    series->input_b = (uint32_t*)malloc(bytes);
    series->weight = (uint32_t*)malloc(bytes);
    if (!series->input_a || !series->input_high || !series->input_low ||
        !series->input_b || !series->weight) {
        free_series(series);
        return 0;
    }
    return 1;
}

static int allocate_params(example_params* params, uint32_t combo_count) {
    const size_t bytes = (size_t)combo_count * sizeof(uint32_t);

    memset(params, 0, sizeof(*params));
    params->fast_window = (uint32_t*)malloc(bytes);
    params->slow_window = (uint32_t*)malloc(bytes);
    params->threshold = (uint32_t*)malloc(bytes);
    params->scale = (uint32_t*)malloc(bytes);
    params->floor_limit = (uint32_t*)malloc(bytes);
    params->cap_limit = (uint32_t*)malloc(bytes);
    if (!params->fast_window || !params->slow_window || !params->threshold ||
        !params->scale || !params->floor_limit || !params->cap_limit) {
        free_params(params);
        return 0;
    }
    return 1;
}

static void make_series(example_series* series, uint32_t item_count) {
    uint32_t state = 0x12345678u;
    int32_t value = 100000;

    for (uint32_t i = 0; i < item_count; ++i) {
        int32_t step;
        const int32_t previous = value;
        uint32_t spread;
        uint32_t high;
        uint32_t low;
        uint32_t value_u32;
        uint32_t previous_u32;

        state = state * 1664525u + 1013904223u + i;
        step = (int32_t)((state >> 24u) % 101u) - 50;
        step += (i % 97u) == 0u ? 17 : 0;
        step -= (i % 233u) == 0u ? 11 : 0;
        value += step;
        if (value < 10000) value = 10000;

        previous_u32 = (uint32_t)previous;
        value_u32 = (uint32_t)value;
        spread = 4u + ((state >> 17u) % 41u);
        high = max_u32(previous_u32, value_u32) + spread;
        low = min_u32(previous_u32, value_u32);
        low = low > spread ? low - spread : 1u;

        series->input_a[i] = previous_u32;
        series->input_high[i] = high;
        series->input_low[i] = low;
        series->input_b[i] = value_u32;
        series->weight[i] = 1000u + (state & 8191u) + (uint32_t)(step < 0 ? -step : step) * 23u;
    }
}

static void make_params(example_params* params, uint32_t combo_count) {
    for (uint32_t combo = 0; combo < combo_count; ++combo) {
        const uint32_t fast = 3u + (combo % 30u);

        params->fast_window[combo] = fast;
        params->slow_window[combo] = fast + 10u + ((combo / 30u) % 90u);
        params->threshold[combo] = 8u + ((combo / 2700u) % 80u);
        params->scale[combo] = 1u + ((combo >> 2u) & 15u);
        params->floor_limit[combo] = 6u + ((combo >> 6u) & 63u);
        params->cap_limit[combo] = 12u + ((combo >> 11u) & 127u);
    }
}

static example_result eval_combo_cpu(
    uint32_t combo,
    const example_series* series,
    const example_params* params,
    uint32_t item_count) {
    const int32_t fast = (int32_t)params->fast_window[combo];
    const int32_t slow = (int32_t)params->slow_window[combo];
    const int32_t gate_base = (int32_t)params->threshold[combo];
    const int32_t scale_value = (int32_t)params->scale[combo];
    const int32_t floor_value = (int32_t)params->floor_limit[combo];
    const int32_t cap_value = (int32_t)params->cap_limit[combo];
    int32_t fast_state = (int32_t)series->input_b[0];
    int32_t slow_state = (int32_t)series->input_b[0];
    int32_t score = 0;
    int32_t peak = 0;
    int32_t max_drop = 0;
    uint32_t events = 0u;
    uint32_t positive_events = 0u;
    uint32_t negative_events = 0u;
    uint32_t checksum = 0x811c9dc5u;
    example_result result;

    for (uint32_t i = 1u; i < item_count; ++i) {
        const int32_t a = (int32_t)series->input_a[i];
        const int32_t hi = (int32_t)series->input_high[i];
        const int32_t lo = (int32_t)series->input_low[i];
        const int32_t b = (int32_t)series->input_b[i];
        const int32_t w = (int32_t)series->weight[i];
        const int32_t span = hi - lo;
        const int32_t step = b - a;
        const int32_t sensitivity = 1 + (scale_value & 7);
        const int32_t gate = gate_base + (span & 15);
        int32_t signal;
        int32_t abs_signal;

        fast_state += (b - fast_state) / fast;
        slow_state += (b - slow_state) / slow;
        signal = (fast_state - slow_state) + step * sensitivity;
        abs_signal = signal < 0 ? -signal : signal;

        if (abs_signal > gate) {
            const int32_t base = abs_signal - gate;
            const int32_t cap = cap_value + (span & 31);
            const int32_t floor = -floor_value;
            int32_t delta = (step * scale_value) / 16;

            if (signal < 0) delta = -delta;
            delta += base;
            delta -= floor_value;
            if (delta > cap) delta = cap;
            if (delta < floor) delta = floor;

            score += delta;
            events += 1u;
            if (delta > 0) {
                positive_events += 1u;
            } else {
                negative_events += 1u;
            }
        } else {
            score -= span & 3;
        }

        if (score > peak) peak = score;
        if (peak - score > max_drop) max_drop = peak - score;
        checksum ^= (uint32_t)(
            score +
            (int32_t)(events * 17u) +
            (int32_t)(positive_events * 31u) +
            (int32_t)(negative_events * 13u) +
            w +
            (int32_t)i);
        checksum *= 16777619u;
    }

    memset(&result, 0, sizeof(result));
    result.score = score;
    result.max_drop = (uint32_t)max_drop;
    result.events = events;
    result.positive_events = positive_events;
    result.negative_events = negative_events;
    result.checksum = checksum;
    result.combo = combo;
    result.item_count = item_count;
    return result;
}

static void eval_sweep_cpu(
    const example_series* series,
    const example_params* params,
    uint32_t combo_count,
    uint32_t item_count,
    example_result* out_results) {
    for (uint32_t combo = 0u; combo < combo_count; ++combo) {
        out_results[combo] = eval_combo_cpu(combo, series, params, item_count);
    }
}

static void free_device_buffers(
    cxcu_buffer* input_a_buffer,
    cxcu_buffer* input_high_buffer,
    cxcu_buffer* input_low_buffer,
    cxcu_buffer* input_b_buffer,
    cxcu_buffer* weight_buffer,
    cxcu_buffer* fast_buffer,
    cxcu_buffer* slow_buffer,
    cxcu_buffer* threshold_buffer,
    cxcu_buffer* scale_buffer,
    cxcu_buffer* floor_buffer,
    cxcu_buffer* cap_buffer,
    cxcu_buffer* results_buffer) {
    cxcu_buffer_free(results_buffer);
    cxcu_buffer_free(cap_buffer);
    cxcu_buffer_free(floor_buffer);
    cxcu_buffer_free(scale_buffer);
    cxcu_buffer_free(threshold_buffer);
    cxcu_buffer_free(slow_buffer);
    cxcu_buffer_free(fast_buffer);
    cxcu_buffer_free(weight_buffer);
    cxcu_buffer_free(input_b_buffer);
    cxcu_buffer_free(input_low_buffer);
    cxcu_buffer_free(input_high_buffer);
    cxcu_buffer_free(input_a_buffer);
}

static int run_sweep_cuda(
    const example_series* series,
    const example_params* params,
    uint32_t combo_count,
    uint32_t item_count,
    example_result* out_results,
    example_timing* timing,
    cxcu_error* err) {
    static const char* compile_options[] = {
        "--std=c++11",
        "--gpu-architecture=compute_50",
        "--fmad=false"
    };
    const unsigned int block_size = 128u;
    const size_t item_bytes = (size_t)item_count * sizeof(uint32_t);
    const size_t param_bytes = (size_t)combo_count * sizeof(uint32_t);
    const size_t result_bytes = (size_t)combo_count * sizeof(out_results[0]);
    cxcu_module_image image = {0};
    cxcu_module module = {0};
    cxcu_buffer input_a_buffer = {0};
    cxcu_buffer input_high_buffer = {0};
    cxcu_buffer input_low_buffer = {0};
    cxcu_buffer input_b_buffer = {0};
    cxcu_buffer weight_buffer = {0};
    cxcu_buffer fast_buffer = {0};
    cxcu_buffer slow_buffer = {0};
    cxcu_buffer threshold_buffer = {0};
    cxcu_buffer scale_buffer = {0};
    cxcu_buffer floor_buffer = {0};
    cxcu_buffer cap_buffer = {0};
    cxcu_buffer results_buffer = {0};
    cxcu_launch_config cfg;
    uint64_t input_a_ptr = 0u;
    uint64_t input_high_ptr = 0u;
    uint64_t input_low_ptr = 0u;
    uint64_t input_b_ptr = 0u;
    uint64_t weight_ptr = 0u;
    uint64_t fast_ptr = 0u;
    uint64_t slow_ptr = 0u;
    uint64_t threshold_ptr = 0u;
    uint64_t scale_ptr = 0u;
    uint64_t floor_ptr = 0u;
    uint64_t cap_ptr = 0u;
    uint64_t results_ptr = 0u;
    uint32_t combo_arg = combo_count;
    uint32_t item_arg = item_count;
    void* args[14];
    double t0 = 0.0;
    double t1 = 0.0;
    int ok = 0;

    if (timing) {
        timing->gpu_compile_ms = 0.0;
        timing->gpu_h2d_ms = 0.0;
        timing->gpu_kernel_ms = 0.0;
        timing->gpu_d2h_ms = 0.0;
    }

    t0 = now_ms();
    if (!cxcu_compile_module_image_for_device(
            example_cuda_source,
            "cxcu_example_sweep.cu",
            compile_options,
            sizeof(compile_options) / sizeof(compile_options[0]),
            0,
            &image,
            err)) {
        return 0;
    }
    if (!cxcu_module_load_data(&module, image.data, image.size, err)) goto cleanup;
    t1 = now_ms();
    if (timing) timing->gpu_compile_ms = t1 - t0;

    if (!cxcu_buffer_alloc(&input_a_buffer, item_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&input_high_buffer, item_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&input_low_buffer, item_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&input_b_buffer, item_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&weight_buffer, item_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&fast_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&slow_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&threshold_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&scale_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&floor_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&cap_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&results_buffer, result_bytes, err)) goto cleanup;

    t0 = now_ms();
    if (!cxcu_memcpy_h2d(&input_a_buffer, series->input_a, item_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&input_high_buffer, series->input_high, item_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&input_low_buffer, series->input_low, item_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&input_b_buffer, series->input_b, item_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&weight_buffer, series->weight, item_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&fast_buffer, params->fast_window, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&slow_buffer, params->slow_window, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&threshold_buffer, params->threshold, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&scale_buffer, params->scale, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&floor_buffer, params->floor_limit, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&cap_buffer, params->cap_limit, param_bytes, err)) goto cleanup;
    t1 = now_ms();
    if (timing) timing->gpu_h2d_ms = t1 - t0;

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid_x = (combo_count + block_size - 1u) / block_size;
    cfg.grid_y = 1u;
    cfg.grid_z = 1u;
    cfg.block_x = block_size;
    cfg.block_y = 1u;
    cfg.block_z = 1u;

    input_a_ptr = (uint64_t)input_a_buffer.device_ptr;
    input_high_ptr = (uint64_t)input_high_buffer.device_ptr;
    input_low_ptr = (uint64_t)input_low_buffer.device_ptr;
    input_b_ptr = (uint64_t)input_b_buffer.device_ptr;
    weight_ptr = (uint64_t)weight_buffer.device_ptr;
    fast_ptr = (uint64_t)fast_buffer.device_ptr;
    slow_ptr = (uint64_t)slow_buffer.device_ptr;
    threshold_ptr = (uint64_t)threshold_buffer.device_ptr;
    scale_ptr = (uint64_t)scale_buffer.device_ptr;
    floor_ptr = (uint64_t)floor_buffer.device_ptr;
    cap_ptr = (uint64_t)cap_buffer.device_ptr;
    results_ptr = (uint64_t)results_buffer.device_ptr;

    args[0] = &input_a_ptr;
    args[1] = &input_high_ptr;
    args[2] = &input_low_ptr;
    args[3] = &input_b_ptr;
    args[4] = &weight_ptr;
    args[5] = &fast_ptr;
    args[6] = &slow_ptr;
    args[7] = &threshold_ptr;
    args[8] = &scale_ptr;
    args[9] = &floor_ptr;
    args[10] = &cap_ptr;
    args[11] = &results_ptr;
    args[12] = &combo_arg;
    args[13] = &item_arg;

    t0 = now_ms();
    if (!cxcu_launch(&module, "cxcu_example_sweep_eval", &cfg, args, err)) goto cleanup;
    if (!cxcu_synchronize(err)) goto cleanup;
    t1 = now_ms();
    if (timing) timing->gpu_kernel_ms = t1 - t0;

    t0 = now_ms();
    if (!cxcu_memcpy_d2h(out_results, &results_buffer, result_bytes, err)) goto cleanup;
    t1 = now_ms();
    if (timing) timing->gpu_d2h_ms = t1 - t0;
    ok = 1;

cleanup:
    free_device_buffers(
        &input_a_buffer,
        &input_high_buffer,
        &input_low_buffer,
        &input_b_buffer,
        &weight_buffer,
        &fast_buffer,
        &slow_buffer,
        &threshold_buffer,
        &scale_buffer,
        &floor_buffer,
        &cap_buffer,
        &results_buffer);
    cxcu_module_unload(&module);
    cxcu_module_image_free(&image);
    return ok;
}

static int verify_results(
    const example_result* cpu_results,
    const example_result* gpu_results,
    uint32_t combo_count) {
    for (uint32_t i = 0u; i < combo_count; ++i) {
        if (memcmp(&cpu_results[i], &gpu_results[i], sizeof(cpu_results[i])) != 0) {
            fprintf(
                stderr,
                "mismatch at combo %" PRIu32
                ": cpu(score=%" PRId32 ", drop=%" PRIu32 ", events=%" PRIu32
                ", checksum=%" PRIu32 ") gpu(score=%" PRId32 ", drop=%" PRIu32
                ", events=%" PRIu32 ", checksum=%" PRIu32 ")\n",
                i,
                cpu_results[i].score,
                cpu_results[i].max_drop,
                cpu_results[i].events,
                cpu_results[i].checksum,
                gpu_results[i].score,
                gpu_results[i].max_drop,
                gpu_results[i].events,
                gpu_results[i].checksum);
            return 0;
        }
    }
    return 1;
}

static example_result best_result(const example_result* results, uint32_t combo_count) {
    example_result best = results[0];

    for (uint32_t i = 1u; i < combo_count; ++i) {
        if (results[i].score > best.score ||
            (results[i].score == best.score && results[i].max_drop < best.max_drop)) {
            best = results[i];
        }
    }
    return best;
}

static uint32_t parse_u32_arg(const char* text, uint32_t fallback) {
    char* end = NULL;
    unsigned long value;

    if (!text || text[0] == '\0') return fallback;
    value = strtoul(text, &end, 10);
    if (!end || *end != '\0' || value == 0ul || value > UINT32_MAX) return fallback;
    return (uint32_t)value;
}

int main(int argc, char** argv) {
    const uint32_t combo_count = argc > 1 ? parse_u32_arg(argv[1], 4096u) : 4096u;
    const uint32_t item_count = argc > 2 ? parse_u32_arg(argv[2], 1024u) : 1024u;
    const size_t result_bytes = (size_t)combo_count * sizeof(example_result);
    cxcu_error err = {0};
    example_series series;
    example_params params;
    example_result* cpu_results = NULL;
    example_result* gpu_results = NULL;
    example_timing timing;
    example_result best;
    double t0 = 0.0;
    double t1 = 0.0;
    double gpu_ms = 0.0;
    int exit_code = 1;

    memset(&series, 0, sizeof(series));
    memset(&params, 0, sizeof(params));
    memset(&timing, 0, sizeof(timing));

    if (combo_count == 0u || item_count < 2u) {
        fprintf(stderr, "usage: %s [combo_count] [item_count]\n", argv[0]);
        return 1;
    }

    cpu_results = (example_result*)malloc(result_bytes);
    gpu_results = (example_result*)malloc(result_bytes);
    if (!cpu_results || !gpu_results ||
        !allocate_series(&series, item_count) ||
        !allocate_params(&params, combo_count)) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }

    make_series(&series, item_count);
    make_params(&params, combo_count);
    t0 = now_ms();
    eval_sweep_cpu(&series, &params, combo_count, item_count, cpu_results);
    t1 = now_ms();
    timing.cpu_ms = t1 - t0;
    best = best_result(cpu_results, combo_count);

    printf(
        "CPU sweep: combos=%" PRIu32 ", items=%" PRIu32
        ", best_combo=%" PRIu32 ", score=%" PRId32
        ", max_drop=%" PRIu32 ", events=%" PRIu32
        ", cpu_ms=%.3f\n",
        combo_count,
        item_count,
        best.combo,
        best.score,
        best.max_drop,
        best.events,
        timing.cpu_ms);

    if (!cxcu_available(&err)) {
        printf("CUDA unavailable, keeping CPU result: %s\n", err.message);
        exit_code = 0;
        goto cleanup;
    }

    if (!run_sweep_cuda(&series, &params, combo_count, item_count, gpu_results, &timing, &err)) {
        printf("CUDA sweep unavailable, keeping CPU result: %s\n", err.message);
        exit_code = err.status == CXCU_STATUS_UNAVAILABLE ? 0 : 1;
        goto cleanup;
    }

    if (!verify_results(cpu_results, gpu_results, combo_count)) goto cleanup;
    best = best_result(gpu_results, combo_count);
    gpu_ms = timing.gpu_h2d_ms + timing.gpu_kernel_ms + timing.gpu_d2h_ms;
    printf(
        "CUDA sweep verified: best_combo=%" PRIu32 ", score=%" PRId32
        ", max_drop=%" PRIu32 ", events=%" PRIu32
        ", gpu_ms=%.3f, kernel_ms=%.3f, h2d_ms=%.3f, d2h_ms=%.3f"
        ", compile_ms=%.3f, speedup=%.2fx\n",
        best.combo,
        best.score,
        best.max_drop,
        best.events,
        gpu_ms,
        timing.gpu_kernel_ms,
        timing.gpu_h2d_ms,
        timing.gpu_d2h_ms,
        timing.gpu_compile_ms,
        gpu_ms > 0.0 ? timing.cpu_ms / gpu_ms : 0.0);

    exit_code = 0;

cleanup:
    cxcu_shutdown();
    free(gpu_results);
    free(cpu_results);
    free_params(&params);
    free_series(&series);
    return exit_code;
}
