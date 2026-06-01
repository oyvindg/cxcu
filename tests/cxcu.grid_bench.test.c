#define _POSIX_C_SOURCE 200809L

#include <cxcu/cxcu.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct cxcu_grid_bench_case {
    uint32_t combos;
    uint32_t bars;
} cxcu_grid_bench_case;

typedef struct cxcu_grid_market {
    uint32_t* open;
    uint32_t* high;
    uint32_t* low;
    uint32_t* close;
    uint32_t* volume;
} cxcu_grid_market;

typedef struct cxcu_grid_params {
    uint32_t* fast_period;
    uint32_t* slow_period;
    uint32_t* threshold;
    uint32_t* risk;
    uint32_t* stop_loss;
    uint32_t* take_profit;
} cxcu_grid_params;

typedef struct cxcu_grid_bench_result {
    int32_t profit;
    uint32_t max_drawdown;
    uint32_t trades;
    uint32_t wins;
    uint32_t losses;
    uint32_t checksum;
    uint32_t combo;
    uint32_t bars;
} cxcu_grid_bench_result;

typedef struct cxcu_grid_bench_timing {
    double cpu_ms;
    double gpu_h2d_ms;
    double gpu_kernel_ms;
    double gpu_d2h_ms;
} cxcu_grid_bench_timing;

static const char* cxcu_grid_bench_ptx =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry cxcu_grid_bench_eval(\n"
    "    .param .u64 open_ptr,\n"
    "    .param .u64 high_ptr,\n"
    "    .param .u64 low_ptr,\n"
    "    .param .u64 close_ptr,\n"
    "    .param .u64 volume_ptr,\n"
    "    .param .u64 fast_ptr,\n"
    "    .param .u64 slow_ptr,\n"
    "    .param .u64 threshold_ptr,\n"
    "    .param .u64 risk_ptr,\n"
    "    .param .u64 stop_loss_ptr,\n"
    "    .param .u64 take_profit_ptr,\n"
    "    .param .u64 results_ptr,\n"
    "    .param .u32 combo_count,\n"
    "    .param .u32 bar_count\n"
    ")\n"
    "{\n"
    "    .reg .pred %p<16>;\n"
    "    .reg .b32 %r<80>;\n"
    "    .reg .b64 %rd<32>;\n"
    "\n"
    "    ld.param.u64 %rd1, [open_ptr];\n"
    "    ld.param.u64 %rd2, [high_ptr];\n"
    "    ld.param.u64 %rd3, [low_ptr];\n"
    "    ld.param.u64 %rd4, [close_ptr];\n"
    "    ld.param.u64 %rd5, [volume_ptr];\n"
    "    ld.param.u64 %rd6, [fast_ptr];\n"
    "    ld.param.u64 %rd7, [slow_ptr];\n"
    "    ld.param.u64 %rd8, [threshold_ptr];\n"
    "    ld.param.u64 %rd9, [risk_ptr];\n"
    "    ld.param.u64 %rd10, [stop_loss_ptr];\n"
    "    ld.param.u64 %rd11, [take_profit_ptr];\n"
    "    ld.param.u64 %rd12, [results_ptr];\n"
    "    ld.param.u32 %r1, [combo_count];\n"
    "    ld.param.u32 %r2, [bar_count];\n"
    "\n"
    "    mov.u32 %r3, %tid.x;\n"
    "    mov.u32 %r4, %ctaid.x;\n"
    "    mov.u32 %r5, %ntid.x;\n"
    "    mad.lo.u32 %r6, %r4, %r5, %r3;\n"
    "    setp.ge.u32 %p1, %r6, %r1;\n"
    "    @%p1 bra DONE;\n"
    "\n"
    "    mul.wide.u32 %rd13, %r6, 4;\n"
    "    add.s64 %rd14, %rd6, %rd13;\n"
    "    ld.global.u32 %r20, [%rd14];\n"
    "    add.s64 %rd14, %rd7, %rd13;\n"
    "    ld.global.u32 %r21, [%rd14];\n"
    "    add.s64 %rd14, %rd8, %rd13;\n"
    "    ld.global.u32 %r22, [%rd14];\n"
    "    add.s64 %rd14, %rd9, %rd13;\n"
    "    ld.global.u32 %r23, [%rd14];\n"
    "    add.s64 %rd14, %rd10, %rd13;\n"
    "    ld.global.u32 %r24, [%rd14];\n"
    "    add.s64 %rd14, %rd11, %rd13;\n"
    "    ld.global.u32 %r25, [%rd14];\n"
    "\n"
    "    ld.global.u32 %r30, [%rd4];\n"
    "    mov.u32 %r31, %r30;\n"
    "    mov.u32 %r32, %r30;\n"
    "    mov.u32 %r33, 0;\n"
    "    mov.u32 %r34, 0;\n"
    "    mov.u32 %r35, 0;\n"
    "    mov.u32 %r36, 0;\n"
    "    mov.u32 %r37, 0;\n"
    "    mov.u32 %r38, 0;\n"
    "    mov.u32 %r39, 0x811c9dc5;\n"
    "    mov.u32 %r40, 1;\n"
    "\n"
    "LOOP:\n"
    "    mul.wide.u32 %rd15, %r40, 4;\n"
    "    add.s64 %rd16, %rd1, %rd15;\n"
    "    ld.global.u32 %r41, [%rd16];\n"
    "    add.s64 %rd16, %rd2, %rd15;\n"
    "    ld.global.u32 %r42, [%rd16];\n"
    "    add.s64 %rd16, %rd3, %rd15;\n"
    "    ld.global.u32 %r43, [%rd16];\n"
    "    add.s64 %rd16, %rd4, %rd15;\n"
    "    ld.global.u32 %r44, [%rd16];\n"
    "    add.s64 %rd16, %rd5, %rd15;\n"
    "    ld.global.u32 %r45, [%rd16];\n"
    "\n"
    "    sub.s32 %r46, %r44, %r31;\n"
    "    div.s32 %r46, %r46, %r20;\n"
    "    add.s32 %r31, %r31, %r46;\n"
    "    sub.s32 %r47, %r44, %r32;\n"
    "    div.s32 %r47, %r47, %r21;\n"
    "    add.s32 %r32, %r32, %r47;\n"
    "\n"
    "    sub.u32 %r48, %r42, %r43;\n"
    "    sub.s32 %r49, %r44, %r41;\n"
    "    and.b32 %r50, %r23, 7;\n"
    "    add.u32 %r50, %r50, 1;\n"
    "    sub.s32 %r51, %r31, %r32;\n"
    "    mul.lo.s32 %r52, %r49, %r50;\n"
    "    add.s32 %r51, %r51, %r52;\n"
    "    and.b32 %r53, %r48, 15;\n"
    "    add.u32 %r53, %r53, %r22;\n"
    "\n"
    "    mov.u32 %r54, %r51;\n"
    "    setp.lt.s32 %p2, %r51, 0;\n"
    "    @!%p2 bra ABS_DONE;\n"
    "    neg.s32 %r54, %r51;\n"
    "ABS_DONE:\n"
    "    setp.gt.s32 %p3, %r54, %r53;\n"
    "    @%p3 bra TRADE;\n"
    "    and.b32 %r55, %r48, 3;\n"
    "    sub.s32 %r33, %r33, %r55;\n"
    "    bra AFTER_PNL;\n"
    "\n"
    "TRADE:\n"
    "    sub.s32 %r56, %r54, %r53;\n"
    "    mul.lo.s32 %r57, %r49, %r23;\n"
    "    div.s32 %r57, %r57, 16;\n"
    "    setp.lt.s32 %p4, %r51, 0;\n"
    "    @!%p4 bra DIR_DONE;\n"
    "    neg.s32 %r57, %r57;\n"
    "DIR_DONE:\n"
    "    add.s32 %r57, %r57, %r56;\n"
    "    sub.s32 %r57, %r57, %r24;\n"
    "    and.b32 %r58, %r48, 31;\n"
    "    add.s32 %r58, %r58, %r25;\n"
    "    setp.gt.s32 %p5, %r57, %r58;\n"
    "    @!%p5 bra CAP_HIGH_DONE;\n"
    "    mov.u32 %r57, %r58;\n"
    "CAP_HIGH_DONE:\n"
    "    neg.s32 %r59, %r24;\n"
    "    setp.lt.s32 %p6, %r57, %r59;\n"
    "    @!%p6 bra CAP_LOW_DONE;\n"
    "    mov.u32 %r57, %r59;\n"
    "CAP_LOW_DONE:\n"
    "    add.s32 %r33, %r33, %r57;\n"
    "    add.u32 %r36, %r36, 1;\n"
    "    setp.gt.s32 %p7, %r57, 0;\n"
    "    @%p7 bra WIN;\n"
    "    add.u32 %r38, %r38, 1;\n"
    "    bra AFTER_PNL;\n"
    "WIN:\n"
    "    add.u32 %r37, %r37, 1;\n"
    "\n"
    "AFTER_PNL:\n"
    "    setp.gt.s32 %p8, %r33, %r34;\n"
    "    @!%p8 bra PEAK_DONE;\n"
    "    mov.u32 %r34, %r33;\n"
    "PEAK_DONE:\n"
    "    sub.s32 %r60, %r34, %r33;\n"
    "    setp.gt.s32 %p9, %r60, %r35;\n"
    "    @!%p9 bra DD_DONE;\n"
    "    mov.u32 %r35, %r60;\n"
    "DD_DONE:\n"
    "    mul.lo.u32 %r61, %r36, 17;\n"
    "    mul.lo.u32 %r62, %r37, 31;\n"
    "    add.u32 %r61, %r61, %r62;\n"
    "    mul.lo.u32 %r62, %r38, 13;\n"
    "    add.u32 %r61, %r61, %r62;\n"
    "    add.u32 %r61, %r61, %r33;\n"
    "    add.u32 %r61, %r61, %r45;\n"
    "    add.u32 %r61, %r61, %r40;\n"
    "    xor.b32 %r39, %r39, %r61;\n"
    "    mul.lo.u32 %r39, %r39, 16777619;\n"
    "\n"
    "    add.u32 %r40, %r40, 1;\n"
    "    setp.lt.u32 %p10, %r40, %r2;\n"
    "    @%p10 bra LOOP;\n"
    "\n"
    "    mul.wide.u32 %rd27, %r6, 32;\n"
    "    add.s64 %rd28, %rd12, %rd27;\n"
    "    st.global.u32 [%rd28], %r33;\n"
    "    st.global.u32 [%rd28+4], %r35;\n"
    "    st.global.u32 [%rd28+8], %r36;\n"
    "    st.global.u32 [%rd28+12], %r37;\n"
    "    st.global.u32 [%rd28+16], %r38;\n"
    "    st.global.u32 [%rd28+20], %r39;\n"
    "    st.global.u32 [%rd28+24], %r6;\n"
    "    st.global.u32 [%rd28+28], %r2;\n"
    "\n"
    "DONE:\n"
    "    ret;\n"
    "}\n";

static double now_ms(void) {
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static uint32_t cxcu_max_u32(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static uint32_t cxcu_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static void make_market(cxcu_grid_market* market, uint32_t bar_count) {
    uint32_t state = 0x12345678u;
    int32_t close_value = 100000;

    for (uint32_t i = 0; i < bar_count; ++i) {
        int32_t step;
        int32_t open_value = close_value;
        uint32_t spread;
        uint32_t high;
        uint32_t low;
        uint32_t close_u32;
        uint32_t open_u32;

        state = state * 1664525u + 1013904223u + i;
        step = (int32_t)((state >> 24u) % 101u) - 50;
        step += (i % 97u) == 0u ? 17 : 0;
        step -= (i % 233u) == 0u ? 11 : 0;
        close_value += step;
        if (close_value < 10000) close_value = 10000;

        open_u32 = (uint32_t)open_value;
        close_u32 = (uint32_t)close_value;
        spread = 4u + ((state >> 17u) % 41u);
        high = cxcu_max_u32(open_u32, close_u32) + spread;
        low = cxcu_min_u32(open_u32, close_u32);
        low = low > spread ? low - spread : 1u;

        market->open[i] = open_u32;
        market->high[i] = high;
        market->low[i] = low;
        market->close[i] = close_u32;
        market->volume[i] = 1000u + (state & 8191u) + (uint32_t)(step < 0 ? -step : step) * 23u;
    }
}

static void make_params(cxcu_grid_params* params, uint32_t combo_count) {
    for (uint32_t combo = 0; combo < combo_count; ++combo) {
        const uint32_t fast = 3u + (combo % 30u);

        params->fast_period[combo] = fast;
        params->slow_period[combo] = fast + 10u + ((combo / 30u) % 90u);
        params->threshold[combo] = 8u + ((combo / 2700u) % 80u);
        params->risk[combo] = 1u + ((combo >> 2u) & 15u);
        params->stop_loss[combo] = 6u + ((combo >> 6u) & 63u);
        params->take_profit[combo] = 12u + ((combo >> 11u) & 127u);
    }
}

static cxcu_grid_bench_result eval_combo_cpu(
    uint32_t combo,
    const cxcu_grid_market* market,
    const cxcu_grid_params* params,
    uint32_t bar_count) {
    const int32_t fast_period = (int32_t)params->fast_period[combo];
    const int32_t slow_period = (int32_t)params->slow_period[combo];
    const int32_t threshold = (int32_t)params->threshold[combo];
    const int32_t risk = (int32_t)params->risk[combo];
    const int32_t stop_loss = (int32_t)params->stop_loss[combo];
    const int32_t take_profit = (int32_t)params->take_profit[combo];
    int32_t fast_ma = (int32_t)market->close[0];
    int32_t slow_ma = (int32_t)market->close[0];
    int32_t equity = 0;
    int32_t peak = 0;
    int32_t max_drawdown = 0;
    uint32_t trades = 0;
    uint32_t wins = 0;
    uint32_t losses = 0;
    uint32_t checksum = 0x811c9dc5u;
    cxcu_grid_bench_result result;

    for (uint32_t i = 1u; i < bar_count; ++i) {
        const int32_t open = (int32_t)market->open[i];
        const int32_t high = (int32_t)market->high[i];
        const int32_t low = (int32_t)market->low[i];
        const int32_t close = (int32_t)market->close[i];
        const int32_t volume = (int32_t)market->volume[i];
        const int32_t range = high - low;
        const int32_t momentum = close - open;
        const int32_t sensitivity = 1 + (risk & 7);
        const int32_t gate = threshold + (range & 15);
        int32_t signal;
        int32_t abs_signal;

        fast_ma += (close - fast_ma) / fast_period;
        slow_ma += (close - slow_ma) / slow_period;
        signal = (fast_ma - slow_ma) + momentum * sensitivity;
        abs_signal = signal < 0 ? -signal : signal;

        if (abs_signal > gate) {
            const int32_t base = abs_signal - gate;
            const int32_t cap = take_profit + (range & 31);
            const int32_t floor = -stop_loss;
            int32_t pnl = (momentum * risk) / 16;

            if (signal < 0) pnl = -pnl;
            pnl += base;
            pnl -= stop_loss;
            if (pnl > cap) pnl = cap;
            if (pnl < floor) pnl = floor;

            equity += pnl;
            trades += 1u;
            if (pnl > 0) {
                wins += 1u;
            } else {
                losses += 1u;
            }
        } else {
            equity -= range & 3;
        }

        if (equity > peak) peak = equity;
        if (peak - equity > max_drawdown) max_drawdown = peak - equity;
        checksum ^= (uint32_t)(
            equity +
            (int32_t)(trades * 17u) +
            (int32_t)(wins * 31u) +
            (int32_t)(losses * 13u) +
            volume +
            (int32_t)i);
        checksum *= 16777619u;
    }

    result.profit = equity;
    result.max_drawdown = (uint32_t)max_drawdown;
    result.trades = trades;
    result.wins = wins;
    result.losses = losses;
    result.checksum = checksum;
    result.combo = combo;
    result.bars = bar_count;
    return result;
}

static void eval_grid_cpu(
    const cxcu_grid_market* market,
    const cxcu_grid_params* params,
    uint32_t combo_count,
    uint32_t bar_count,
    cxcu_grid_bench_result* out_results) {
    for (uint32_t combo = 0; combo < combo_count; ++combo) {
        out_results[combo] = eval_combo_cpu(combo, market, params, bar_count);
    }
}

static int verify_results(
    const cxcu_grid_bench_result* cpu,
    const cxcu_grid_bench_result* gpu,
    uint32_t combo_count) {
    for (uint32_t i = 0; i < combo_count; ++i) {
        if (memcmp(&cpu[i], &gpu[i], sizeof(cpu[i])) != 0) {
            fprintf(
                stderr,
                "mismatch at combo %" PRIu32
                ": cpu(profit=%" PRId32 ", dd=%" PRIu32 ", trades=%" PRIu32
                ", wins=%" PRIu32 ", losses=%" PRIu32 ", checksum=%" PRIu32
                ") gpu(profit=%" PRId32 ", dd=%" PRIu32 ", trades=%" PRIu32
                ", wins=%" PRIu32 ", losses=%" PRIu32 ", checksum=%" PRIu32 ")\n",
                i,
                cpu[i].profit,
                cpu[i].max_drawdown,
                cpu[i].trades,
                cpu[i].wins,
                cpu[i].losses,
                cpu[i].checksum,
                gpu[i].profit,
                gpu[i].max_drawdown,
                gpu[i].trades,
                gpu[i].wins,
                gpu[i].losses,
                gpu[i].checksum);
            return 0;
        }
    }
    return 1;
}

static void free_market(cxcu_grid_market* market) {
    if (!market) return;
    free(market->volume);
    free(market->close);
    free(market->low);
    free(market->high);
    free(market->open);
    memset(market, 0, sizeof(*market));
}

static void free_params(cxcu_grid_params* params) {
    if (!params) return;
    free(params->take_profit);
    free(params->stop_loss);
    free(params->risk);
    free(params->threshold);
    free(params->slow_period);
    free(params->fast_period);
    memset(params, 0, sizeof(*params));
}

static int allocate_market(cxcu_grid_market* market, uint32_t bar_count) {
    const size_t bytes = (size_t)bar_count * sizeof(uint32_t);

    memset(market, 0, sizeof(*market));
    market->open = (uint32_t*)malloc(bytes);
    market->high = (uint32_t*)malloc(bytes);
    market->low = (uint32_t*)malloc(bytes);
    market->close = (uint32_t*)malloc(bytes);
    market->volume = (uint32_t*)malloc(bytes);
    if (!market->open || !market->high || !market->low || !market->close || !market->volume) {
        free_market(market);
        return 0;
    }
    return 1;
}

static int allocate_params(cxcu_grid_params* params, uint32_t combo_count) {
    const size_t bytes = (size_t)combo_count * sizeof(uint32_t);

    memset(params, 0, sizeof(*params));
    params->fast_period = (uint32_t*)malloc(bytes);
    params->slow_period = (uint32_t*)malloc(bytes);
    params->threshold = (uint32_t*)malloc(bytes);
    params->risk = (uint32_t*)malloc(bytes);
    params->stop_loss = (uint32_t*)malloc(bytes);
    params->take_profit = (uint32_t*)malloc(bytes);
    if (!params->fast_period || !params->slow_period || !params->threshold ||
        !params->risk || !params->stop_loss || !params->take_profit) {
        free_params(params);
        return 0;
    }
    return 1;
}

static void free_device_buffers(
    cxcu_buffer* open_buffer,
    cxcu_buffer* high_buffer,
    cxcu_buffer* low_buffer,
    cxcu_buffer* close_buffer,
    cxcu_buffer* volume_buffer,
    cxcu_buffer* fast_buffer,
    cxcu_buffer* slow_buffer,
    cxcu_buffer* threshold_buffer,
    cxcu_buffer* risk_buffer,
    cxcu_buffer* stop_loss_buffer,
    cxcu_buffer* take_profit_buffer,
    cxcu_buffer* results_buffer) {
    cxcu_buffer_free(results_buffer);
    cxcu_buffer_free(take_profit_buffer);
    cxcu_buffer_free(stop_loss_buffer);
    cxcu_buffer_free(risk_buffer);
    cxcu_buffer_free(threshold_buffer);
    cxcu_buffer_free(slow_buffer);
    cxcu_buffer_free(fast_buffer);
    cxcu_buffer_free(volume_buffer);
    cxcu_buffer_free(close_buffer);
    cxcu_buffer_free(low_buffer);
    cxcu_buffer_free(high_buffer);
    cxcu_buffer_free(open_buffer);
}

static int run_gpu_eval(
    cxcu_module* module,
    const cxcu_grid_market* market,
    const cxcu_grid_params* params,
    uint32_t combo_count,
    uint32_t bar_count,
    cxcu_grid_bench_result* out_results,
    cxcu_grid_bench_timing* timing,
    cxcu_error* err) {
    const unsigned int block_size = 128u;
    const size_t bar_bytes = (size_t)bar_count * sizeof(uint32_t);
    const size_t param_bytes = (size_t)combo_count * sizeof(uint32_t);
    const size_t result_bytes = (size_t)combo_count * sizeof(out_results[0]);
    uint64_t open_device_ptr = 0u;
    uint64_t high_device_ptr = 0u;
    uint64_t low_device_ptr = 0u;
    uint64_t close_device_ptr = 0u;
    uint64_t volume_device_ptr = 0u;
    uint64_t fast_device_ptr = 0u;
    uint64_t slow_device_ptr = 0u;
    uint64_t threshold_device_ptr = 0u;
    uint64_t risk_device_ptr = 0u;
    uint64_t stop_loss_device_ptr = 0u;
    uint64_t take_profit_device_ptr = 0u;
    uint64_t results_device_ptr = 0u;
    uint32_t combo_arg = combo_count;
    uint32_t bars_arg = bar_count;
    void* args[14];
    cxcu_buffer open_buffer = {0};
    cxcu_buffer high_buffer = {0};
    cxcu_buffer low_buffer = {0};
    cxcu_buffer close_buffer = {0};
    cxcu_buffer volume_buffer = {0};
    cxcu_buffer fast_buffer = {0};
    cxcu_buffer slow_buffer = {0};
    cxcu_buffer threshold_buffer = {0};
    cxcu_buffer risk_buffer = {0};
    cxcu_buffer stop_loss_buffer = {0};
    cxcu_buffer take_profit_buffer = {0};
    cxcu_buffer results_buffer = {0};
    cxcu_launch_config cfg;
    double t0 = 0.0;
    double t1 = 0.0;
    int ok = 0;

    if (!cxcu_buffer_alloc(&open_buffer, bar_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&high_buffer, bar_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&low_buffer, bar_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&close_buffer, bar_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&volume_buffer, bar_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&fast_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&slow_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&threshold_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&risk_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&stop_loss_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&take_profit_buffer, param_bytes, err)) goto cleanup;
    if (!cxcu_buffer_alloc(&results_buffer, result_bytes, err)) goto cleanup;

    t0 = now_ms();
    if (!cxcu_memcpy_h2d(&open_buffer, market->open, bar_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&high_buffer, market->high, bar_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&low_buffer, market->low, bar_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&close_buffer, market->close, bar_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&volume_buffer, market->volume, bar_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&fast_buffer, params->fast_period, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&slow_buffer, params->slow_period, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&threshold_buffer, params->threshold, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&risk_buffer, params->risk, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&stop_loss_buffer, params->stop_loss, param_bytes, err)) goto cleanup;
    if (!cxcu_memcpy_h2d(&take_profit_buffer, params->take_profit, param_bytes, err)) goto cleanup;
    t1 = now_ms();
    timing->gpu_h2d_ms = t1 - t0;

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid_x = (combo_count + block_size - 1u) / block_size;
    cfg.grid_y = 1u;
    cfg.grid_z = 1u;
    cfg.block_x = block_size;
    cfg.block_y = 1u;
    cfg.block_z = 1u;

    open_device_ptr = (uint64_t)open_buffer.device_ptr;
    high_device_ptr = (uint64_t)high_buffer.device_ptr;
    low_device_ptr = (uint64_t)low_buffer.device_ptr;
    close_device_ptr = (uint64_t)close_buffer.device_ptr;
    volume_device_ptr = (uint64_t)volume_buffer.device_ptr;
    fast_device_ptr = (uint64_t)fast_buffer.device_ptr;
    slow_device_ptr = (uint64_t)slow_buffer.device_ptr;
    threshold_device_ptr = (uint64_t)threshold_buffer.device_ptr;
    risk_device_ptr = (uint64_t)risk_buffer.device_ptr;
    stop_loss_device_ptr = (uint64_t)stop_loss_buffer.device_ptr;
    take_profit_device_ptr = (uint64_t)take_profit_buffer.device_ptr;
    results_device_ptr = (uint64_t)results_buffer.device_ptr;
    args[0] = &open_device_ptr;
    args[1] = &high_device_ptr;
    args[2] = &low_device_ptr;
    args[3] = &close_device_ptr;
    args[4] = &volume_device_ptr;
    args[5] = &fast_device_ptr;
    args[6] = &slow_device_ptr;
    args[7] = &threshold_device_ptr;
    args[8] = &risk_device_ptr;
    args[9] = &stop_loss_device_ptr;
    args[10] = &take_profit_device_ptr;
    args[11] = &results_device_ptr;
    args[12] = &combo_arg;
    args[13] = &bars_arg;

    t0 = now_ms();
    if (!cxcu_launch(module, "cxcu_grid_bench_eval", &cfg, args, err)) goto cleanup;
    if (!cxcu_synchronize(err)) goto cleanup;
    t1 = now_ms();
    timing->gpu_kernel_ms = t1 - t0;

    t0 = now_ms();
    if (!cxcu_memcpy_d2h(out_results, &results_buffer, result_bytes, err)) goto cleanup;
    t1 = now_ms();
    timing->gpu_d2h_ms = t1 - t0;
    ok = 1;

cleanup:
    free_device_buffers(
        &open_buffer,
        &high_buffer,
        &low_buffer,
        &close_buffer,
        &volume_buffer,
        &fast_buffer,
        &slow_buffer,
        &threshold_buffer,
        &risk_buffer,
        &stop_loss_buffer,
        &take_profit_buffer,
        &results_buffer);
    return ok;
}

static int run_case(cxcu_module* module, cxcu_grid_bench_case bench_case, int print_header) {
    cxcu_error err = {0};
    cxcu_grid_market market;
    cxcu_grid_params params;
    cxcu_grid_bench_result* cpu_results = NULL;
    cxcu_grid_bench_result* gpu_results = NULL;
    cxcu_grid_bench_timing timing;
    const uint64_t work_items = (uint64_t)bench_case.combos * (uint64_t)(bench_case.bars - 1u);
    const size_t results_bytes = (size_t)bench_case.combos * sizeof(cpu_results[0]);
    double t0 = 0.0;
    double t1 = 0.0;
    double gpu_total_ms = 0.0;
    int ok = 0;

    memset(&market, 0, sizeof(market));
    memset(&params, 0, sizeof(params));
    memset(&timing, 0, sizeof(timing));
    cpu_results = (cxcu_grid_bench_result*)malloc(results_bytes);
    gpu_results = (cxcu_grid_bench_result*)malloc(results_bytes);
    if (!cpu_results || !gpu_results ||
        !allocate_market(&market, bench_case.bars) ||
        !allocate_params(&params, bench_case.combos)) {
        fprintf(stderr, "allocation failed for benchmark case\n");
        goto cleanup;
    }

    make_market(&market, bench_case.bars);
    make_params(&params, bench_case.combos);

    t0 = now_ms();
    eval_grid_cpu(&market, &params, bench_case.combos, bench_case.bars, cpu_results);
    t1 = now_ms();
    timing.cpu_ms = t1 - t0;

    if (!run_gpu_eval(
            module,
            &market,
            &params,
            bench_case.combos,
            bench_case.bars,
            gpu_results,
            &timing,
            &err)) {
        fprintf(stderr, "GPU benchmark failed: %s\n", err.message);
        goto cleanup;
    }
    if (!verify_results(cpu_results, gpu_results, bench_case.combos)) goto cleanup;

    gpu_total_ms = timing.gpu_h2d_ms + timing.gpu_kernel_ms + timing.gpu_d2h_ms;
    if (print_header) {
        printf(
            "  %10s %10s %14s %10s %10s %10s %10s %10s %10s\n",
            "combos",
            "bars",
            "combo-bars",
            "cpu_ms",
            "gpu_ms",
            "kernel_ms",
            "h2d_ms",
            "d2h_ms",
            "speedup");
    }
    printf(
        "  %10" PRIu32 " %10" PRIu32 " %14" PRIu64
        " %10.3f %10.3f %10.3f %10.3f %10.3f %9.2fx\n",
        bench_case.combos,
        bench_case.bars,
        work_items,
        timing.cpu_ms,
        gpu_total_ms,
        timing.gpu_kernel_ms,
        timing.gpu_h2d_ms,
        timing.gpu_d2h_ms,
        gpu_total_ms > 0.0 ? timing.cpu_ms / gpu_total_ms : 0.0);
    ok = 1;

cleanup:
    free(gpu_results);
    free(cpu_results);
    free_params(&params);
    free_market(&market);
    return ok;
}

static int run_warmup(cxcu_module* module) {
    const cxcu_grid_bench_case warmup_case = {128u, 128u};
    cxcu_error err = {0};
    cxcu_grid_market market;
    cxcu_grid_params params;
    cxcu_grid_bench_result results[128];
    cxcu_grid_bench_timing timing;
    int ok = 0;

    memset(&market, 0, sizeof(market));
    memset(&params, 0, sizeof(params));
    memset(&timing, 0, sizeof(timing));
    if (!allocate_market(&market, warmup_case.bars) ||
        !allocate_params(&params, warmup_case.combos)) {
        fprintf(stderr, "GPU benchmark warmup allocation failed\n");
        goto cleanup;
    }
    make_market(&market, warmup_case.bars);
    make_params(&params, warmup_case.combos);
    if (!run_gpu_eval(
            module,
            &market,
            &params,
            warmup_case.combos,
            warmup_case.bars,
            results,
            &timing,
            &err)) {
        fprintf(stderr, "GPU benchmark warmup failed: %s\n", err.message);
        goto cleanup;
    }
    ok = 1;

cleanup:
    free_params(&params);
    free_market(&market);
    return ok;
}

static int parse_u32(const char* text, uint32_t* out_value) {
    char* end = NULL;
    unsigned long value = 0ul;

    if (!text || !out_value) return 0;
    value = strtoul(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value == 0ul || value > 0xfffffffful) return 0;
    *out_value = (uint32_t)value;
    return 1;
}

static void print_usage(const char* argv0) {
    printf("usage: %s [--smoke|--full|--case <combos> <bars>]\n", argv0);
}

int main(int argc, char** argv) {
    static const cxcu_grid_bench_case smoke_cases[] = {
        {256u, 512u},
        {2048u, 1024u},
    };
    static const cxcu_grid_bench_case default_cases[] = {
        {128u, 256u},
        {512u, 512u},
        {2048u, 1024u},
        {8192u, 2048u},
        {32768u, 2048u},
    };
    static const cxcu_grid_bench_case full_cases[] = {
        {128u, 256u},
        {512u, 512u},
        {2048u, 1024u},
        {8192u, 2048u},
        {32768u, 4096u},
        {65536u, 4096u},
        {131072u, 8192u},
    };

    const cxcu_grid_bench_case* cases = default_cases;
    size_t case_count = sizeof(default_cases) / sizeof(default_cases[0]);
    cxcu_grid_bench_case single_case = {0u, 0u};
    cxcu_error err = {0};
    cxcu_module module = {0};
    cxcu_device_info info;
    int count = 0;

    if (argc == 2 && strcmp(argv[1], "--smoke") == 0) {
        cases = smoke_cases;
        case_count = sizeof(smoke_cases) / sizeof(smoke_cases[0]);
    } else if (argc == 2 && strcmp(argv[1], "--full") == 0) {
        cases = full_cases;
        case_count = sizeof(full_cases) / sizeof(full_cases[0]);
    } else if (argc == 4 && strcmp(argv[1], "--case") == 0) {
        if (!parse_u32(argv[2], &single_case.combos) || !parse_u32(argv[3], &single_case.bars)) {
            print_usage(argv[0]);
            return 2;
        }
        if (single_case.bars < 2u) {
            fprintf(stderr, "bars must be >= 2\n");
            return 2;
        }
        cases = &single_case;
        case_count = 1u;
    } else if (argc != 1) {
        print_usage(argv[0]);
        return 2;
    }

    if (!cxcu_available(&err)) {
        printf("  ok cxcu_grid_bench skipped: %s\n", err.message);
        return 0;
    }
    if (!cxcu_device_count(&count, &err) || count <= 0) {
        printf("  ok cxcu_grid_bench skipped: %s\n", err.message);
        return 0;
    }
    if (!cxcu_device_info_get(0, &info, &err)) {
        fprintf(stderr, "failed to read CUDA device info: %s\n", err.message);
        return 1;
    }
    if (!cxcu_module_load_data(&module, cxcu_grid_bench_ptx, strlen(cxcu_grid_bench_ptx) + 1u, &err)) {
        fprintf(stderr, "failed to load grid benchmark PTX: %s\n", err.message);
        return 1;
    }

    printf("cxcu synthetic grid benchmark\n");
    printf(
        "  device: %s, compute capability %d.%d, memory %.2f GiB\n",
        info.name,
        info.compute_capability_major,
        info.compute_capability_minor,
        (double)info.total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  workload: one CUDA thread per grid combo, each thread loops over OHLCV bars\n");
    printf("  model: per-combo params, EMA-like state, trade metrics, drawdown, checksum parity\n");
    printf("  note: gpu_ms = h2d + kernel + d2h, module load is excluded; warmup is run first\n");

    if (!run_warmup(&module)) {
        cxcu_module_unload(&module);
        cxcu_shutdown();
        return 1;
    }

    for (size_t i = 0u; i < case_count; ++i) {
        if (!run_case(&module, cases[i], i == 0u)) {
            cxcu_module_unload(&module);
            cxcu_shutdown();
            return 1;
        }
    }

    cxcu_module_unload(&module);
    cxcu_shutdown();
    printf("  ok cxcu_grid_bench\n");
    return 0;
}
