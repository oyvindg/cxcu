#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cxcu_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
build_dir=${CXCU_BUILD_DIR:-"$cxcu_dir/build"}
out_dir=${CXCU_BENCH_OUT_DIR:-"$build_dir/benchmark"}
jobs=${CXCU_BUILD_JOBS:-4}
use_venv=${CXCU_BENCH_USE_VENV:-0}
install_deps=${CXCU_BENCH_INSTALL_DEPS:-0}
venv_dir=${CXCU_BENCH_VENV:-"$out_dir/.venv"}
python_bin=${CXCU_PYTHON:-python3}
plot_only=0

binary="$build_dir/tests/test_cxcu_batched_parameter_sweep"
plotter="$script_dir/benchmark_curve.py"
raw_log="$out_dir/cxcu_benchmark_raw.log"
csv="$out_dir/cxcu_benchmark_curve.csv"
png="$out_dir/cxcu_benchmark_curve.png"

if [ "${1:-}" = "--help" ]; then
    printf 'usage: %s [--plot-only] [COMBOS:ITEMS ...]\n' "$0"
    printf '\n'
    printf 'examples:\n'
    printf '  %s\n' "$0"
    printf '  %s 4096:1024 16384:4096\n' "$0"
    printf '  CXCU_BENCH_USE_VENV=1 %s --plot-only\n' "$0"
    exit 0
fi

if [ "${1:-}" = "--plot-only" ]; then
    plot_only=1
    shift
fi

if [ "$plot_only" = "1" ] && [ "$#" -ne 0 ]; then
    printf '%s\n' '--plot-only does not accept benchmark cases' >&2
    exit 1
fi

if [ "$plot_only" = "0" ] && [ "$#" -eq 0 ]; then
    set -- 1024:512 4096:1024 8192:2048 16384:4096
fi

mkdir -p "$out_dir"
if [ -z "${MPLCONFIGDIR:-}" ]; then
    export MPLCONFIGDIR="$out_dir/matplotlib-cache"
fi
mkdir -p "$MPLCONFIGDIR"

if [ "$use_venv" = "1" ]; then
    if [ ! -x "$venv_dir/bin/python" ]; then
        "$python_bin" -m venv "$venv_dir"
    fi
    python_bin="$venv_dir/bin/python"
    if [ "$install_deps" = "1" ]; then
        "$python_bin" -m pip install --upgrade pip matplotlib
    fi
fi

if [ "$plot_only" = "1" ]; then
    if [ ! -s "$raw_log" ]; then
        printf 'benchmark raw log not found: %s\n' "$raw_log" >&2
        printf 'run without --plot-only first, or set CXCU_BENCH_OUT_DIR to an existing benchmark directory\n' >&2
        exit 1
    fi
else
    cmake -S "$cxcu_dir" -B "$build_dir" -DCXCU_BUILD_TESTS=ON -DCXCU_BUILD_EXAMPLES=ON
    cmake --build "$build_dir" -j "$jobs" --target test_cxcu_batched_parameter_sweep
    : > "$raw_log"

    for bench_case in "$@"; do
        bench_case=${bench_case//X/:}
        bench_case=${bench_case//x/:}
        bench_case=${bench_case//,/:}
        combos=${bench_case%%:*}
        items=${bench_case#*:}

        if [ "$items" = "$bench_case" ] ||
           [ -z "$combos" ] ||
           [ -z "$items" ] ||
           [[ "$combos" == *[^0-9]* ]] ||
           [[ "$items" == *[^0-9]* ]]; then
            printf 'invalid benchmark case: %s\n' "$bench_case" >&2
            printf 'use COMBOS:ITEMS, for example 65536:4096\n' >&2
            exit 1
        fi

        {
            printf '$ %s %s %s\n' "$binary" "$combos" "$items"
            "$binary" "$combos" "$items"
            printf '\n'
        } | tee -a "$raw_log"
    done
fi

printf '$ %s %s --from-log %s --csv %s --png %s\n' "$python_bin" "$plotter" "$raw_log" "$csv" "$png"
"$python_bin" "$plotter" --from-log "$raw_log" --csv "$csv" --png "$png"

printf 'raw_log=%s\n' "$raw_log"
printf 'csv=%s\n' "$csv"
if [ -f "$png" ]; then
    printf 'png=%s\n' "$png"
else
    printf 'png=not_written\n'
fi
