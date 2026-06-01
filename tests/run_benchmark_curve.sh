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

binary="$build_dir/tests/test_cxcu_batched_parameter_sweep"
plotter="$script_dir/benchmark_curve.py"
raw_log="$out_dir/cxcu_benchmark_raw.log"
csv="$out_dir/cxcu_benchmark_curve.csv"
png="$out_dir/cxcu_benchmark_curve.png"

if [ "$#" -eq 0 ]; then
    set -- 1024:512 4096:1024 8192:2048 16384:4096
fi

cmake -S "$cxcu_dir" -B "$build_dir" -DCXCU_BUILD_TESTS=ON -DCXCU_BUILD_EXAMPLES=ON
cmake --build "$build_dir" -j "$jobs" --target test_cxcu_batched_parameter_sweep

mkdir -p "$out_dir"
: > "$raw_log"

if [ "$use_venv" = "1" ]; then
    if [ ! -x "$venv_dir/bin/python" ]; then
        "$python_bin" -m venv "$venv_dir"
    fi
    python_bin="$venv_dir/bin/python"
    if [ "$install_deps" = "1" ]; then
        "$python_bin" -m pip install --upgrade pip matplotlib
    fi
fi

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

printf '$ %s %s --from-log %s --csv %s --png %s\n' "$python_bin" "$plotter" "$raw_log" "$csv" "$png"
"$python_bin" "$plotter" --from-log "$raw_log" --csv "$csv" --png "$png"

printf 'raw_log=%s\n' "$raw_log"
printf 'csv=%s\n' "$csv"
if [ -f "$png" ]; then
    printf 'png=%s\n' "$png"
else
    printf 'png=not_written\n'
fi
