#!/usr/bin/env python3
"""Run or parse the cxcu benchmark test and plot runtime curves."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CPU_RE = re.compile(
    r"CPU sweep: combos=(?P<combos>\d+), items=(?P<items>\d+), .* cpu_ms=(?P<cpu_ms>[0-9.]+)"
)
GPU_RE = re.compile(
    r"CUDA sweep verified: .* gpu_ms=(?P<gpu_ms>[0-9.]+), "
    r"kernel_ms=(?P<kernel_ms>[0-9.]+), h2d_ms=(?P<h2d_ms>[0-9.]+), "
    r"d2h_ms=(?P<d2h_ms>[0-9.]+), compile_ms=(?P<compile_ms>[0-9.]+), "
    r"speedup=(?P<speedup>[0-9.]+)x"
)

DEFAULT_CASES = (
    (1024, 512),
    (2048, 1024),
    (4096, 1024),
    (8192, 2048),
    (16384, 4096),
    (32768, 8192),
    (65536, 8192),
)


def default_binary() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / "tests" / "test_cxcu_batched_parameter_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=default_binary(),
        help="Path to test_cxcu_batched_parameter_sweep.",
    )
    parser.add_argument(
        "--from-log",
        type=Path,
        help="Parse benchmark output from an existing raw log instead of running the binary.",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        type=int,
        metavar=("COMBOS", "ITEMS"),
        help="Benchmark case. Can be passed multiple times.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("cxcu_benchmark_curve.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--png",
        default="cxcu_benchmark_curve.png",
        help="PNG output path. Use an empty value to skip plotting.",
    )
    parser.add_argument(
        "--speedup-png",
        help="Speedup PNG output path. Defaults to <png stem>_speedup.png. Use an empty value to skip.",
    )
    parser.add_argument(
        "--per-unit-png",
        help="Per-work-item PNG output path. Defaults to <png stem>_per_unit.png. Use an empty value to skip.",
    )
    parser.add_argument(
        "--throughput-png",
        help="Throughput PNG output path. Defaults to <png stem>_throughput.png. Use an empty value to skip.",
    )
    parser.add_argument(
        "--normalized-png",
        help="Deprecated alias for --per-unit-png.",
    )
    return parser.parse_args()


def make_row(cpu_match, gpu_match) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp_utc": timestamp,
        "combo_count": cpu_match.group("combos"),
        "item_count": cpu_match.group("items"),
        "work_items": str(int(cpu_match.group("combos")) * int(cpu_match.group("items"))),
        "cpu_ms": cpu_match.group("cpu_ms"),
        "cuda_available": "true" if gpu_match else "false",
        "gpu_ms": "",
        "kernel_ms": "",
        "h2d_ms": "",
        "d2h_ms": "",
        "compile_ms": "",
        "speedup": "",
    }
    if gpu_match:
        row.update(gpu_match.groupdict())
    return row


def parse_output(text: str) -> list[dict[str, str]]:
    rows = []
    pending_cpu = None

    for line in text.splitlines():
        cpu_match = CPU_RE.search(line)
        if cpu_match:
            if pending_cpu:
                rows.append(make_row(pending_cpu, None))
            pending_cpu = cpu_match
            continue

        gpu_match = GPU_RE.search(line)
        if gpu_match and pending_cpu:
            rows.append(make_row(pending_cpu, gpu_match))
            pending_cpu = None

    if pending_cpu:
        rows.append(make_row(pending_cpu, None))
    return rows


def run_case(binary: Path, combos: int, items: int) -> dict[str, str]:
    completed = subprocess.run(
        [str(binary), str(combos), str(items)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or f"benchmark failed for {combos}x{items}")

    rows = parse_output(completed.stdout)
    if not rows:
        raise RuntimeError(f"could not parse benchmark output:\n{completed.stdout}")
    return rows[-1]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "timestamp_utc",
        "combo_count",
        "item_count",
        "work_items",
        "cpu_ms",
        "cuda_available",
        "gpu_ms",
        "kernel_ms",
        "h2d_ms",
        "d2h_ms",
        "compile_ms",
        "speedup",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def remove_stale_plot(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def plot_png(path: Path, rows: list[dict[str, str]]) -> bool:
    remove_stale_plot(path)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; CSV was written. "
            "Install guide: https://matplotlib.org/stable/users/installing/index.html",
            file=sys.stderr,
        )
        return False

    x = [int(row["work_items"]) for row in rows]
    cpu = [float(row["cpu_ms"]) for row in rows]
    gpu_rows = [row for row in rows if row["gpu_ms"]]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, cpu, marker="o", label="CPU ms")
    if gpu_rows:
        gpu_x = [int(row["work_items"]) for row in gpu_rows]
        gpu = [float(row["gpu_ms"]) for row in gpu_rows]
        kernel = [float(row["kernel_ms"]) for row in gpu_rows]
        plt.plot(gpu_x, gpu, marker="o", label="GPU ms")
        plt.plot(gpu_x, kernel, marker="o", label="Kernel ms")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("work items (combo_count * item_count)")
    plt.ylabel("milliseconds")
    plt.title("cxcu batched parameter sweep")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    return True


def plot_speedup_png(path: Path, rows: list[dict[str, str]]) -> bool:
    remove_stale_plot(path)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; CSV was written. "
            "Install guide: https://matplotlib.org/stable/users/installing/index.html",
            file=sys.stderr,
        )
        return False

    speedup_rows = [row for row in rows if row["gpu_ms"]]
    if not speedup_rows:
        print("no GPU rows available; skipped speedup PNG", file=sys.stderr)
        return False

    x = [int(row["work_items"]) for row in speedup_rows]
    speedup = [
        float(row["speedup"]) if row["speedup"] else float(row["cpu_ms"]) / float(row["gpu_ms"])
        for row in speedup_rows
    ]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, speedup, marker="o", label="CPU ms / GPU ms")
    plt.axhline(1.0, color="0.35", linestyle="--", linewidth=1, label="1x")
    plt.xscale("log")
    plt.ylim(bottom=0)
    plt.xlabel("work items (combo_count * item_count)")
    plt.ylabel("speedup")
    plt.title("cxcu batched parameter sweep speedup")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    return True


def ns_per_work_item(row: dict[str, str], field: str) -> float:
    work_items = int(row["work_items"])
    if work_items == 0:
        return 0.0
    return float(row[field]) * 1000000.0 / float(work_items)


def billion_work_items_per_second(row: dict[str, str], field: str) -> float:
    elapsed_ms = float(row[field])
    if elapsed_ms == 0.0:
        return 0.0
    return float(row["work_items"]) / (elapsed_ms / 1000.0) / 1000000000.0


def plot_per_unit_png(path: Path, rows: list[dict[str, str]]) -> bool:
    remove_stale_plot(path)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; CSV was written. "
            "Install guide: https://matplotlib.org/stable/users/installing/index.html",
            file=sys.stderr,
        )
        return False

    x = [int(row["work_items"]) for row in rows]
    cpu = [ns_per_work_item(row, "cpu_ms") for row in rows]
    gpu_rows = [row for row in rows if row["gpu_ms"]]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, cpu, marker="o", label="CPU ns/work item")
    if gpu_rows:
        gpu_x = [int(row["work_items"]) for row in gpu_rows]
        gpu = [ns_per_work_item(row, "gpu_ms") for row in gpu_rows]
        kernel = [ns_per_work_item(row, "kernel_ms") for row in gpu_rows]
        plt.plot(gpu_x, gpu, marker="o", label="GPU total ns/work item")
        plt.plot(gpu_x, kernel, marker="o", label="Kernel ns/work item")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("work items (combo_count * item_count)")
    plt.ylabel("nanoseconds per work item")
    plt.title("cxcu batched parameter sweep per-work-item cost")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    return True


def plot_throughput_png(path: Path, rows: list[dict[str, str]]) -> bool:
    remove_stale_plot(path)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; CSV was written. "
            "Install guide: https://matplotlib.org/stable/users/installing/index.html",
            file=sys.stderr,
        )
        return False

    x = [int(row["work_items"]) for row in rows]
    cpu = [billion_work_items_per_second(row, "cpu_ms") for row in rows]
    gpu_rows = [row for row in rows if row["gpu_ms"]]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, cpu, marker="o", label="CPU G work items/s")
    if gpu_rows:
        gpu_x = [int(row["work_items"]) for row in gpu_rows]
        gpu = [billion_work_items_per_second(row, "gpu_ms") for row in gpu_rows]
        kernel = [billion_work_items_per_second(row, "kernel_ms") for row in gpu_rows]
        plt.plot(gpu_x, gpu, marker="o", label="GPU total G work items/s")
        plt.plot(gpu_x, kernel, marker="o", label="Kernel G work items/s")

    plt.xscale("log")
    plt.ylim(bottom=0.0)
    plt.xlabel("work items (combo_count * item_count)")
    plt.ylabel("billion work items / second")
    plt.title("cxcu batched parameter sweep throughput")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    return True


def main() -> int:
    args = parse_args()
    cases = tuple((int(c), int(i)) for c, i in args.case) if args.case else DEFAULT_CASES
    png_path = Path(args.png) if args.png else None
    if args.speedup_png is None:
        speedup_png_path = (
            png_path.with_name(f"{png_path.stem}_speedup{png_path.suffix}") if png_path else None
        )
    else:
        speedup_png_path = Path(args.speedup_png) if args.speedup_png else None
    per_unit_png_arg = args.per_unit_png if args.per_unit_png is not None else args.normalized_png
    if per_unit_png_arg is None:
        per_unit_png_path = (
            png_path.with_name(f"{png_path.stem}_per_unit{png_path.suffix}") if png_path else None
        )
    else:
        per_unit_png_path = Path(per_unit_png_arg) if per_unit_png_arg else None
    if args.throughput_png is None:
        throughput_png_path = (
            png_path.with_name(f"{png_path.stem}_throughput{png_path.suffix}") if png_path else None
        )
    else:
        throughput_png_path = Path(args.throughput_png) if args.throughput_png else None

    if args.from_log:
        if not args.from_log.exists():
            print(f"benchmark log not found: {args.from_log}", file=sys.stderr)
            return 1
        rows = parse_output(args.from_log.read_text())
    else:
        if not args.binary.exists():
            print(f"benchmark binary not found: {args.binary}", file=sys.stderr)
            print("Build it with: cmake --build libs/cxcu/build -j4", file=sys.stderr)
            return 1

        rows = []
        for combos, items in cases:
            if combos <= 0 or items < 2:
                print(f"skipping invalid case: {combos} {items}", file=sys.stderr)
                continue
            row = run_case(args.binary, combos, items)
            rows.append(row)

    if not rows:
        print("no benchmark rows produced", file=sys.stderr)
        return 1

    for row in rows:
        gpu_text = f", gpu_ms={row['gpu_ms']}" if row["gpu_ms"] else ", gpu_ms=unavailable"
        print(f"{row['combo_count']}x{row['item_count']}: cpu_ms={row['cpu_ms']}{gpu_text}")

    write_csv(args.csv, rows)
    png_written = False
    if png_path:
        png_written = plot_png(png_path, rows)
    speedup_png_written = False
    if speedup_png_path:
        speedup_png_written = plot_speedup_png(speedup_png_path, rows)
    per_unit_png_written = False
    if per_unit_png_path:
        per_unit_png_written = plot_per_unit_png(per_unit_png_path, rows)
    throughput_png_written = False
    if throughput_png_path:
        throughput_png_written = plot_throughput_png(throughput_png_path, rows)
    print(f"wrote {args.csv}")
    if png_path and png_written:
        print(f"wrote {png_path}")
    elif png_path:
        print(f"skipped {png_path}")
    if speedup_png_path and speedup_png_written:
        print(f"wrote {speedup_png_path}")
    elif speedup_png_path:
        print(f"skipped {speedup_png_path}")
    if per_unit_png_path and per_unit_png_written:
        print(f"wrote {per_unit_png_path}")
    elif per_unit_png_path:
        print(f"skipped {per_unit_png_path}")
    if throughput_png_path and throughput_png_written:
        print(f"wrote {throughput_png_path}")
    elif throughput_png_path:
        print(f"skipped {throughput_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
