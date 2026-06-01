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
    (4096, 1024),
    (8192, 2048),
    (16384, 4096),
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


def plot_png(path: Path, rows: list[dict[str, str]]) -> bool:
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


def main() -> int:
    args = parse_args()
    cases = tuple((int(c), int(i)) for c, i in args.case) if args.case else DEFAULT_CASES
    png_path = Path(args.png) if args.png else None

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
    print(f"wrote {args.csv}")
    if png_path and png_written:
        print(f"wrote {png_path}")
    elif png_path:
        print(f"skipped {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
