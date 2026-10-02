"""Plot recorded CPU/GPU benchmark medians and observed ranges without new solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#406c86", "#ca7739", "#739453", "#876a99", "#578b8b", "#b85a66")
LABELS = {
    "serial": "Serial",
    "thread": "Threads",
    "process": "Processes",
    "cpu": "CPU",
    "global-cupy": "CuPy\n(global)",
    "global-cudss": "cuDSS\n(global)",
    "local-cudss": "cuDSS\n(local)",
    "local-global-cudss": "cuDSS\n(both)",
    "local-amgx": "AmgX\n(local)",
}


def read_report(path: Path) -> dict[str, Any]:
    """Read a stable-source benchmark and validate the recorded repetitions."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["source_changed_during_run"]:
        raise ValueError(f"Cannot plot a changing-source benchmark: {path.name}")
    for case in report["cases"]:
        for measurement in case["measurements"]:
            durations = np.asarray(measurement["seconds"])
            if len(durations) < 3 or not np.isfinite(durations).all() or min(durations) <= 0:
                raise ValueError("At least three finite, positive durations are required")
            if not np.isclose(
                np.median(durations), measurement["median_seconds"], rtol=1e-12, atol=0
            ):
                raise ValueError("Recorded median disagrees with individual timings")
    return report


def bar_panel(
    axis: Any,
    labels: list[str],
    medians: np.ndarray,
    minima: np.ndarray,
    maxima: np.ndarray,
    *,
    speedup: bool,
) -> None:
    """Render linear-axis bars with observed minimum/maximum whiskers."""
    positions = np.arange(len(labels))
    bars = axis.bar(
        positions,
        medians,
        width=0.65,
        color=COLORS[: len(labels)],
        yerr=np.stack((medians - minima, maxima - medians)),
        capsize=4,
        error_kw={"elinewidth": 1.3, "capthick": 1.3},
    )
    axis.set_xticks(positions, labels)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylabel("CPU baseline / elapsed time" if speedup else "Complete solve time (s)")
    if speedup:
        axis.axhline(1, color="#303030", linestyle="--", linewidth=1.2, label="Baseline = 1")
    for bar, value, maximum in zip(bars, medians, maxima, strict=True):
        axis.annotate(
            f"{value:.2f}" if speedup else f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, maximum),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )


def plot_report(report: dict[str, Any], title: str, output: Path) -> None:
    """Compare workloads using common zero-origin time and speedup axes."""
    cases = report["cases"]
    figure, axes = plt.subplots(
        2, len(cases), figsize=(13, 8), squeeze=False, constrained_layout=True
    )
    largest_time = 0.0
    largest_speedup = 1.0
    for index, case in enumerate(cases):
        measurements = case["measurements"]
        labels = [LABELS[item.get("name", item["backend"])] for item in measurements]
        durations = [np.asarray(item["seconds"]) for item in measurements]
        medians = np.array([np.median(values) for values in durations])
        minima = np.array([min(values) for values in durations])
        maxima = np.array([max(values) for values in durations])
        baseline = medians[0]
        bar_panel(axes[0, index], labels, medians, minima, maxima, speedup=False)
        bar_panel(
            axes[1, index],
            labels,
            baseline / medians,
            baseline / maxima,
            baseline / minima,
            speedup=True,
        )
        largest_time = max(largest_time, float(max(maxima)))
        largest_speedup = max(largest_speedup, float(max(baseline / minima)))
        axes[0, index].set_title(
            f"{case['macrocells']} macrotriangles; {case['total_fine_triangles']:,} fine triangles"
        )
        axes[1, index].set_title("Speedup relative to this workload's CPU median")
    for axis in axes[0]:
        axis.set_ylim(0, largest_time * 1.18)
    for axis in axes[1]:
        axis.set_ylim(0, largest_speedup * 1.2)
    axes[1, 0].legend(loc="upper right", framealpha=0.95, fontsize=9)
    figure.suptitle(
        title + "\n45 local pressure unknowns; one native thread; complete solves including setup",
        fontsize=14,
    )
    repetitions = sorted({len(item["seconds"]) for case in cases for item in case["measurements"]})
    count = ", ".join(str(value) for value in repetitions)
    figure.supxlabel(
        f"Bars: medians. Whiskers: observed min–max of {count} repetitions, "
        "not confidence intervals.\n"
        "Speedup ranges use the fixed CPU median as numerator; all axes are linear and start at 0.",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(output.with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)


def main() -> None:
    """Render separate CPU direct, CPU AMG, and GPU performance figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "benchmarks/results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures/performance")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10.5, "svg.fonttype": "none"})
    for name, title in (
        ("linux-cpu", "CPU direct local solves: serial, two threads, two processes"),
        ("linux-cpu-amg", "CPU AMG local solves: serial, two threads, two processes"),
        ("linux-gpu", "GPU solver placement: NVIDIA GeForce RTX 3060 (12 GB)"),
    ):
        report = read_report(args.input_dir / f"{name}.json")
        plot_report(report, title, args.output_dir / name)
        print(f"Rendered {name}.png and {name}.svg")


if __name__ == "__main__":
    main()
