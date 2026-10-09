"""Plot complete-workflow and separately qualified CPU kernel measurements."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(figure: Any, output: Path, name: str) -> list[Path]:
    """Save a publication-size figure and its vector originals with attribution."""
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = output / f"{name}.{suffix}"
        metadata = (
            {"Creator": "IPES Research Group", "Rights": "CC BY 4.0"}
            if suffix == "svg"
            else {"Author": "IPES Research Group"}
        )
        figure.savefig(path, dpi=220, metadata=metadata)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        paths.append(path)
    return paths


def plot_comparison(report: dict[str, Any], reports: dict[str, Any], output: Path) -> list[Path]:
    """Show first/warm complete times and kernel gains without mixing their scopes.

    Every warm repetition appears on the complete-duration figure. First-call
    bars exclude imports and outer interpreter startup, whose separately
    measured values remain in the launch receipts. The microkernel figure
    explicitly excludes all surrounding assembly and solution costs.
    """
    rows = report["rows"]
    colors = ("#355c7d", "#b45f25")
    labels = []
    for row in rows:
        dimension = row["case"]["dimension"]
        fine_cells = reports[f"original-{dimension}d"]["rows"][0]["first"]["fine_cells"]
        labels.append(f"{dimension}D\n{fine_cells:,} local elements")
    positions = np.arange(len(rows))
    paths: list[Path] = []
    with plt.rc_context(
        {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10}
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), layout="constrained")
        for axis, scope in zip(axes, ("first", "warm"), strict=True):
            for offset, (mode, title) in enumerate(
                (("original", "Original kernels"), ("compiled", "Compiled kernels"))
            ):
                centers = positions + (offset - 0.5) * 0.32
                values = [
                    row[f"{mode}_{'first_seconds' if scope == 'first' else 'warm_median_seconds'}"]
                    for row in rows
                ]
                axis.bar(centers, values, width=0.30, color=colors[offset], label=title)
                if scope == "warm":
                    for center, row in zip(centers, rows, strict=True):
                        record = reports[f"{mode}-{row['case']['dimension']}d"]["rows"][0]
                        samples = [sample["complete_seconds"] for sample in record["warm"]]
                        axis.scatter(
                            np.full(len(samples), center),
                            samples,
                            color="white",
                            edgecolor="#222222",
                            s=23,
                            linewidth=0.65,
                            zorder=3,
                        )
            axis.set_xticks(positions, labels)
            axis.set_ylabel("Complete workflow wall time [s]")
            axis.set_title(
                "First call in a fresh process"
                if scope == "first"
                else "Warm median and repetitions"
            )
            axis.grid(axis="y", alpha=0.20)
            axis.set_axisbelow(True)
            axis.set_ylim(0, max(values) * 1.18)
        axes[0].legend(loc="upper left", framealpha=0.95)
        paths.extend(_save_figure(figure, output, "complete-workflow"))
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.8, 4.6), layout="constrained")
        categories = (
            ("scalar_diffusion_gram", "Diffusion Gram\narray kernel"),
            ("shared_global_reduction", "1,024 shared\ncontribution blocks"),
            ("global_reduction", "Actual MHM\ncontribution blocks"),
        )
        centers = np.arange(len(categories))
        for offset, row in enumerate(rows):
            axis.bar(
                centers + (offset - 0.5) * 0.32,
                [row["micro_speedups"][name] for name, _ in categories],
                width=0.30,
                color=colors[offset],
                label=f"{row['case']['dimension']}D inputs",
            )
        axis.axhline(1, color="#222222", linestyle="--", linewidth=1, label="Equal time")
        axis.set_xticks(centers, [label for _, label in categories])
        axis.set_ylabel("Original / compiled warm median")
        axis.set_title("Kernel measurements; surrounding solve costs excluded")
        axis.set_ylim(
            0, max(row["micro_speedups"][name] for row in rows for name, _ in categories) * 1.18
        )
        axis.grid(axis="y", alpha=0.20)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", framealpha=0.95)
        paths.extend(_save_figure(figure, output, "kernel-speedups"))
        plt.close(figure)
    return paths
