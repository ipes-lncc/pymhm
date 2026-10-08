"""Render measured complete and prepared-local Darcy timings with phase profiles.

Run ``python -m examples.plot_large_local --input RESULTS.json``. This plotting
script never executes a numerical benchmark or estimates unmeasured speedups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from pymhm.io.workspace import case_workspace, read_resource_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(figure: Any, output: Path, name: str) -> None:
    """Save identical measured data in SVG and PNG formats."""
    figure.savefig(output / f"{name}.svg")
    figure.savefig(output / f"{name}.png", dpi=180)
    plt.close(figure)


def title(case: dict[str, Any]) -> str:
    """Identify the macro and local work without conflating their problem sizes."""
    return (
        f"{case['macrocells']} macros, r={case['local_refinement']}\n"
        f"{case['local_pressure_dofs']:,} local P1 pressure DOFs"
    )


def timing_panels(
    data: dict[str, Any], output: Path, *, prepared: bool, factory: bool = False
) -> None:
    """Plot measured times with observed ranges, using a fixed serial reference."""
    figure, axes = plt.subplots(1, len(data["cases"]), figsize=(13, 4.3), constrained_layout=True)
    for axis, case in zip(np.atleast_1d(axes), data["cases"], strict=True):
        key = "factory_complete_solve" if factory else "complete_solve"
        rows = case["prepared_condensation" if prepared else key]
        baseline = rows[0] if prepared else rows[0]["phases"]["total"]
        axis.axhline(baseline["median_seconds"], color="black", ls=":", label="Serial median")
        if factory:
            axis.axhline(
                case["complete_solve"][0]["phases"]["total"]["median_seconds"],
                color="tab:red",
                ls="--",
                label="solve_darcy serial",
            )
        for backend, style in (("thread", "o-"), ("process", "s-")):
            selected = [r for r in rows if r["backend"] == backend]
            measures = [r if prepared else r["phases"]["total"] for r in selected]
            medians = np.array([r["median_seconds"] for r in measures])
            ranges = np.array(
                [
                    medians - [r["minimum_seconds"] for r in measures],
                    [r["maximum_seconds"] for r in measures] - medians,
                ]
            )
            axis.errorbar(
                [r["workers"] for r in selected],
                medians,
                yerr=ranges,
                fmt=style,
                capsize=3,
                label=backend.capitalize(),
            )
        axis.set(title=title(case), xlabel="Python workers", ylabel="Wall time (seconds)")
        axis.set_ylim(bottom=0)
        axis.set_xticks([1, 2, 4, 8])
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    description = "Prepared local condensation" if prepared else "Complete solve_darcy"
    if factory:
        description = "Complete factory solve: local assembly and condensation in workers"
    figure.suptitle(f"{description}: medians and observed min–max of recorded repetitions")
    name = "factory-complete-solve" if factory else "complete-solve"
    save(figure, output, "prepared-condensation" if prepared else name)


def phases(data: dict[str, Any], output: Path) -> None:
    """Display absolute serial phase costs and measured parallel speedup separately."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(len(data["cases"]))
    bottom = np.zeros_like(x, dtype=float)
    for key, label in (
        ("local_assembly", "Local assembly"),
        ("local_condensation", "Local condensation"),
        ("global_assembly", "Global assembly"),
        ("global_solve", "Global solve"),
        ("reconstruction", "Reconstruction"),
    ):
        values = [r["complete_solve"][0]["phases"][key]["median_seconds"] for r in data["cases"]]
        axes[0].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[0].set_xticks(
        x, [f"{r['macrocells']} macros\nr={r['local_refinement']}" for r in data["cases"]]
    )
    axes[0].set(
        title="Serial phase costs (independent medians)",
        ylabel="Median phase duration (seconds)",
    )
    axes[0].legend(fontsize=8)
    for i, case in enumerate(data["cases"]):
        marker = ("o", "s", "^")[i % 3]
        for key, linestyle, label in (
            ("complete_solve", "-", "complete"),
            ("prepared_condensation", "--", "prepared"),
        ):
            rows = [r for r in case[key] if r["backend"] == "thread"]
            axes[1].plot(
                [r["workers"] for r in rows],
                [r["speedup_over_serial_median"] for r in rows],
                marker=marker,
                linestyle=linestyle,
                label=f"{case['macrocells']} macros / r{case['local_refinement']}: {label}",
            )
    axes[1].axhline(1, color="black", ls=":")
    axes[1].set(
        title="Measured thread speedup", xlabel="Python workers", ylabel="Serial / parallel median"
    )
    axes[1].set_xticks([1, 2, 4, 8])
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("solve_darcy distributes condensation; local assembly is serial")
    save(figure, output, "phase-profile")


def main() -> None:
    """Render complete, unchanged-source benchmark records and preserve their raw data."""
    root = case_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "benchmarks/results/large-local.json")
    parser.add_argument("--output", type=Path, default=root / "docs/figures/large-local")
    args = parser.parse_args()
    data = json.loads(read_resource_text(args.input, encoding="utf-8"))
    if data["pilot"] or data["source_changed_during_run"]:
        raise ValueError("Publication figures require the complete, unchanged-source measured run")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "svg.fonttype": "none"})
    timing_panels(data, args.output, prepared=False)
    timing_panels(data, args.output, prepared=True)
    timing_panels(data, args.output, prepared=False, factory=True)
    phases(data, args.output)
    (args.output / "metrics.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_large_local").main()
