"""Plot recorded Darcy accuracy studies without repeating timed or native solves.

Read the archived Darcy audit measurements and
render it with ``python -m examples.plot_darcy_audit``.
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
from matplotlib.ticker import NullFormatter


def save(figure: Any, directory: Path, name: str) -> None:
    """Export identical numerical data to SVG and PNG without field interpolation."""
    figure.savefig(directory / f"{name}.svg")
    figure.savefig(directory / f"{name}.png", dpi=180)
    plt.close(figure)


def series(axis: Any, rows: list[dict[str, Any]], coordinate: str) -> None:
    """Compare physical relative flux errors, retaining only accepted solves."""
    for method, label, style in (
        ("primal", "Raw primal P1", "o-"),
        ("mixed", "Mixed RT0", "s-"),
    ):
        accepted = [r for r in rows if r["formulation"] == method and "rejected" not in r]
        if not accepted:
            continue
        axis.plot(
            [r[coordinate] for r in accepted],
            [100 * r["flux_relative_l2"] for r in accepted],
            style,
            label=label,
        )
        if method == "primal" and all("equilibrated" in r for r in accepted):
            axis.plot(
                [r[coordinate] for r in accepted],
                [100 * r["equilibrated"]["flux_relative_l2"] for r in accepted],
                "^--",
                label="Equilibrated primal RT0",
                alpha=0.8,
            )
    axis.set_ylabel(r"$\|q_h-q\|_{L^2}/\|q\|_{L^2}$ (%)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)


def refinement(data: dict[str, Any], directory: Path) -> None:
    """Show distinct approximation parameters, including the local-refinement plateau."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    configurations = (
        ("macro_refinement", "macro_resolution", "Macro resolution n; local r=4, trace P0", True),
        ("local_refinement", "local_refinement", "Local edge subdivisions r; n=4, trace P0", True),
        ("trace_degree", "trace_degree", "Trace degree k; n=4, local r=8", False),
        ("trace_partition", "trace_segments", "Constant trace segments s; n=4, local r=12", True),
    )
    for axis, (key, coordinate, xlabel, log) in zip(axes.flat, configurations, strict=True):
        rows = data["studies"][key]
        series(axis, rows, coordinate)
        axis.set_xlabel(xlabel)
        if log:
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            ticks = sorted({r[coordinate] for r in rows})
            axis.set_xticks(ticks, [str(t) for t in ticks])
        else:
            axis.set_xticks(range(5))
    axes[0, 0].set_title("Macro refinement converges; n=1 is a symmetry exception")
    axes[0, 0].plot([4, 32], [26, 26 / 8], "k:", label="n⁻¹ slope guide")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Fine local meshes cannot remove coarse-trace error")
    axes[1, 0].set_title("Trace enrichment reaches the local P1 accuracy floor")
    axes[1, 1].set_title("Primal s=12 rejected: numerically rank-deficient system")
    figure.suptitle("Cosine Darcy flux: separate refinement studies; exact norm π/√2")
    save(figure, directory, "refinement")


def references(data: dict[str, Any], directory: Path) -> None:
    """Plot independently assembled global FEM solutions against the exact fields."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for method, style in (("CG1", "o-"), ("RT0", "s-")):
        rows = [r for r in data["global_references"] if r["method"] == method]
        x = [r["resolution"] for r in rows]
        for axis, key in zip(axes, ("pressure_relative_l2", "flux_relative_l2"), strict=True):
            axis.loglog(x, [100 * r[key] for r in rows], style, label=f"DOLFINx {method}")
    for axis, title in zip(axes, ("Pressure", "Physical flux"), strict=True):
        axis.set(
            title=title, xlabel="Global fine-grid resolution m", ylabel="Relative L2 error (%)"
        )
        axis.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle("DOLFINx/UFL assembly: five-level global conforming reference study")
    save(figure, directory, "independent-reference")


def conservation(data: dict[str, Any], directory: Path) -> None:
    """Distinguish relative accuracy from normal continuity and physical cell balance."""
    original = [r for r in data["studies"]["macro_refinement"] if r["macro_resolution"] == 4]
    refined = [r for r in data["studies"]["trace_partition"] if r["trace_segments"] == 6]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    names = ["Raw P1", "Recovered RT0", "Mixed RT0"]
    for offset, rows, label in (
        (-0.2, original, "local r=4, trace s=1"),
        (0.2, refined, "r=12, s=6"),
    ):
        primal = next(r for r in rows if r["formulation"] == "primal")
        mixed = next(r for r in rows if r["formulation"] == "mixed")
        fields = [primal, primal["equilibrated"], mixed]
        for axis, key, multiplier in zip(
            axes,
            ("flux_relative_l2", "normal_jump_l2", "physical_fine_balance_max"),
            (100, 1, 1),
            strict=True,
        ):
            axis.bar(
                np.arange(3) + offset,
                [r[key] * multiplier for r in fields],
                width=0.38,
                label=label,
            )
    for axis in axes:
        axis.set_xticks(range(3), names, rotation=16)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set(title="Accuracy", ylabel="Relative physical flux L2 error (%)")
    axes[1].set(
        title="Normal continuity", ylabel="Unweighted interior normal-jump L2 norm", yscale="log"
    )
    axes[2].set(
        title="Fine-cell conservation",
        ylabel="Maximum integrated physical balance defect",
        yscale="log",
    )
    figure.suptitle("Conservation to roundoff does not imply an accurate physical flux")
    save(figure, directory, "accuracy-and-conservation")


def main() -> None:
    """Read complete reproducible records and export three figure pairs and their data."""
    root = case_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "examples/results/darcy-audit.json")
    parser.add_argument("--output", type=Path, default=root / "docs/figures/darcy-audit")
    args = parser.parse_args()
    data = json.loads(read_resource_text(args.input, encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "svg.fonttype": "none"})
    refinement(data, args.output)
    references(data, args.output)
    conservation(data, args.output)
    (args.output / "metrics.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_darcy_audit").main()
