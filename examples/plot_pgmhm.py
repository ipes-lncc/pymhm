"""Render PGMHM base/enriched fields and separate convergence mechanisms."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples.pgmhm_campaign import exact, exact_flux
from examples.plot_mh2m import panel, read_archive, save
from examples.plot_style import set_refinement_ticks
from pymhm.io.workspace import case_workspace, read_resource_text

ROOT = case_workspace()
RESULTS = ROOT / "examples/results/pgmhm"
OUTPUT = ROOT / "docs/figures/pgmhm"


def macro_convergence(record: dict[str, Any], output: Path, kind: str) -> None:
    """Show base and enriched volume errors without identifying the two solutions."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), layout="constrained")
    for ell in (0, 1):
        rows = [
            r
            for r in record["rows"]
            if r["study"] == "macro" and r["mesh"] == kind and r["trace_degree"] == ell
        ]
        h = np.array([row["macro_diameter"] for row in rows])
        for axis, keys, title in zip(
            axes.ravel(),
            (
                ("pressure_l2", "enriched_pressure_l2"),
                ("flux_l2", "enriched_flux_l2"),
                ("enriched_divergence_l2",),
                ("enriched_macro_balance_max", "unenriched_macro_balance_max"),
            ),
            ("Pressure", "Physical Darcy flux", "Enriched broken divergence", "Macro balance"),
            strict=True,
        ):
            for key in keys:
                values = np.array([row[key] for row in rows])
                label = "enriched" if key.startswith("enriched") else "base"
                axis.loglog(
                    h, values, "o-" if label == "base" else "s--", label=rf"$\ell={ell}$, {label}"
                )
            axis.set(
                title=title,
                xlabel="Macro diameter $H$",
                ylabel="Absolute physical norm"
                if title != "Macro balance"
                else "Maximum absolute balance",
            )
            set_refinement_ticks(axis, h, [f"{value:.3g}" for value in h])
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=12)
    figure.suptitle(rf"{kind}: macro refinement; $k=\ell+2$, $\alpha=0.1$")
    save(figure, output, f"{kind}-convergence")


def skeleton_convergence(record: dict[str, Any], output: Path) -> None:
    """Hold macro geometry fixed and show each independently refined trace partition."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), layout="constrained")
    for row, kind in zip(axes, ("triangles", "L-polygons"), strict=True):
        for ell in (0, 1):
            rows = [
                r
                for r in record["rows"]
                if r["study"] == "skeleton" and r["mesh"] == kind and r["trace_degree"] == ell
            ]
            x = np.array([r["trace_segments"] for r in rows])
            for axis, key, title in zip(
                row, ("pressure_l2", "flux_l2"), ("base pressure", "base Darcy flux"), strict=True
            ):
                errors = np.array([r[key] for r in rows])
                rate = np.log(errors[-2] / errors[-1]) / np.log(x[-1] / x[-2])
                axis.loglog(
                    x, errors, "o-", label=rf"$\ell={ell},k={ell + 2}$; final rate {rate:.2f}"
                )
                axis.set(
                    xlabel="Segments per macroface",
                    ylabel="Absolute L² error",
                    title=f"{kind}: {title}",
                )
                axis.grid(True, alpha=0.3)
                axis.legend(fontsize=12)
                set_refinement_ticks(axis, x)
    save(figure, output, "skeleton-convergence")


def fields(results: Path, output: Path, kind: str) -> None:
    """Preserve all one-sided samples, macro edges and signed common color scales."""
    data = read_archive(results / f"{kind}-fields.npz")
    mesh = SimpleNamespace(points=data["macro_points"], faces=data["macro_faces"])
    points, cells = data["points"], data["cells"]
    truth_p, truth_q = exact(points), exact_flux(points)
    figure, axes = plt.subplots(3, 4, figsize=(16, 12.6), layout="constrained")
    for index, (truth, base, enriched, name, symbol) in enumerate(
        (
            (truth_p, data["values"], data["enriched_pressure"], "pressure", "p"),
            (truth_q[:, 0], data["flux"][:, 0], data["enriched_flux"][:, 0], "flux x", "q_x"),
            (truth_q[:, 1], data["flux"][:, 1], data["enriched_flux"][:, 1], "flux y", "q_y"),
        )
    ):
        bound = float(max(np.max(abs(truth)), np.max(abs(base)), np.max(abs(enriched))))
        difference = enriched - truth
        error_bound = float(np.max(abs(difference)))
        for column, (values, title, extent, label) in enumerate(
            (
                (truth, f"Exact {name}", bound, f"${symbol}$"),
                (base, f"Base {name}", bound, rf"${symbol}^{{\rm PG}}$"),
                (enriched, f"Enriched {name}", bound, rf"$\widetilde{{{symbol}}}$"),
                (
                    difference,
                    "Enriched − exact",
                    error_bound,
                    rf"$\widetilde{{{symbol}}}-{symbol}$",
                ),
            )
        ):
            panel(
                figure,
                axes[index, column],
                points,
                cells,
                values,
                mesh,
                title,
                label,
                (-extent, extent),
                signed=True,
            )
    for axis in figure.axes:
        axis.tick_params(labelsize=16)
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(16)
        axis.title.set_size(16)
    figure.suptitle(f"{kind}, n=8; local P3 / trace P1; actual macro edges", fontsize=17)
    save(figure, output, f"{kind}-fields")


def main() -> None:
    """Render only archived numerical data; no solver is executed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads(read_resource_text(args.results / "comparison.json"))
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15})
    for kind in ("triangles", "L-polygons"):
        macro_convergence(record, args.output, kind)
        fields(args.results, args.output, kind)
    skeleton_convergence(record, args.output)
    shutil.copyfile(args.results / "comparison.json", args.output / "comparison.json")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_pgmhm").main()
