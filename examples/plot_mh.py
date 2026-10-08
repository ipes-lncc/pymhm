"""Render measured Robin-MH convergence and physical fields from archived data."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples.mh_campaign import exact, exact_flux
from examples.plot_mh2m import panel, read_archive, save
from examples.plot_style import set_refinement_ticks

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/results/mh"
OUTPUT = ROOT / "docs/figures/mh"


def convergence(record: dict[str, Any], output: Path) -> None:
    """Separate mesh families and physical norms, displaying all measured resolutions."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9.2), layout="constrained")
    for row, kind in zip(axes, ("triangles", "L-polygons"), strict=True):
        for ell in (1, 2):
            data = [r for r in record["smooth"] if r["mesh"] == kind and r["trace_degree"] == ell]
            h = np.array([r["macro_diameter"] for r in data])
            for axis, key in zip(
                row, ("flux_relative_error", "pressure_relative_error"), strict=True
            ):
                errors = np.array([r[key] for r in data])
                rate = np.log(errors[-2] / errors[-1]) / np.log(h[-2] / h[-1])
                axis.loglog(
                    h, errors, "o-", label=rf"$\ell={ell},k={ell + 2}$; final rate {rate:.2f}"
                )
                axis.set(xlabel="Macro diameter $H$", ylabel="Relative L² error")
                axis.grid(True, alpha=0.3)
                axis.legend(fontsize=11)
                set_refinement_ticks(axis, h, [f"{value:.3g}" for value in h])
        row[0].set_title(f"{kind}: physical Darcy flux")
        row[1].set_title(f"{kind}: pressure")
    save(figure, output, "convergence")


def vanishing_robin(record: dict[str, Any], output: Path) -> None:
    """Show the same-space energy limit and separately measured spectral conditioning."""
    rows = record["vanishing_robin"]["rows"]
    parameter = np.array([row["robin_parameter"] for row in rows])
    error = np.array([row["energy_difference"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].loglog(parameter, error, "o-", label="Measured broken energy difference")
    axes[0].loglog(parameter, error[0] * parameter / parameter[0], "--", label=r"$O(\nu)$ guide")
    axes[0].set(ylabel=r"$\|\nabla(p_{\rm MH}-p_{\rm MHM})\|_{L^2}$", title="Same discrete spaces")
    for key, label in (
        ("global_condition_2", "Global Robin-trace operator"),
        ("first_local_condition_2", "First local Robin operator"),
    ):
        axes[1].loglog(parameter, [row[key] for row in rows], "o-", label=label)
    axes[1].set(ylabel="Spectral condition number", title="Measured conditioning")
    for axis in axes:
        axis.set_xlabel(r"Robin parameter $\nu$")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)
        set_refinement_ticks(axis, parameter)
    save(figure, output, "vanishing-robin")


def fields(results: Path, output: Path, kind: str) -> None:
    """Keep macro edges and one-sided signed pressure/flux fields in every panel."""
    data = read_archive(results / f"{kind}-ell2.npz")
    mesh = SimpleNamespace(points=data["macro_points"], faces=data["macro_faces"])
    points, cells = data["points"], data["cells"]
    exact_values = exact(points)
    exact_vector = exact_flux(points)
    figure, axes = plt.subplots(3, 3, figsize=(12, 12.6), layout="constrained")
    for index, (truth, computed, name, symbol) in enumerate(
        (
            (exact_values, data["values"], "pressure", "p"),
            (exact_vector[:, 0], data["flux"][:, 0], "Darcy flux x", "q_x"),
            (exact_vector[:, 1], data["flux"][:, 1], "Darcy flux y", "q_y"),
        )
    ):
        extent = float(max(np.max(abs(truth)), np.max(abs(computed))))
        error = computed - truth
        error_extent = float(np.max(abs(error)))
        for column, (values, title, bound, label) in enumerate(
            (
                (truth, f"Exact {name}", extent, f"${symbol}$"),
                (computed, f"MH {name}", extent, rf"${symbol}^{{\mathrm{{MH}}}}$"),
                (error, "Signed error", error_extent, rf"${symbol}^{{\mathrm{{MH}}}}-{symbol}$"),
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
                (-bound, bound),
                signed=True,
            )
    figure.suptitle(f"{kind}, resolution 8; local P4 / trace P2; actual macro edges")
    save(figure, output, f"{kind}-fields")


def main() -> None:
    """Regenerate publication figures without solving any finite-element systems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads((args.results / "comparison.json").read_text())
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    convergence(record, args.output)
    vanishing_robin(record, args.output)
    for kind in ("triangles", "L-polygons"):
        fields(args.results, args.output, kind)
    shutil.copy2(args.results / "comparison.json", args.output / "comparison.json")


if __name__ == "__main__":
    main()
