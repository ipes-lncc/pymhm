"""Render MH2M convergence and broken physical fields from archived numerical data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from examples.mh2m_campaign import _owners, exact, oscillatory
from examples.plot_mesh import draw_macro_mesh
from examples.plot_style import set_refinement_ticks
from pymhm.mesh import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/results/mh2m"
OUTPUT = ROOT / "docs/figures/mh2m"


def read_archive(path: Path) -> dict[str, np.ndarray]:
    """Read each compressed array once, retaining independent interface values."""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def save(figure: Any, output: Path, name: str) -> None:
    """Export raster fields with vector text/axes and a publication-resolution PNG."""
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def panel(
    figure: Any,
    axis: Any,
    points: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    mesh: TriangleMesh,
    title: str,
    label: str,
    limits: tuple[float, float],
    *,
    signed: bool = False,
) -> None:
    """Draw one independent broken field and reserve a separate horizontal scale."""
    triangles = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    artist = axis.tripcolor(
        triangles,
        values,
        shading="gouraud",
        cmap="RdBu_r" if signed else "viridis",
        norm=Normalize(*limits),
        rasterized=True,
    )
    draw_macro_mesh(axis, mesh)
    axis.set(xlabel="$x$", ylabel="$y$", title=title, aspect="equal", xlim=(0, 1), ylim=(0, 1))
    bar = figure.colorbar(artist, ax=axis, orientation="horizontal", pad=0.13, fraction=0.055)
    bar.set_label(label)
    bar.locator = MaxNLocator(nbins=3)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=12)


def p1_fields(
    data: dict[str, np.ndarray], vertices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate P1 pressure/raw flux on one side of each target triangle.

    Target cell centroids select the owning cell. Thus vertices on a broken
    interface retain the value of their incident target cell instead of an
    average across that interface.
    """
    target = data["vertices"] if vertices is None else vertices
    owners = (
        np.arange(len(target)) if vertices is None else _owners(data["vertices"], target.mean(1))
    )
    source = data["vertices"][owners]
    inverse = np.linalg.inv((source[:, 1:] - source[:, :1]).swapaxes(1, 2))
    coordinates = np.einsum("tij,tqj->tqi", inverse, target - source[:, :1])
    bary = np.concatenate((1 - coordinates.sum(2, keepdims=True), coordinates), axis=2)
    coefficients = data["pressure"][owners]
    pressure = np.einsum("tqi,ti->tq", bary, coefficients)
    gradient = np.einsum("tji,tj->ti", inverse, coefficients[:, 1:] - coefficients[:, :1])
    points = target.reshape(-1, 2)
    flux = -oscillatory(points)[:, None] * np.repeat(gradient, 3, axis=0)
    cells = np.arange(len(points)).reshape(-1, 3)
    return points, cells, pressure.ravel(), flux


def convergence(record: dict[str, Any], output: Path) -> None:
    """Plot five measured resolutions for each of the three published smooth spaces."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for k in (0, 1, 2):
        rows = [row for row in record["smooth"] if row["k"] == k]
        h = np.array([row["macro_diameter"] for row in rows])
        for axis, key in zip(
            axes, ("gradient_relative_error", "pressure_relative_error"), strict=True
        ):
            errors = np.array([row[key] for row in rows])
            rate = np.log(errors[-2] / errors[-1]) / np.log(h[-2] / h[-1])
            axis.loglog(h, errors, "o-", label=f"$k={k}$; final rate {rate:.2f}")
    for axis, title in zip(axes, ("Broken gradient", "Pressure"), strict=True):
        axis.set(xlabel="Macro diameter $H$", ylabel="Relative error", title=title)
        axis.grid(True, which="major", alpha=0.3)
        axis.legend(fontsize=11)
        set_refinement_ticks(axis, h, [rf"$\sqrt{{2}}/{row['resolution']}$" for row in rows])
    save(figure, output, "convergence")


def enrichment(record: dict[str, Any], output: Path) -> None:
    """Compare fixed-Gamma enrichment against a separately refined conforming baseline."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    rows = record["enrichment"]
    x = [row["flux_segments"] for row in rows]
    for axis, key, title in zip(
        axes,
        ("pressure_relative_difference", "flux_relative_difference"),
        ("Pressure", "Raw physical Darcy flux"),
        strict=True,
    ):
        axis.loglog(x, [row[key] for row in rows], "o-", label="MH²M / finest P1 reference")
        axis.axhline(
            record["references"][-1][key],
            color="#b0542d",
            linestyle="--",
            label="Last classical refinement change",
        )
        axis.set(
            xlabel="Lambda segments per macroface", ylabel="Relative L² difference", title=title
        )
        set_refinement_ticks(axis, x)
        axis.grid(True, which="major", alpha=0.3)
        axis.legend(fontsize=10)
    figure.suptitle("Fixed Gamma P1: 9 free global unknowns; local P1 resolution unchanged")
    save(figure, output, "enrichment")


def smooth_fields(results: Path, output: Path) -> None:
    """Display exact, computed and signed-error pressure for each local degree."""
    figure, axes = plt.subplots(3, 3, figsize=(12, 12.6), layout="constrained")
    for k, row in enumerate(axes):
        data = read_archive(results / f"smooth-k{k}.npz")
        mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
        pressure = exact(data["points"])
        error = data["values"] - pressure
        extent = float(np.max(abs(error)))
        for j, values in enumerate((pressure, data["values"], error)):
            panel(
                figure,
                row[j],
                data["points"],
                data["cells"],
                values,
                mesh,
                ("Exact pressure", f"MH²M, $k={k}$", "Signed pressure error")[j],
                "$p$" if j < 2 else "$p_h-p$",
                (0, 1 / 16) if j < 2 else (-extent, extent),
                signed=j == 2,
            )
    save(figure, output, "smooth-fields")


def pressure_enrichment(record: dict[str, Any], output: Path) -> None:
    """Show independent Gamma convergence at fixed Lambda and volume resolution."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    rows = record["pressure_enrichment"]
    sizes = [row["free_trace_dofs"] for row in rows]
    for axis, key, title in zip(
        axes,
        ("pressure_relative_difference", "flux_relative_difference"),
        ("Pressure", "Raw physical Darcy flux"),
        strict=True,
    ):
        axis.loglog(sizes, [row[key] for row in rows], "o-", label="MH²M / finest P1 reference")
        axis.axhline(
            record["reference_last_refinement"][key],
            color="#b0542d",
            linestyle="--",
            label="Last classical refinement change",
        )
        axis.set(
            xlabel="Free pressure-trace unknowns", ylabel="Relative L² difference", title=title
        )
        set_refinement_ticks(axis, sizes)
        axis.grid(True, which="major", alpha=0.3)
        axis.legend(fontsize=10)
    save(figure, output, "pressure-trace-enrichment")


def oscillatory_fields(record: dict[str, Any], results: Path, output: Path) -> None:
    """Render pressure and signed raw-flux components on the common fine partition."""
    reference = read_archive(results / f"reference-n{record['references'][-1]['resolution']}.npz")
    coarse = read_archive(results / "enriched-lambda16.npz")
    enriched = read_archive(results / "enriched-gamma16.npz")
    mesh = TriangleMesh(enriched["macro_points"], enriched["macro_cells"])
    points, cells, p_ref, q_ref = p1_fields(reference)
    _, _, p_low, _ = p1_fields(coarse, reference["vertices"])
    _, _, p_high, q_high = p1_fields(enriched, reference["vertices"])
    common = (
        min(p_ref.min(), p_low.min(), p_high.min()),
        max(p_ref.max(), p_low.max(), p_high.max()),
    )
    difference = p_high - p_ref
    extent = float(max(abs(difference)))
    figure, axes = plt.subplots(2, 2, figsize=(10, 9.4), layout="constrained")
    for i, (axis, values, title) in enumerate(
        zip(
            axes.ravel(),
            (p_ref, p_low, p_high, difference),
            (
                "Conforming P1 reference",
                "Gamma: 1 segment",
                "Gamma: 16 segments",
                "MH²M − reference",
            ),
            strict=True,
        )
    ):
        panel(
            figure,
            axis,
            points,
            cells,
            values,
            mesh,
            title,
            "$p$" if i < 3 else "$p_h-p_{ref}$",
            common if i < 3 else (-extent, extent),
            signed=i == 3,
        )
    save(figure, output, "oscillatory-pressure")
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 9.0), layout="constrained")
    for component, row in enumerate(axes):
        limit = float(max(abs(q_ref[:, component]).max(), abs(q_high[:, component]).max()))
        error = q_high[:, component] - q_ref[:, component]
        error_limit = float(abs(error).max())
        for column, values in enumerate((q_ref[:, component], q_high[:, component], error)):
            label = "$q_x$" if component == 0 else "$q_y$"
            panel(
                figure,
                row[column],
                points,
                cells,
                values,
                mesh,
                ("Conforming P1 reference", "MH²M: 16 Gamma segments", "MH²M − reference")[column],
                label if column < 2 else f"Difference in {label}",
                (-limit, limit) if column < 2 else (-error_limit, error_limit),
                signed=True,
            )
    save(figure, output, "oscillatory-flux")


def main() -> None:
    """Render archived records without solving or modifying numerical fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14})
    record = json.loads((args.results / "comparison.json").read_text())
    convergence(record, args.output)
    enrichment(record, args.output)
    pressure_record = json.loads((args.results / "pressure-enrichment.json").read_text())
    pressure_enrichment(pressure_record, args.output)
    smooth_fields(args.results, args.output)
    oscillatory_fields(record, args.results, args.output)
    shutil.copyfile(args.results / "comparison.json", args.output / "comparison.json")
    shutil.copyfile(
        args.results / "pressure-enrichment.json", args.output / "pressure-enrichment.json"
    )


if __name__ == "__main__":
    main()
