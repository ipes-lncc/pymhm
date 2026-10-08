"""Plot admissible P5/P2 convergence and broken fields from archived tetrahedral coefficients."""

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from threadpoolctl import threadpool_limits

from examples.plot_style import set_refinement_ticks
from examples.reconstruction3d_data import fields
from examples.reconstruction3d_replay import evaluate, profile
from examples.tetra_section_samples import section_grid
from pymhm import TetraMesh
from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)

ROOT = case_workspace()
DATA = ROOT / "examples/results/tetra-pk"
OUTPUT = ROOT / "docs/figures/tetra-pk"


def save(figure: plt.Figure, name: str) -> None:
    """Export both image formats using separate fields, color scales and caption regions."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=230, bbox_inches="tight")
    plt.close(figure)


def archive_path(row: dict[str, Any], directory: Path = DATA) -> Path:
    """Resolve an archive only after matching its published acquisition digest."""
    path = directory / row["archive"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != row["archive_sha256"]:
        raise ValueError("archive differs from its recorded digest")
    return path


def convergence(record: dict[str, Any]) -> None:
    """Show five physical error measurements and the measured energy-indicator ratio."""
    rows = record["rows"]
    size = [row["macro_cells"] for row in rows]
    figure = plt.figure(figsize=(11, 5.7), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 0.25])
    axes = [figure.add_subplot(grid[0, i]) for i in range(2)]
    for key, label in (
        ("pressure_l2", "Pressure L²"),
        ("raw_flux_l2", "Raw flux L²"),
        ("rt_flux_l2", "RT2 flux L²"),
        ("indicator", "Energy indicator η"),
    ):
        axes[0].loglog(size, [row[key] for row in rows], "o-", label=label)
    axes[1].semilogx(size, [row["effectivity"] for row in rows], "o-", label="η / energy error")
    axes[1].axhline(1, color="black", ls=":", label="Unit ratio")
    for i, axis in enumerate(axes):
        axis.set(xlabel="Macrotetrahedra", ylabel="Absolute norm" if i == 0 else "Ratio")
        set_refinement_ticks(axis, size)
        axis.grid(alpha=0.2)
        legend = figure.add_subplot(grid[1, i])
        legend.set_axis_off()
        legend.legend(*axis.get_legend_handles_labels(), loc="center", frameon=False, ncol=2)
    figure.suptitle("Localized Gaussian · local P5 / skeletal P2 / RT2\nFive uniform resolutions")
    save(figure, "uniform-convergence")


def fixed_comparison(record: dict[str, Any]) -> None:
    """Compare local P4/P5 at unchanged macro/fine geometry and two face partitions."""
    previous = json.loads(
        read_resource_text(ROOT / "examples/results/reconstruction3d/resolution.json")
    )
    rows = [row for row in previous["rows"] if row["trace_degree"] == 2] + record["rows"]
    figure = plt.figure(figsize=(11, 5.8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 0.24])
    for i, (keys, ylabel) in enumerate(
        (
            ((("pressure_relative", "Pressure L²"),), "Relative pressure error (%)"),
            (
                (("raw_flux_relative", "Raw flux L²"), ("rt_flux_relative", "RT2 flux L²")),
                "Relative flux error (%)",
            ),
        )
    ):
        axis = figure.add_subplot(grid[0, i])
        positions = np.arange(len(rows))
        width = 0.65 / len(keys)
        for j, (key, label) in enumerate(keys):
            axis.bar(
                positions + (j - (len(keys) - 1) / 2) * width,
                [100 * row[key] for row in rows],
                width=width,
                label=label,
            )
        axis.set_xticks(
            positions, labels=[f"P{r['local_degree']}\ns={r['trace_subdivisions']}" for r in rows]
        )
        axis.set(ylabel=ylabel, xlabel="Local degree and subdivisions per face edge")
        axis.grid(axis="y", alpha=0.2)
        legend = figure.add_subplot(grid[1, i])
        legend.set_axis_off()
        legend.legend(*axis.get_legend_handles_labels(), loc="center", frameon=False, ncol=2)
    figure.suptitle(
        "Fixed 162 macros / 1296 fine tetrahedra · skeletal P2\n"
        "P5/P2 satisfies k = ℓ + 3; P4/P2 is a physical-error control"
    )
    save(figure, "fixed-resolution")


def section(row: dict[str, Any]) -> None:
    """Display full polynomials on disconnected fine-cell sections, preserving every interface."""
    rt_degree = row["reconstruction_degree"]
    with np.load(local_resource(archive_path(row))) as archive:
        macro = TetraMesh(archive["macro_points"], archive["macro_cells"])
        edges = section_grid(macro, refinement=1)["segments"]
        points, cells, numerical = [], [], []
        offset = 0
        for cell in range(len(macro.cells)):
            fine = TetraMesh(archive[f"points_{cell}"], archive[f"cells_{cell}"])
            cut = section_grid(fine, refinement=6)
            if not len(cut["points"]):
                continue
            points.append(cut["points"])
            cells.append(cut["cells"] + offset)
            numerical.append(evaluate(archive, cell, cut["parents"], cut["barycentric"]))
            offset += len(cut["points"])
    xyz, connectivity, values = (
        np.concatenate(points),
        np.concatenate(cells),
        np.concatenate(numerical),
    )
    p, q, _ = fields(xyz, True)
    exact = np.column_stack((p, q))
    tri = mtri.Triangulation(xyz[:, 0], xyz[:, 1], connectivity)
    figure = plt.figure(figsize=(13, 18), layout="constrained")
    grid = figure.add_gridspec(8, 3, height_ratios=[1, 0.06] * 4, hspace=0.12)
    for component, label in enumerate(("Pressure", "Flux qₓ", "Flux qᵧ", "Flux q_z")):
        difference = values[:, component] - exact[:, component]
        limit = max(abs(values[:, component]).max(), abs(exact[:, component]).max())
        error_limit = max(abs(difference).max(), np.finfo(float).eps)
        for column, (data, name) in enumerate(
            (
                (exact[:, component], "Exact"),
                (values[:, component], "P5" if component == 0 else f"RT{rt_degree}"),
                (difference, "Difference"),
            )
        ):
            axis = figure.add_subplot(grid[component * 2, column])
            lower, upper = (
                (-error_limit, error_limit)
                if column == 2
                else ((-limit if component else min(values[:, 0].min(), exact[:, 0].min())), limit)
            )
            artist = axis.tripcolor(
                tri,
                data,
                shading="gouraud",
                rasterized=True,
                cmap="RdBu_r" if component or column == 2 else "viridis",
                vmin=lower,
                vmax=upper,
            )
            axis.add_collection(LineCollection(edges, colors="white", linewidths=1.0, alpha=0.75))
            axis.add_collection(LineCollection(edges, colors="0.18", linewidths=0.5, alpha=0.8))
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=f"{name} · {label}",
            )
            colorbar = figure.colorbar(
                artist,
                cax=figure.add_subplot(grid[component * 2 + 1, column]),
                orientation="horizontal",
            )
            colorbar.locator = MaxNLocator(3)
            colorbar.update_ticks()
    figure.suptitle(
        f"P5 / skeletal P2 / RT{rt_degree} · four subtriangles per macroface\n"
        "Section z = 0.37 · actual macro intersections · independent fine-cell values"
    )
    save(figure, f"fixed-components-rt{rt_degree}")


def profiles(record: dict[str, Any], recovered: dict[str, Any]) -> None:
    """Show independent P4/P5 and RT2/RT3 traces with actual macro intersections."""
    previous_dir = ROOT / "examples/results/reconstruction3d"
    previous = json.loads(read_resource_text(previous_dir / "resolution.json"))["rows"][-1]
    first, last = np.array([0.0, 0.413, 0.37]), np.array([1.0, 0.413, 0.37])
    samples = []
    for directory, row in ((previous_dir, previous), (DATA, record["rows"][-1]), (DATA, recovered)):
        with np.load(local_resource(archive_path(row, directory))) as archive:
            samples.append(profile(archive, first, last))
    parameter = np.linspace(0, 1, 501)
    p, q, _ = fields(first + parameter[:, None] * (last - first), True)
    exact = np.column_stack((p, q))
    figure = plt.figure(figsize=(10, 13), layout="constrained")
    grid = figure.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.22])
    axes = []
    for component, label in enumerate(("Pressure", "Flux qₓ", "Flux qᵧ", "Flux q_z")):
        axis = figure.add_subplot(grid[component, 0])
        axes.append(axis)
        axis.plot(parameter, exact[:, component], "k-", label="Exact", linewidth=1.5)
        for model_label, sample, color in zip(
            ("P4 / P2 / RT2", "P5 / P2 / RT2", "P5 / P2 / RT3"),
            samples,
            ("#cc6677", "#0072b2", "#009e73"),
            strict=True,
        ):
            for segment, t in enumerate(sample["parameter"]):
                axis.plot(
                    t,
                    sample["actual"][segment, :, component],
                    color=color,
                    label=model_label if segment == 0 else None,
                )
        for i, position in enumerate(samples[-1]["macro_breaks"][1:-1]):
            axis.axvline(
                position,
                color="0.55",
                linestyle=":",
                linewidth=0.65,
                label="Macro interface" if i == 0 else None,
            )
        axis.set(xlabel="x", ylabel=label, xlim=(0, 1))
        axis.grid(alpha=0.15)
    legend = figure.add_subplot(grid[4, 0])
    legend.set_axis_off()
    legend.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=3, frameon=False)
    figure.suptitle(
        "Fixed geometry · y = 0.413, z = 0.37\n"
        "Four subtriangles per macroface; independent one-sided polynomial traces"
    )
    save(figure, "fixed-profiles")


def run() -> None:
    """Render finalized records only, without assembling or resolving a finite element system."""
    uniform = json.loads(read_resource_text(DATA / "uniform.json"))
    fixed = json.loads(read_resource_text(DATA / "fixed.json"))
    if len(uniform["rows"]) != 5 or len(fixed["rows"]) != 2:
        raise ValueError(
            "the plotting campaign requires five uniform and two fixed-geometry states"
        )
    convergence(uniform)
    fixed_comparison(fixed)
    recovered = json.loads(read_resource_text(DATA / "reconstruction-order.json"))
    section(fixed["rows"][-1])
    section(recovered)
    profiles(fixed, recovered)


def main() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    with threadpool_limits(1):
        run()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_tetra_pk").main()
