"""Render certified nonconvex macrogeometry and archived broken tetrahedral RAD fields."""

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from threadpoolctl import threadpool_limits

from examples.plot_mesh import draw_macro_mesh
from examples.plot_style import set_refinement_ticks
from examples.polygon_meshes import polygon_partition
from examples.solve_rad3d import exact, physical_flux
from examples.tetra_section_samples import section_grid
from pymhm import PolygonMesh, PolyhedralMesh
from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/star-polyhedra"
OUTPUT = ROOT / "docs/figures/star-polyhedra"


def save(figure: plt.Figure, name: str) -> None:
    """Write vector/raster pairs with separately allocated field, colorbar and label regions."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def geometry() -> None:
    """Show the preserved reentrant boundary, kernel center and complete macro tiling."""
    partition = polygon_partition(1, "L")
    vertices = partition.points[partition.cells[0]]
    base = PolygonMesh(vertices, (np.arange(len(vertices)),))
    mesh = PolyhedralMesh.extrude(base, 1)
    figure = plt.figure(figsize=(11, 5), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 0.13))
    axis = figure.add_subplot(grid[0, 0], projection="3d")
    axis.add_collection3d(
        Poly3DCollection(
            [mesh.points[ids] for ids in mesh.faces],
            facecolor="#90c3df",
            edgecolor="#23465b",
            alpha=0.38,
            linewidth=1.1,
        )
    )
    axis.scatter(*mesh.centers[0], color="#b2182b", s=30)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        zlim=(0, 1),
        xlabel="x",
        ylabel="y",
        title="Star-shaped L prism · original faces",
    )
    axis.set_box_aspect((1, 1, 1), zoom=0.82)
    axis.view_init(elev=31, azim=-58)
    axis.set_xticks((0, 0.5, 1))
    axis.set_yticks((0.5, 1))
    axis.set_zticks((0, 0.5, 1))
    axis.text2D(0.95, 0.55, "z", transform=axis.transAxes)
    caption = figure.add_subplot(grid[1, 0])
    caption.set_axis_off()
    caption.text(
        0.5,
        0.5,
        "Red point: certified interior kernel center; the notch remains empty",
        ha="center",
        va="center",
        fontsize=9,
    )
    axis = figure.add_subplot(grid[0, 1])
    tiling = polygon_partition(2, "L")
    for cell, ids in enumerate(tiling.cells):
        points = tiling.points[ids]
        axis.fill(*points.T, color="#9ecae1" if cell % 2 == 0 else "#fdd49e", alpha=0.6)
    draw_macro_mesh(axis, tiling)
    axis.set(
        xlabel="x",
        ylabel="y",
        aspect="equal",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Horizontal section · n = 2 macro tiling",
    )
    caption = figure.add_subplot(grid[1, 1])
    caption.set_axis_off()
    caption.text(
        0.5,
        0.5,
        "L prisms and cuboids tile the unit cube without gaps or filled cavities",
        ha="center",
        va="center",
        fontsize=9,
    )
    save(figure, "geometry")


def convergence(record: dict[str, Any]) -> None:
    """Plot every measured resolution with exact-field norms and the geometric certificate."""
    rows = record["rows"]
    figure = plt.figure(figsize=(11, 5.5), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 0.18))
    axis = figure.add_subplot(grid[0, 0])
    for key, label in (
        ("l2_error", "Scalar L²"),
        ("V_error", "V norm"),
        ("physical_flux_l2_error", "Physical flux L²"),
    ):
        axis.loglog([r["macro_cells"] for r in rows], [r[key] for r in rows], "o-", label=label)
    axis.set(xlabel="Polyhedral macrocells", ylabel="Absolute error")
    set_refinement_ticks(axis, [r["macro_cells"] for r in rows])
    axis.grid(alpha=0.2)
    legend = figure.add_subplot(grid[1, 0])
    legend.set_axis_off()
    legend.legend(*axis.get_legend_handles_labels(), loc="center", ncol=3, frameon=False)
    axis = figure.add_subplot(grid[0, 1])
    axis.plot([r["n"] for r in rows], [r["kernel_radius_over_diameter_min"] for r in rows], "o-")
    axis.set(xlabel="Macro subdivisions per coordinate", ylabel="Minimum kernel radius / diameter")
    axis.set_ylim(0, 0.2)
    axis.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    axis.set_xticks([r["n"] for r in rows])
    axis.grid(alpha=0.2)
    figure.suptitle(
        "Conservative RAD · P4 local / P1 per original polygonal face\n"
        "Nonconvex star-shaped L-prism / cuboid family"
    )
    save(figure, "convergence")


def fields(row: dict[str, Any]) -> None:
    """Evaluate full P4 scalar and physical flux on disconnected fine-cell sections."""
    path = DATA / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("archived fields differ from their acquisition digest")
    points, cells, values = [], [], []
    offset = 0
    with np.load(path) as archive:
        for cell in range(row["macro_cells"]):
            fine = TetraMesh(archive[f"points_{cell}"], archive[f"cells_{cell}"])
            cut = section_grid(fine, refinement=5)
            if not len(cut["points"]):
                continue
            dofs, _ = tetra_nodal_space(fine, 4)
            coefficient = archive[f"coefficients_{cell}"][dofs[cut["parents"]]]
            basis, derivative = tetra_basis(4, cut["barycentric"])
            scalar = np.einsum("qi,qi->q", basis, coefficient)
            jacobian = (fine.points[fine.cells[:, 1:]] - fine.points[fine.cells[:, :1]]).transpose(
                0, 2, 1
            )
            inverse = np.linalg.inv(jacobian)
            bary_gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
            gradient = np.einsum(
                "qia,qi,qab->qb",
                derivative,
                coefficient,
                bary_gradients[cut["parents"]],
            )
            flux = -0.1 * gradient
            flux[:, 0] += scalar
            points.append(cut["points"])
            cells.append(cut["cells"] + offset)
            values.append(np.column_stack((scalar, flux)))
            offset += len(scalar)
    xyz, connectivity, numerical = (
        np.concatenate(points),
        np.concatenate(cells),
        np.concatenate(values),
    )
    analytical = np.column_stack((exact(xyz), physical_flux(xyz)))
    triangulation = mtri.Triangulation(xyz[:, 0], xyz[:, 1], connectivity)
    base = polygon_partition(row["n"], "L")
    figure = plt.figure(figsize=(13, 18), layout="constrained")
    grid = figure.add_gridspec(8, 3, height_ratios=[1, 0.06] * 4)
    for component, label in enumerate(
        ("Scalar u", "Physical flux qₓ", "Physical flux qᵧ", "Physical flux q_z")
    ):
        limit = float(max(abs(analytical[:, component]).max(), abs(numerical[:, component]).max()))
        difference = numerical[:, component] - analytical[:, component]
        delta = float(max(abs(difference).max(), np.finfo(float).eps))
        for column, (data, name) in enumerate(
            (
                (analytical[:, component], "Exact"),
                (numerical[:, component], "P4/P1"),
                (difference, "Difference"),
            )
        ):
            axis = figure.add_subplot(grid[2 * component, column])
            bound = delta if column == 2 else limit
            artist = axis.tripcolor(
                triangulation,
                data,
                shading="gouraud",
                rasterized=True,
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            draw_macro_mesh(axis, base)
            axis.set(
                aspect="equal",
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="x",
                ylabel="y",
                title=f"{name} · {label}",
            )
            colorbar = figure.colorbar(
                artist,
                cax=figure.add_subplot(grid[2 * component + 1, column]),
                orientation="horizontal",
            )
            colorbar.locator = MaxNLocator(3)
            colorbar.update_ticks()
    figure.suptitle(
        f"Nonconvex polyhedral RAD · {row['macro_cells']} macrocells\n"
        "Section z = 0.37 · actual macro boundaries · independent fine-cell values"
    )
    save(figure, "fields")


def run() -> None:
    """Render only complete, digest-verified records without another PDE solve."""
    record = json.loads((DATA / "comparison.json").read_text())
    if len(record["rows"]) != 5:
        raise ValueError("five completed mesh levels are required")
    geometry()
    convergence(record)
    fields(record["rows"][-1])


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
