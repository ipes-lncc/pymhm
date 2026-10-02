"""Render broken RAD fields on physical tetrahedral cross-sections without interface smoothing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from plot_darcy3d import slice_polygon
from solve_rad3d import exact, physical_flux

from pymhm.tetrahedral import TetraMesh, tetra_basis, tetra_nodal_space

ROOT = Path(__file__).resolve().parents[1]


def slice_fields(mesh: TetraMesh, coefficients: np.ndarray, degree: int, height: float) -> tuple:
    """Evaluate each intersected fine tetrahedron on its own side at its polygon centroid."""
    dofs, _ = tetra_nodal_space(mesh, degree)
    polygons, ids, points, inverse = [], [], [], []
    for cell, vertices in enumerate(mesh.points[mesh.cells]):
        polygon = slice_polygon(vertices, height)
        if not len(polygon):
            continue
        polygons.append(polygon[:, :2])
        ids.append(cell)
        points.append(polygon.mean(axis=0))
        inverse.append(np.linalg.inv(np.column_stack((np.ones(4), vertices))))
    if not ids:
        return [], np.empty((0, 2)), np.empty((0, 2))
    points, inverse = np.asarray(points), np.asarray(inverse)
    bary = np.einsum("ti,tij->tj", np.column_stack((np.ones(len(points)), points)), inverse)
    basis, derivative = tetra_basis(degree, bary)
    local = coefficients[dofs[ids]]
    values = np.einsum("ti,ti->t", local, basis)
    gradient = np.einsum("ti,tij,taj->ta", local, derivative, inverse[:, 1:])
    flux = -0.1 * gradient
    flux[:, 0] += values
    return (
        polygons,
        np.column_stack((values, np.linalg.norm(flux, axis=1))),
        np.column_stack((exact(points), np.linalg.norm(physical_flux(points), axis=1))),
    )


def main() -> None:
    """Plot the archived classical/MHM fields with matching physical ranges and macro geometry."""
    report = json.loads((ROOT / "examples/results/rad3d.json").read_text())
    last = report["rows"][-1]
    archive = ROOT / "examples/results" / last["fields"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != last["fields_sha256"]:
        raise ValueError("field archive digest mismatch")
    data = np.load(archive)
    height = 0.25
    edges = []
    for vertices in data["macro_points"][data["macro_cells"]]:
        polygon = slice_polygon(vertices, height)
        if len(polygon):
            edges.extend(zip(polygon[:, :2], np.roll(polygon[:, :2], -1, axis=0), strict=True))
    polygons, numerical, expected = [], [], []
    for points, cells, coefficients in zip(
        data["local_points"], data["local_cells"], data["values"], strict=True
    ):
        poly, actual, reference = slice_fields(TetraMesh(points, cells), coefficients, 4, height)
        polygons.extend(poly)
        numerical.extend(actual)
        expected.extend(reference)
    mhm = polygons, np.asarray(numerical), np.asarray(expected)
    classical = slice_fields(
        TetraMesh(data["classical_points"], data["classical_cells"]),
        data["classical_values"],
        2,
        height,
    )
    output = ROOT / "docs/figures/rad3d"
    output.mkdir(parents=True, exist_ok=True)
    for component, label, name in (
        (0, "Scalar field", "scalar"),
        (1, "Physical flux magnitude", "flux"),
    ):
        maximum = max(
            np.max(np.abs(item[i][:, component])) for item in (mhm, classical) for i in (1, 2)
        )
        norm = (
            TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
            if component == 0
            else Normalize(0, maximum)
        )
        error_bound = max(
            np.max(np.abs(item[1][:, component] - item[2][:, component]))
            for item in (mhm, classical)
        )
        difference_norm = TwoSlopeNorm(vmin=-error_bound, vcenter=0, vmax=error_bound)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), layout="constrained")
        for row, (poly, actual, reference) in enumerate((mhm, classical)):
            method = ("MHM P4/P1", "Classical P2")[row]
            for column, values in enumerate(
                (
                    reference[:, component],
                    actual[:, component],
                    actual[:, component] - reference[:, component],
                )
            ):
                axis = axes[row, column]
                artist = PolyCollection(
                    poly,
                    array=values,
                    edgecolors="none",
                    norm=norm if column < 2 else difference_norm,
                    cmap="RdBu_r" if component == 0 or column == 2 else "viridis",
                )
                axis.add_collection(artist)
                axis.add_collection(LineCollection(edges, colors="white", linewidths=1.6, zorder=3))
                axis.add_collection(
                    LineCollection(edges, colors="black", linewidths=0.55, zorder=4)
                )
                axis.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    aspect="equal",
                    xlabel="x",
                    ylabel="y",
                    title=("Exact", method, f"{method} − exact")[column],
                )
                fig.colorbar(artist, ax=axis, shrink=0.84, pad=0.03)
        fig.suptitle(f"{label} · z = 1/4 · actual MHM macro intersections", fontsize=17)
        for ext in ("png", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=170)
        plt.close(fig)
    spacing = 1 / np.array([row["n"] for row in report["rows"]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for axis, key, label in zip(
        axes,
        ("l2_error", "V_error", "physical_flux_l2_error"),
        ("Scalar L² error", "Broken V error", "Physical flux L² error"),
        strict=True,
    ):
        for method, style in (("mhm", "o-"), ("classical", "s--")):
            axis.loglog(
                spacing,
                [row[method][key] for row in report["rows"]],
                style,
                label="MHM P4 / P1 faces" if method == "mhm" else "Classical P2",
            )
        axis.set(xlabel="MHM macro spacing 1/n", ylabel=label)
        axis.set_xticks(spacing, labels=["1", "1/2", "1/3", "1/4", "1/5"])
        axis.minorticks_off()
        axis.grid(alpha=0.25)
        axis.legend(fontsize=10)
    for ext in ("png", "svg"):
        fig.savefig(output / f"convergence.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
