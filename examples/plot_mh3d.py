"""Render archived MH/MH2M convergence and one-sided three-dimensional sections."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from examples.mh3d_campaign import flux, pressure
from examples.plot_darcy3d import slice_polygon
from examples.plot_style import set_refinement_ticks
from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]


def fields(archive: Path, row: dict, output: Path) -> None:
    """Sample the archived polynomial independently on each actual fine-cell cut."""
    if hashlib.sha256(archive.read_bytes()).hexdigest() != row["fields_sha256"]:
        raise ValueError("MH3D field archive digest mismatch")
    with np.load(archive) as stored:
        data = {key: stored[key] for key in stored.files}
    height, degree = 0.375, int(data["degree"])
    polygons, actual, exact, lines = [], [], [], []
    for cell, vertices in enumerate(data["macro_points"][data["macro_cells"]]):
        macro = slice_polygon(vertices, height)
        if not len(macro):
            continue
        lines.extend(zip(macro[:, :2], np.roll(macro[:, :2], -1, axis=0), strict=True))
        mesh = TetraMesh(data["local_points"][cell], data["local_cells"][cell])
        dofs, _ = tetra_nodal_space(mesh, degree)
        for index, vertices in enumerate(mesh.points[mesh.cells]):
            polygon = slice_polygon(vertices, height)
            if not len(polygon):
                continue
            point = polygon.mean(axis=0)
            inverse = np.linalg.inv(np.column_stack((np.ones(4), vertices)))
            bary = (np.r_[1.0, point] @ inverse)[None]
            basis, derivative = tetra_basis(degree, bary)
            coefficients = data["pressure"][cell, dofs[index]]
            value = basis[0] @ coefficients
            vector = -coefficients @ derivative[0] @ inverse[1:].T
            polygons.append(polygon[:, :2])
            actual.append((value, vector[0]))
            exact.append((pressure(point[None])[0], flux(point[None])[0, 0]))
    actual, exact = np.asarray(actual, dtype=float), np.asarray(exact)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.8), layout="constrained")
    method = "MH²M" if row["method"] == "MH2M" else "MH"
    for index, quantity in enumerate(("Pressure p", "Signed flux component qₓ")):
        lower, upper = (
            min(actual[:, index].min(), exact[:, index].min()),
            max(actual[:, index].max(), exact[:, index].max()),
        )
        common = Normalize(lower, upper)
        if index == 1:
            maximum = max(abs(lower), abs(upper))
            common = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
        delta = actual[:, index] - exact[:, index]
        bound = max(float(np.max(abs(delta))), np.finfo(float).tiny)
        for column, values in enumerate((exact[:, index], actual[:, index], delta)):
            axis = axes[index, column]
            artist = PolyCollection(
                polygons,
                array=values,
                norm=common if column < 2 else TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                cmap="viridis" if index == 0 and column < 2 else "RdBu_r",
                edgecolors="none",
                rasterized=True,
            )
            axis.add_collection(artist)
            axis.add_collection(LineCollection(lines, colors="white", linewidths=1.25, zorder=3))
            axis.add_collection(LineCollection(lines, colors="black", linewidths=0.4, zorder=4))
            label = ("Analytical", method, f"{method} − analytical")[column]
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=f"{label}\n{quantity}",
            )
            fig.colorbar(artist, ax=axis, pad=0.025, shrink=0.8, format="%.3g")
    fig.suptitle(
        f"{method}: section z = 3/8, n = {row['resolution']}\n"
        "One-sided fine-cell samples; actual macroface intersections",
        fontsize=16,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"{row['method'].lower()}-fields.{extension}", dpi=180)
    plt.close(fig)


def render(record: Path, output: Path) -> None:
    """Render convergence and the finest saved field from each independent method."""
    data = json.loads(record.read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), layout="constrained")
    for method in ("MH", "MH2M"):
        rows = [row for row in data["rows"] if row["method"] == method]
        n = np.array([row["resolution"] for row in rows])
        label = "MH: P3 local / P1 Robin" if method == "MH" else "MH²M: P2 / ΓP2 / ΛP1"
        for axis, key in zip(axes, ("pressure_relative", "flux_relative"), strict=True):
            axis.loglog(n, [row[key] for row in rows], "o-", label=label)
            set_refinement_ticks(axis, n)
        finest = max(rows, key=lambda row: row["resolution"])
        fields(record.parent / finest["fields"], finest, output)
    for axis, title in zip(axes, ("Pressure", "Physical Darcy flux"), strict=True):
        axis.set(xlabel="Cartesian macro resolution n", ylabel="Relative L² error", title=title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=10, loc="lower left")
    fig.suptitle("Original smooth three-dimensional verification; K = I")
    for extension in ("png", "svg"):
        fig.savefig(output / f"convergence.{extension}", dpi=180)
    plt.close(fig)
    shutil.copy2(record, output / record.name)


def main() -> None:
    """Parse the frozen record and figure destination without resolving the PDE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record", type=Path, default=ROOT / "examples/results/mh3d/comparison.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/mh3d")
    args = parser.parse_args()
    render(args.record, args.output)


if __name__ == "__main__":
    main()
