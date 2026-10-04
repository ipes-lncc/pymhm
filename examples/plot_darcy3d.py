"""Render exact/numerical tetrahedral cross-sections and recorded 3D convergence."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]


def slice_polygon(vertices: np.ndarray, height: float) -> np.ndarray:
    """Intersect a tetrahedron with a horizontal plane, keeping its polygon without smoothing."""
    points = []
    for a, b in combinations(range(4), 2):
        za, zb = vertices[a, 2] - height, vertices[b, 2] - height
        if za * zb < 0:
            points.append(vertices[a] + (-za / (zb - za)) * (vertices[b] - vertices[a]))
        elif za == 0:
            points.append(vertices[a])
        elif zb == 0:
            points.append(vertices[b])
    if not points:
        return np.empty((0, 3))
    unique = np.unique(np.round(points, 14), axis=0)
    if len(unique) < 3:
        return np.empty((0, 3))
    center = unique[:, :2].mean(axis=0)
    angle = np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
    return unique[np.argsort(angle)]


def exact(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exact sinusoidal potential and physical Darcy flux."""
    s, c = np.sin(2 * np.pi * points), np.cos(2 * np.pi * points)
    p = s.prod(axis=1)
    q = (
        -2
        * np.pi
        * np.column_stack(
            (c[:, 0] * s[:, 1] * s[:, 2], s[:, 0] * c[:, 1] * s[:, 2], s[:, 0] * s[:, 1] * c[:, 2])
        )
    )
    return p, q


def main() -> None:
    """Plot archived fields with actual macro intersections and independent one-sided values."""
    report = json.loads((ROOT / "examples/results/darcy3d.json").read_text())
    finest = report["rows"][-1]
    path = ROOT / "examples/results" / finest["fields"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != finest["fields_sha256"]:
        raise ValueError("field archive digest mismatch")
    data = np.load(path)
    height = 0.375
    polygons = []
    actual = []
    expected = []
    macro_segments = []
    for cell, vertices in enumerate(data["macro_points"][data["macro_cells"]]):
        macro = slice_polygon(vertices, height)
        if not len(macro):
            continue
        macro_segments.extend(zip(macro[:, :2], np.roll(macro[:, :2], -1, axis=0), strict=True))
        mesh = TetraMesh(data["local_points"][cell], data["local_cells"][cell])
        dofs, _ = tetra_nodal_space(mesh, 2)
        for index, vertices in enumerate(mesh.points[mesh.cells]):
            polygon = slice_polygon(vertices, height)
            if not len(polygon):
                continue
            point = polygon.mean(axis=0)
            affine = np.column_stack((np.ones(4), vertices))
            inverse = np.linalg.inv(affine)
            bary = (np.r_[1.0, point] @ inverse)[None, :]
            basis, derivative = tetra_basis(2, bary)
            coefficients = data["pressure"][cell, dofs[index]]
            p = float(basis[0] @ coefficients)
            q = -coefficients @ derivative[0] @ inverse[1:].T
            pe, qe = exact(point[None])
            polygons.append(polygon[:, :2])
            actual.append((p, np.linalg.norm(q)))
            expected.append((pe[0], np.linalg.norm(qe[0])))
    actual, expected = np.asarray(actual), np.asarray(expected)
    output = ROOT / "docs/figures/darcy3d"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for row, label in enumerate(("Pressure", "Flux magnitude")):
        lo = min(actual[:, row].min(), expected[:, row].min())
        hi = max(actual[:, row].max(), expected[:, row].max())
        common = (
            TwoSlopeNorm(vmin=-max(abs(lo), abs(hi)), vcenter=0, vmax=max(abs(lo), abs(hi)))
            if row == 0
            else Normalize(vmin=0, vmax=hi)
        )
        difference = actual[:, row] - expected[:, row]
        bound = max(np.max(abs(difference)), 1e-15)
        for column, values in enumerate((expected[:, row], actual[:, row], difference)):
            axis = axes[row, column]
            norm = common if column < 2 else TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            colors = "RdBu_r" if row == 0 or column == 2 else "viridis"
            artist = PolyCollection(
                polygons, array=values, norm=norm, cmap=colors, edgecolors="none"
            )
            axis.add_collection(artist)
            axis.add_collection(
                LineCollection(macro_segments, colors="white", linewidths=1.9, zorder=3)
            )
            axis.add_collection(
                LineCollection(macro_segments, colors="black", linewidths=0.65, zorder=4)
            )
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=f"{label}: {('exact', 'MHM P2', 'MHM − exact')[column]}",
            )
            fig.colorbar(artist, ax=axis, shrink=0.83, pad=0.03)
    fig.suptitle("Tetrahedral MHM cross-section z = 3/8 · actual macro intersections", fontsize=16)
    for ext in ("png", "svg"):
        fig.savefig(output / f"fields.{ext}", dpi=170)
    plt.close(fig)
    rows = report["rows"]
    h = 1 / np.array([row["n"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, name, label in zip(
        axes,
        ("pressure_l2_error", "flux_l2_error"),
        ("Pressure L² error", "Physical flux L² error"),
        strict=True,
    ):
        errors = np.array([row[name] for row in rows])
        axis.loglog(h, errors, "o-", label="P2 local / P0 face subtriangles")
        axis.set(xlabel="Cartesian macro spacing 1/n", ylabel=label)
        axis.set_xticks(h, labels=["1", "1/2", "1/3", "1/4", "1/5"])
        axis.minorticks_off()
        axis.grid(alpha=0.25, which="major")
        axis.legend(fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(output / f"convergence.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
