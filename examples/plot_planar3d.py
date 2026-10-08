"""Plot exact transmission fields on fitted tetrahedra without interface smoothing."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from examples.planar3d_data import Planar3DData
from examples.plot_darcy3d import slice_polygon
from examples.plot_flow3d import overlay
from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]


def section(path: Path, data: Planar3DData, height: float = 0.37) -> tuple:
    """Evaluate owned polynomial fields at centroids of true fine-tetrahedron cuts."""
    polygons, points, values, macros = [], [], [], []
    with np.load(path) as archive:
        degree = int(archive["degree"])
        for vertices in archive["macro_points"][archive["macro_cells"]]:
            polygon = slice_polygon(vertices, height)
            if len(polygon):
                macros.extend(zip(polygon[:, :2], np.roll(polygon[:, :2], -1, axis=0), strict=True))
        for owner in range(len(archive["macro_cells"])):
            mesh = TetraMesh(archive[f"local_points_{owner}"], archive[f"local_cells_{owner}"])
            dofs, _ = tetra_nodal_space(mesh, degree)
            coefficients = archive[f"pressure_{owner}"]
            for cell, vertices in enumerate(mesh.points[mesh.cells]):
                polygon = slice_polygon(vertices, height)
                if not len(polygon):
                    continue
                point = polygon.mean(axis=0)
                transform = (vertices[1:] - vertices[0]).T
                reference = np.linalg.solve(transform, point - vertices[0])
                basis, derivative = tetra_basis(
                    degree, np.r_[1 - reference.sum(), reference][None]
                )[:2]
                local = coefficients[dofs[cell]]
                pressure = float(basis[0] @ local)
                reference_gradient = local @ derivative[0]
                gradient = (reference_gradient[1:] - reference_gradient[:1]) @ np.linalg.inv(
                    transform
                )
                flux = -data.material(point[None])[0] @ gradient
                polygons.append(polygon[:, :2])
                points.append(point)
                values.append(np.r_[pressure, flux])
    points = np.array(points)
    exact = np.column_stack((data.pressure(points), data.flux(points)))
    return polygons, np.array(values), exact, points, macros


def field_panels(
    polygons: list,
    numerical: np.ndarray,
    exact: np.ndarray,
    macros: list,
    output: Path,
    *,
    components: bool = False,
) -> None:
    """Share exact/numerical scales, reserving individual error scales and actual macro edges."""
    labels = ["Flux x", "Flux y", "Flux z"] if components else ["Pressure", "Flux magnitude"]
    actual = (
        numerical[:, 1:]
        if components
        else np.column_stack((numerical[:, 0], np.linalg.norm(numerical[:, 1:], axis=1)))
    )
    expected = (
        exact[:, 1:]
        if components
        else np.column_stack((exact[:, 0], np.linalg.norm(exact[:, 1:], axis=1)))
    )
    errors = actual - expected
    if not components:
        errors[:, 1] = np.linalg.norm(numerical[:, 1:] - exact[:, 1:], axis=1)
    fig, axes = plt.subplots(
        len(labels), 3, figsize=(15.8, 4.5 * len(labels)), layout="constrained"
    )
    for index, label in enumerate(labels):
        positive = not components and index == 1
        limit = max(abs(actual[:, index]).max(), abs(expected[:, index]).max(), 1e-15)
        error_limit = max(abs(errors[:, index]).max(), 1e-15)
        common = Normalize(0, limit) if positive else TwoSlopeNorm(0, vmin=-limit, vmax=limit)
        error_norm = (
            Normalize(0, error_limit)
            if positive
            else TwoSlopeNorm(0, vmin=-error_limit, vmax=error_limit)
        )
        for column, array in enumerate((expected[:, index], actual[:, index], errors[:, index])):
            axis = axes[index, column]
            collection = PolyCollection(
                polygons,
                array=array,
                edgecolors="none",
                rasterized=True,
                cmap="viridis" if positive else "RdBu_r",
                norm=common if column < 2 else error_norm,
            )
            axis.add_collection(collection)
            overlay(axis, macros)
            axis.set_title(
                ("Exact", "MHM raw field", "Vector error magnitude" if positive else "MHM − exact")[
                    column
                ],
                fontsize=15,
            )
            fig.colorbar(collection, ax=axis, pad=0.025, fraction=0.052).set_label(
                label if column < 2 else "Flux vector error norm" if positive else f"{label} error",
                fontsize=13,
            )
    fig.suptitle(
        "Planar transmission — material-fitted P4 locals / P1 face modes, z=0.37", fontsize=18
    )
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.93))
    fig.text(
        0.5,
        0.02,
        (
            "Flat one-sided values on actual fine-cell cuts; macro intersections in black/white.\n"
            "Errors are at floating-point accuracy; no smoothing, clipping "
            "or cross-interface averaging."
        ),
        ha="center",
        fontsize=12,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{'components' if components else 'fields'}.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    """Replay the accepted n=2 mixed-boundary field and its geometry metadata."""
    source = ROOT / "examples/results/planar3d/campaign.json"
    report = json.loads(source.read_text())
    row = next(
        r for r in report["rows"] if r["macro_subdivisions"] == 2 and r["boundary"] == "mixed"
    )
    archive = source.parent / row["fields"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != row["fields_sha256"]:
        raise ValueError("planar field archive digest mismatch")
    output = ROOT / "docs/figures/planar3d"
    output.mkdir(parents=True, exist_ok=True)
    data = Planar3DData()
    polygons, values, exact, points, macros = section(archive, data)
    if np.max(abs(values - exact)) > 2e-8:
        raise ValueError("archived transmission patch failed physical point evaluation")
    field_panels(polygons, values, exact, macros, output)
    field_panels(polygons, values, exact, macros, output, components=True)
    fig, axis = plt.subplots(figsize=(8.1, 7.2), layout="constrained")
    material = PolyCollection(
        polygons,
        array=np.log10(np.where(points @ data.normal <= 0.63, data.contrast, 1.0)),
        cmap="viridis",
        edgecolors="none",
        rasterized=True,
    )
    axis.add_collection(material)
    overlay(axis, macros)
    colorbar = fig.colorbar(material, ax=axis, pad=0.04)
    colorbar.set_label("log10 material multiplier k", fontsize=14)
    axis.set_title("Oblique material interface and actual macro cuts", fontsize=16)
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.90))
    fig.text(
        0.5,
        0.015,
        "K=k A; k=25 inside x+0.4y+0.2z≤0.63 and k=1 outside. Section z=0.37.",
        ha="center",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"geometry.{suffix}", dpi=180)
    plt.close(fig)
    (output / "campaign.json").write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
