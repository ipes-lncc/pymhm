"""Render one-sided three-dimensional flow sections, macro intersections and convergence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flow3d_data import Flow3DData
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from plot_darcy3d import slice_polygon
from plot_style import set_refinement_ticks

from pymhm.tetrahedral import TetraMesh, tetra_basis, tetra_nodal_space

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "stokes-th2": "Stokes: Taylor–Hood P2/P1",
    "brinkman-usfem1": "Brinkman: USFEM P1/P1",
    "brinkman-usfem2": "Brinkman: USFEM P2/P2",
    "oseen-p2": "Oseen: stabilized P2/P2",
}


def slices(
    archive: Path, data: Flow3DData, height: float = 0.37, *, vector_key: str = "velocity"
) -> tuple:
    """Evaluate actual local nodal fields at polygon centroids without cross-interface averaging."""
    with np.load(archive) as stored:
        polygons, points, actual, macros = [], [], [], []
        degree, pk = int(stored["degree"]), int(stored["pressure_degree"])
        for vertices in stored["macro_points"][stored["macro_cells"]]:
            polygon = slice_polygon(vertices, height)
            if len(polygon):
                macros.extend(zip(polygon[:, :2], np.roll(polygon[:, :2], -1, axis=0), strict=True))
        for coords, cells, velocity, pressure in zip(
            stored["local_points"],
            stored["local_cells"],
            stored[vector_key],
            stored["pressure"],
            strict=True,
        ):
            fine = TetraMesh(coords, cells)
            udofs, _ = tetra_nodal_space(fine, degree)
            pdofs, _ = tetra_nodal_space(fine, pk)
            for cell, vertices in enumerate(fine.points[fine.cells]):
                polygon = slice_polygon(vertices, height)
                if not len(polygon):
                    continue
                point = polygon.mean(axis=0)
                bary = np.r_[1.0, point] @ np.linalg.inv(np.column_stack((np.ones(4), vertices)))
                ubasis = tetra_basis(degree, bary[None])[0][0]
                pbasis = tetra_basis(pk, bary[None])[0][0]
                actual.append(np.r_[ubasis @ velocity[udofs[cell]], pbasis @ pressure[pdofs[cell]]])
                points.append(point)
                polygons.append(polygon[:, :2])
        points = np.asarray(points)
        return (
            polygons,
            np.asarray(actual),
            np.column_stack((getattr(data, vector_key)(points), data.pressure(points))),
            macros,
        )


def overlay(axis: plt.Axes, macros: list) -> None:
    """Highlight each actual macro-face intersection with a two-color stroke."""
    axis.add_collection(LineCollection(macros, colors="white", linewidths=1.8, zorder=3))
    axis.add_collection(LineCollection(macros, colors="black", linewidths=0.65, zorder=4))
    axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x", ylabel="y")


def save_fields(row: dict, output: Path, *, components: bool = False) -> None:
    """Give exact/numerical fields common scales and independent explicitly labeled error scales."""
    path = ROOT / "examples/results/flow3d" / row["fields"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["fields_sha256"]:
        raise ValueError("flow archive digest mismatch")
    polygons, actual, exact, macros = slices(path, Flow3DData(row["physical_case"]))
    if components:
        names = ["Velocity x", "Velocity y", "Velocity z"]
        numerical, expected = actual[:, :3], exact[:, :3]
        difference = numerical - expected
    else:
        names = ["Velocity magnitude", "Pressure"]
        numerical = np.column_stack((np.linalg.norm(actual[:, :3], axis=1), actual[:, 3]))
        expected = np.column_stack((np.linalg.norm(exact[:, :3], axis=1), exact[:, 3]))
        difference = np.column_stack(
            (np.linalg.norm(actual[:, :3] - exact[:, :3], axis=1), actual[:, 3] - exact[:, 3])
        )
    fig, axes = plt.subplots(len(names), 3, figsize=(15.8, 4.8 * len(names)), layout="constrained")
    for index, label in enumerate(names):
        positive = not components and index == 0
        limit = max(
            float(np.max(np.abs(numerical[:, index]))),
            float(np.max(np.abs(expected[:, index]))),
            1e-15,
        )
        error_limit = max(float(np.max(np.abs(difference[:, index]))), 1e-15)
        common = Normalize(0, limit) if positive else TwoSlopeNorm(0, vmin=-limit, vmax=limit)
        error_norm = (
            Normalize(0, error_limit)
            if positive
            else TwoSlopeNorm(0, vmin=-error_limit, vmax=error_limit)
        )
        for column, values in enumerate(
            (expected[:, index], numerical[:, index], difference[:, index])
        ):
            axis = axes[index, column]
            artist = PolyCollection(
                polygons,
                array=values,
                norm=common if column < 2 else error_norm,
                cmap="viridis" if positive else "RdBu_r",
                edgecolors="none",
                rasterized=True,
            )
            axis.add_collection(artist)
            overlay(axis, macros)
            axis.set_title(
                ("Exact", "MHM", "Vector error magnitude" if positive else "MHM − exact")[column],
                fontsize=15,
            )
            colorbar = fig.colorbar(artist, ax=axis, pad=0.025, fraction=0.052)
            colorbar.set_label(
                label
                if column < 2
                else "Velocity error magnitude"
                if positive
                else f"{label} difference",
                fontsize=12,
            )
            colorbar.ax.tick_params(labelsize=10)
    fig.suptitle(
        f"{LABELS[row['case']]} — section z=0.37, macro n={row['macro_subdivisions']}", fontsize=19
    )
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.86))
    fig.text(
        0.5,
        0.027,
        (
            "Flat colors evaluate each cut fine tetrahedron at its own polygon centroid.\n"
            "Black/white lines are actual macro-face intersections; no interface averaging."
        ),
        ha="center",
        va="center",
        fontsize=11,
    )
    stem = row["case"] + ("-components" if components else "-fields")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    """Render archived flow fields and absolute physical errors at five recorded resolutions."""
    report_path = ROOT / "examples/results/flow3d/campaign.json"
    report = json.loads(report_path.read_text())
    output = ROOT / "docs/figures/flow3d"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), layout="constrained")
    for name, label in LABELS.items():
        rows = sorted(
            (r for r in report["rows"] if r["case"] == name), key=lambda r: r["macro_subdivisions"]
        )
        if not rows:
            continue
        spacing = [1 / r["macro_subdivisions"] for r in rows]
        for axis, key in zip(
            axes, ["velocity_l2", "pressure_l2", "velocity_h1_seminorm"], strict=True
        ):
            axis.loglog(spacing, [r[key] for r in rows], "o-", label=label, markersize=5)
            set_refinement_ticks(
                axis, spacing, labels=[f"1/{r['macro_subdivisions']}" for r in rows]
            )
        save_fields(rows[-1], output)
        if name == "oseen-p2":
            save_fields(rows[-1], output, components=True)
    for axis, label in zip(
        axes, ["Velocity L2 error", "Pressure L2 error", "Broken gradient L2 error"], strict=True
    ):
        axis.set(xlabel="Macro cube side", ylabel=label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.suptitle("Native three-dimensional flow — absolute physical errors", fontsize=18)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"convergence.{suffix}", dpi=180)
    plt.close(fig)
    (output / "campaign.json").write_bytes(report_path.read_bytes())


if __name__ == "__main__":
    main()
