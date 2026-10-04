"""Render exact, broken P4 and error fields with the original polyhedral macrofaces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plot_mesh import draw_macro_mesh
from plot_style import set_refinement_ticks
from solve_polyhedral_rad import partition
from solve_rad3d import exact

from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/figures/polyhedral-rad"
FAMILIES = ("cube", "triangular-prism", "hexagonal-prism")


def save(figure: plt.Figure, name: str) -> None:
    """Save vector and raster forms with identical explicitly reserved plot regions."""
    FOLDER.mkdir(exist_ok=True, parents=True)
    for extension in ("svg", "png"):
        figure.savefig(FOLDER / f"{name}.{extension}", dpi=190)
    plt.close(figure)


def geometry() -> None:
    """Draw complete original macrofaces; local tetrahedron faces are not skeleton entities."""
    figure = plt.figure(figsize=(14, 4.8), layout="constrained")
    for j, family in enumerate(FAMILIES):
        mesh, _ = partition(2, family)
        axis = figure.add_subplot(1, 3, j + 1, projection="3d")
        faces = [mesh.points[ids] for ids in mesh.faces]
        collection = Poly3DCollection(
            faces, facecolor="#9ecae1", edgecolor="#253746", linewidth=0.65, alpha=0.15
        )
        axis.add_collection3d(collection)
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            zlim=(0, 1),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=family.replace("-", " ").capitalize(),
        )
        axis.set_box_aspect((1, 1, 1), zoom=0.87)
        axis.view_init(elev=23, azim=-57)
        axis.set_xticks([0, 0.5, 1])
        axis.set_yticks([0.5, 1])
        axis.set_zticks([0, 0.5, 1])
        axis.set_zlabel("")
        axis.text2D(0.98, 0.53, "z", transform=axis.transAxes)
    save(figure, "geometry")


def evaluate_slice(
    archive: Path, n: int, family: str, points: np.ndarray
) -> tuple[np.ndarray, object]:
    """Locate each sample in one physical tetrahedron and evaluate its own P4 polynomial."""
    mesh, base = partition(n, family)
    zlayer = min(int(points[0, 2] * n), n - 1)
    result = np.full(len(points), np.nan)
    with np.load(archive) as data:
        for polygon, ids in enumerate(base.cells):
            xy = base.points[ids]
            edges = np.roll(xy, -1, axis=0) - xy
            delta = points[:, None, :2] - xy[None]
            inside = np.all(
                edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0] >= -1e-13,
                axis=1,
            )
            selected = np.flatnonzero(inside)
            cell = zlayer * len(base.cells) + polygon
            fine = TetraMesh(data[f"points_{cell}"], data[f"cells_{cell}"])
            dofs, _ = tetra_nodal_space(fine, 4)
            values = data[f"coefficients_{cell}"]
            remaining = np.ones(len(selected), dtype=bool)
            for element, vertices in enumerate(fine.points[fine.cells]):
                local = (points[selected] - vertices[0]) @ np.linalg.inv(
                    (vertices[1:] - vertices[0]).T
                ).T
                bary = np.column_stack((1 - local.sum(axis=1), local))
                belongs = remaining & np.all(bary >= -1e-12, axis=1)
                result[selected[belongs]] = tetra_basis(4, bary[belongs])[0] @ values[dofs[element]]
                remaining[belongs] = False
            if remaining.any():
                raise ValueError("slice sample not found in its polyhedral local mesh")
    if not np.isfinite(result).all():
        raise ValueError("slice has samples outside the macro partition")
    return result, base


def fields(rows: list[dict]) -> None:
    """Plot signed exact/numerical fields and positive error, preserving broken interfaces."""
    size = 360
    coordinates = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 0.37)))
    exact_values = exact(points)
    samples = []
    for family in FAMILIES:
        row = max((r for r in rows if r["family"] == family), key=lambda r: r["n"])
        values, base = evaluate_slice(
            ROOT / "build/results/polyhedral-rad" / row["field_archive"], row["n"], family, points
        )
        samples.append((row, values, base))
    limit = max(float(abs(exact_values).max()), *(float(abs(item[1]).max()) for item in samples))
    error_limit = max(float(abs(item[1] - exact_values).max()) for item in samples)
    figure = plt.figure(figsize=(12, 11.3), layout="constrained")
    grid = figure.add_gridspec(3, 4, width_ratios=(1, 1, 1, 0.05))
    for j, (row, values, base) in enumerate(samples):
        for k, (label, data) in enumerate(
            (
                ("Exact", exact_values),
                ("P4 / face P1", values),
                ("Absolute error", abs(values - exact_values)),
            )
        ):
            axis = figure.add_subplot(grid[j, k])
            image = axis.imshow(
                data.reshape(size, size),
                origin="lower",
                extent=(0, 1, 0, 1),
                interpolation="nearest",
                cmap="magma" if k == 2 else "RdBu_r",
                vmin=0 if k == 2 else -limit,
                vmax=error_limit if k == 2 else limit,
            )
            draw_macro_mesh(axis, base)
            axis.set(xlabel="x", ylabel="y", title=f"{row['family'].replace('-', ' ')} — {label}")
            axis.set_xticks([0, 0.5, 1])
            axis.set_yticks([0, 0.5, 1])
            if k == 1:
                figure.colorbar(
                    image, ax=axis, location="bottom", fraction=0.065, pad=0.13, label="Scalar u"
                )
            if k == 2:
                figure.colorbar(image, cax=figure.add_subplot(grid[j, 3]), label="|u − uₕ|")
    figure.suptitle(
        "Physical slice z = 0.37; original polygonal macroface intersections", fontsize=14
    )
    save(figure, "fields")


def convergence(rows: list[dict]) -> None:
    """Show five measured mesh levels against the analytical solution, without fitted rates."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for axis, key, label in zip(
        axes,
        ("l2_error", "V_error", "physical_flux_l2_error"),
        ("Scalar L2 error", "V error", "Physical flux L2 error"),
        strict=True,
    ):
        for family in FAMILIES:
            selected = sorted([r for r in rows if r["family"] == family], key=lambda r: r["n"])
            axis.loglog(
                [r["n"] for r in selected],
                [r[key] for r in selected],
                "o-",
                label=family.replace("-", " "),
            )
        axis.set(xlabel="Macro subdivisions per coordinate", ylabel=label)
        set_refinement_ticks(axis, sorted({r["n"] for r in rows}))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, "convergence")


def main() -> None:
    """Read completed records and render geometry, analytical comparisons and convergence."""
    rows = json.loads((ROOT / "examples/results/polyhedral-rad.json").read_text())["convergence"]
    geometry()
    fields(rows)
    convergence(rows)


if __name__ == "__main__":
    main()
