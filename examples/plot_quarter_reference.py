"""Render archived Basix and NeoPZ quarter-five-spot comparisons with PyVista.

This script only reads numerical records. It does not execute reference solvers
or assemble comparison operators. Cell values remain cell values; broken nodal
pressures retain their independently stored macro traces.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
import pyvista as pv

if __package__:
    from .plot_pyvista_layout import horizontal_color_scale
    from .quarter_spot_problem import (
        OBSTACLE_AREA,
        OBSTACLE_LOWER,
        OBSTACLE_UPPER,
        coefficient,
        macro_mesh,
        source,
    )
else:
    from plot_pyvista_layout import horizontal_color_scale
    from quarter_spot_problem import (
        OBSTACLE_AREA,
        OBSTACLE_LOWER,
        OBSTACLE_UPPER,
        coefficient,
        macro_mesh,
        source,
    )

from pymhm import TriangleMesh
from pymhm.visualization import macro_edges

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/quarter-five-spot/reference"
FIGURES = ROOT / "docs/figures/quarter-five-spot"
DIRECTION_LENGTH = 0.035


def load_record(row: dict) -> dict[str, np.ndarray]:
    """Read a field archive after validating its recorded content digest."""
    path = DATA / row["fields"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["fields_sha256"]:
        raise ValueError(f"Field archive checksum differs: {path.name}")
    with np.load(path) as archive:
        return dict(archive)


def grid_from_record(record: dict[str, np.ndarray]) -> pv.UnstructuredGrid:
    """Preserve archived point connectivity without welding coincident vertices."""
    points, cells = record["points"], record["cells"]
    return pv.UnstructuredGrid(
        np.column_stack((np.full(len(cells), 3), cells)).ravel(),
        np.full(len(cells), pv.CellType.TRIANGLE),
        np.column_stack((points, np.zeros(len(points)))),
    )


def overlay(plotter: pv.Plotter, record: dict[str, np.ndarray]) -> None:
    """Draw archived macrofaces and the shared central square obstacle boundary."""
    mesh = TriangleMesh(record["macro_points"], record["macro_cells"])
    edges = macro_edges(mesh)
    edges.points[:, 2] = 0.9e-5
    plotter.add_mesh(edges, color="white", line_width=1.8, opacity=0.45)
    edges = edges.copy()
    edges.points[:, 2] = 1e-5
    plotter.add_mesh(edges, color="#263544", line_width=0.8, opacity=0.7)
    rectangle = pv.lines_from_points(
        np.array(
            [
                [OBSTACLE_LOWER, OBSTACLE_LOWER, 2e-5],
                [OBSTACLE_UPPER, OBSTACLE_LOWER, 2e-5],
                [OBSTACLE_UPPER, OBSTACLE_UPPER, 2e-5],
                [OBSTACLE_LOWER, OBSTACLE_UPPER, 2e-5],
            ]
        ),
        close=True,
    )
    plotter.add_mesh(rectangle, color="white", line_width=4)
    rectangle.points[:, 2] = 3e-5
    plotter.add_mesh(rectangle, color="#111111", line_width=2)


def direction_cell_indices(record: dict[str, np.ndarray], bins: int = 18) -> np.ndarray:
    """Select one actual fine-cell centroid per regular spatial bin.

    The selected centroid is nearest its bin center. Selection uses geometry
    only, so the reference and PyMHM arrows occupy identical physical locations
    even on the nonuniform fitted mesh. No field interpolation is involved.
    """
    centroids = record["points"][record["cells"]].mean(axis=1)
    coordinates = np.minimum((centroids * bins).astype(int), bins - 1)
    centers = (coordinates + 0.5) / bins
    distance = np.sum((centroids - centers) ** 2, axis=1)
    bin_ids = coordinates[:, 0] + bins * coordinates[:, 1]
    ordering = np.lexsort((distance, bin_ids))
    _, first = np.unique(bin_ids[ordering], return_index=True)
    selected = ordering[first]
    inside = np.all(
        (centroids[selected] > DIRECTION_LENGTH) & (centroids[selected] < 1 - DIRECTION_LENGTH),
        axis=1,
    )
    return selected[inside]


def direction_overlay(
    plotter: pv.Plotter,
    record: dict[str, np.ndarray],
    flux: np.ndarray,
    selected: np.ndarray,
) -> None:
    """Draw equal-length flux directions centered on selected real centroids.

    Arrow lengths encode direction only and equal ``DIRECTION_LENGTH`` in
    physical domain coordinates. Exactly zero vectors have no arrow. White
    outlines keep the dark shafts legible over the magnitude color map.
    """
    magnitudes = np.linalg.norm(flux[selected], axis=1)
    nonzero = magnitudes > 0
    selected, magnitudes = selected[nonzero], magnitudes[nonzero]
    if len(selected) == 0:
        return
    centers = record["points"][record["cells"][selected]].mean(axis=1)
    directions = flux[selected] / magnitudes[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    tail = centers - 0.5 * DIRECTION_LENGTH * directions
    tip = centers + 0.5 * DIRECTION_LENGTH * directions
    head = tip - 0.3 * DIRECTION_LENGTH * directions
    wing = 0.16 * DIRECTION_LENGTH * normals
    segments = np.stack((tail, tip, head + wing, tip, head - wing, tip), axis=1)
    points = np.column_stack((segments.reshape(-1, 2), np.full(segments.size // 2, 4e-5)))
    indices = np.arange(len(points)).reshape(-1, 2)
    lines = pv.PolyData(points, lines=np.column_stack((np.full(len(indices), 2), indices)))
    plotter.add_mesh(lines, color="white", line_width=4, lighting=False)
    lines = lines.copy()
    lines.points[:, 2] = 5e-5
    plotter.add_mesh(lines, color="#111111", line_width=1.7, lighting=False)


def finish_quarter_panel(
    plotter: pv.Plotter,
    actor: pv.Actor,
    *,
    title: str,
    scalar: str,
    scientific: bool = True,
) -> None:
    """Reserve separate viewport regions for the map, title and color scale.

    All experiments here occupy the unit square. Its orthographic image is
    fitted inside a central rectangle, with a dedicated lower band for a
    horizontal scale and its title. Separate scalar-bar actors avoid implicit
    sharing between panels with identical field names.
    """
    viewport = plotter.renderer.GetViewport()
    width = plotter.window_size[0] * (viewport[2] - viewport[0])
    height = plotter.window_size[1] * (viewport[3] - viewport[1])
    image_side = min(0.82 * width, 0.58 * height)
    parallel_scale = height / (2 * image_side)
    focal_y = 0.5 - 0.04 * parallel_scale
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.focal_point = (0.5, focal_y, 0.0)
    plotter.camera.position = (0.5, focal_y, 3.0)

    wrapped = "\n".join(
        textwrap.fill(line, width=max(28, int(width / 19))) for line in title.splitlines()
    )
    heading = plotter.add_text(
        wrapped, position=(0.02, 0.985), viewport=True, font_size=18, color="black"
    )
    heading.GetTextProperty().SetVerticalJustificationToTop()
    horizontal_color_scale(plotter, actor, scalar, scientific=scientific)


def panel(
    plotter: pv.Plotter,
    record: dict[str, np.ndarray],
    values: np.ndarray,
    *,
    title: str,
    scalar: str,
    limits: tuple[float, float],
    point_values: bool = False,
    cmap: str = "viridis",
    logarithmic: bool = False,
) -> None:
    """Render one unsmoothed field with its own explicitly labeled scalar bar."""
    grid = grid_from_record(record)
    storage = grid.point_data if point_values else grid.cell_data
    storage[scalar] = values
    plotter.set_background("white")
    actor = plotter.add_mesh(
        grid,
        scalars=scalar,
        preference="point" if point_values else "cell",
        clim=limits,
        cmap=cmap,
        log_scale=logarithmic,
        lighting=False,
        interpolate_before_map=False,
        show_scalar_bar=False,
    )
    overlay(plotter, record)
    finish_quarter_panel(plotter, actor, title=title, scalar=scalar)


def discretization_caption(row: dict, record: dict[str, np.ndarray], column: int) -> str:
    """Label a reference space independently of the compared MHM trace space."""
    if column == 0 and "reference_discretization" in row:
        return row["reference_discretization"]
    return (
        f"{len(record['macro_cells'])} macros; r={row['refinement']}; trace P0, s={row['segments']}"
    )


def compare_fields(row: dict, label: str, output: str) -> None:
    """Compare pressure and sampled flux with common reference/pyMHM color scales."""
    record = load_record(row)
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2100, 2000))
    pressure_at_points = len(record["reference_pressure"]) == len(record["points"])
    for index, (key, scalar, error_key) in enumerate(
        (
            ("pressure", "Pressure", "pressure_l2_difference"),
            ("flux", "Flux magnitude", "flux_l2_difference"),
        )
    ):
        reference, numerical = record[f"reference_{key}"], record[f"pymhm_{key}"]
        error = abs(reference - numerical)
        if key == "flux":
            error = np.linalg.norm(reference - numerical, axis=1)
            reference, numerical = (
                np.linalg.norm(reference, axis=1),
                np.linalg.norm(numerical, axis=1),
            )
        limits = (
            float(min(reference.min(), numerical.min())),
            float(max(reference.max(), numerical.max())),
        )
        if key == "pressure":
            extent = max(abs(limits[0]), abs(limits[1]))
            limits = (-extent, extent)
        error_limits = (0.0, max(float(error.max()), np.finfo(float).tiny))
        for column, (values, caption, bounds) in enumerate(
            (
                (reference, label, limits),
                (numerical, "PyMHM", limits),
                (error, "Absolute difference", error_limits),
            )
        ):
            plotter.subplot(index, column)
            description = "P1 nodal pressure" if pressure_at_points else "P0 pressure"
            error_scalar = (
                "Absolute nodal difference"
                if pressure_at_points
                else "Absolute pressure difference"
            )
            if key == "flux":
                description = "Fine-cell centroid flux"
                error_scalar = "Flux-vector difference magnitude"
            if column == 2:
                description += f"\nL2 difference {row[error_key]:.2e}"
            panel(
                plotter,
                record,
                values,
                title=f"{caption}\n{description}\n{discretization_caption(row, record, column)}",
                scalar=scalar if column != 2 else error_scalar,
                limits=bounds,
                point_values=key == "pressure" and pressure_at_points,
                cmap="magma" if column == 2 else ("RdBu_r" if key == "pressure" else "viridis"),
            )
    plotter.screenshot(FIGURES / output)
    plotter.close()


def compare_components(row: dict, label: str, output: str) -> None:
    """Compare signed centroid flux components on common symmetric color scales.

    Each component has one linear scale shared by reference and PyMHM. Their
    signed difference, PyMHM minus reference, has its own symmetric scale.
    Cell samples remain unchanged and are never averaged across interfaces.
    """
    record = load_record(row)
    reference, numerical = record["reference_flux"], record["pymhm_flux"]
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2100, 2000))
    for component, name in enumerate(("qx", "qy")):
        extent = max(
            float(abs(reference[:, component]).max()),
            float(abs(numerical[:, component]).max()),
            np.finfo(float).tiny,
        )
        difference = numerical[:, component] - reference[:, component]
        error_extent = max(float(abs(difference).max()), np.finfo(float).tiny)
        for column, (values, caption, bound) in enumerate(
            (
                (reference[:, component], label, extent),
                (numerical[:, component], "PyMHM", extent),
                (difference, "PyMHM minus reference", error_extent),
            )
        ):
            plotter.subplot(component, column)
            panel(
                plotter,
                record,
                values,
                title=(
                    f"{caption}\nSigned {name}; fine-cell centroids\n"
                    f"{discretization_caption(row, record, column)}"
                ),
                scalar=f"Flux {name}" if column < 2 else f"Signed {name} difference",
                limits=(-extent, extent) if column < 2 else (-bound, bound),
                cmap="RdBu_r",
            )
    plotter.screenshot(FIGURES / output)
    plotter.close()


def compare_directions(row: dict, label: str, output: str) -> None:
    """Show flux magnitude and normalized directions at common fine-cell centroids."""
    record = load_record(row)
    selected = direction_cell_indices(record)
    reference, numerical = record["reference_flux"], record["pymhm_flux"]
    magnitudes = [np.linalg.norm(values, axis=1) for values in (reference, numerical)]
    upper = max(float(values.max()) for values in magnitudes)
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 1000))
    for column, (flux, magnitude, caption) in enumerate(
        zip((reference, numerical), magnitudes, (label, "PyMHM"), strict=True)
    ):
        plotter.subplot(0, column)
        panel(
            plotter,
            record,
            magnitude,
            title=(
                f"{caption}\nColor: centroid flux magnitude\n"
                f"Arrows: direction only; length {DIRECTION_LENGTH:g}\n"
                f"{discretization_caption(row, record, column)}"
            ),
            scalar="Flux magnitude",
            limits=(0.0, upper),
        )
        direction_overlay(plotter, record, flux, selected)
    plotter.screenshot(FIGURES / output)
    plotter.close()


def geometry_record(refinement: int = 4) -> dict[str, np.ndarray]:
    """Sample shared benchmark data on its material-fitted local triangles."""
    macro = macro_mesh()
    points, cells, offset = [], [], 0
    for index in range(len(macro.cells)):
        fine = macro.submesh(index, refinement)
        points.append(fine.points)
        cells.append(fine.cells + offset)
        offset += len(fine.points)
    vertices, connectivity = np.concatenate(points), np.concatenate(cells)
    centers = vertices[connectivity].mean(axis=1)
    return {
        "points": vertices,
        "cells": connectivity,
        "macro_points": macro.points,
        "macro_cells": macro.cells,
        "permeability": coefficient(centers),
        "source_density": source(centers),
    }


def problem_geometry() -> None:
    """Show the exact obstacle and fixed well supports from shared problem data."""
    record = geometry_record()
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 1000))
    plotter.subplot(0, 0)
    panel(
        plotter,
        record,
        record["permeability"],
        title=(
            f"Permeability: background 1\nObstacle: 1e-4; square area {OBSTACLE_AREA:g}\n"
            f"Bounds: [{OBSTACLE_LOWER:.6f}, {OBSTACLE_UPPER:.6f}] in x and y"
        ),
        scalar="Permeability (log scale)",
        limits=(1e-4, 1.0),
        logarithmic=True,
    )
    plotter.subplot(0, 1)
    panel(
        plotter,
        record,
        record["source_density"],
        title=(
            "Finite wells: density -100 / +100\n"
            "Support area: 0.01 each\nTotal extraction / injection: -1 / +1"
        ),
        scalar="Source density",
        limits=(-100.0, 100.0),
        cmap="RdBu_r",
    )
    plotter.screenshot(FIGURES / "reference-obstacle-geometry.png")
    plotter.close()


def main() -> None:
    """Redraw the numerical archives without invoking any comparison program."""
    report = json.loads((DATA / "comparison.json").read_text())
    candidates = [row for row in report["cases"] if row["segments"] == 2]
    refinement = max(row["refinement"] for row in candidates)
    rows = [row for row in candidates if row["refinement"] == refinement]
    FIGURES.mkdir(parents=True, exist_ok=True)
    primal = next(row for row in rows if row["formulation"] == "primal")
    mixed = next(row for row in rows if row["formulation"] == "mixed")
    problem_geometry()
    compare_fields(primal, "Basix independent P1", "reference-obstacle-basix.png")
    compare_fields(mixed, "NeoPZ restricted RT0/P0", "reference-obstacle-neopz.png")
    for row, label, suffix in (
        (primal, "Basix independent P1", "basix"),
        (mixed, "NeoPZ restricted RT0/P0", "neopz"),
    ):
        compare_components(row, label, f"reference-obstacle-{suffix}-components.png")
        compare_directions(row, label, f"reference-obstacle-{suffix}-directions.png")


if __name__ == "__main__":
    main()
