"""Render the layered point-well Darcy flux as broken elevation surfaces.

Both formulations retain the same physical height and color scales. The
renderer preserves separate finite-element traces and never rescales a field
to match a published peak. An additional view excludes the well neighborhoods
to resolve the interior field without the singular corner values dominating.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pyvista as pv
from plot_pyvista_layout import horizontal_color_scale
from plot_quarter_spot import FIGURES, OUTPUT

from pymhm import TriangleMesh
from pymhm.visualization import macro_edges


def corner_well_distance(segments: np.ndarray) -> np.ndarray:
    """Return the exact minimum distance from each segment to either corner well."""
    start = segments[:, 0, :2]
    tangent = segments[:, 1, :2] - start
    length_squared = np.sum(tangent**2, axis=1)
    distances = []
    for corner in (0.0, 1.0):
        parameter = np.clip(np.sum((corner - start) * tangent, axis=1) / length_squared, 0, 1)
        closest = start + parameter[:, None] * tangent
        distances.append(np.linalg.norm(closest - corner, axis=1))
    return np.minimum(*distances)


def elevated_macro_edges(grid: pv.UnstructuredGrid, macro: TriangleMesh) -> pv.PolyData:
    """Follow each macroface on its own broken surface without averaging traces."""
    cells = grid.cells.reshape(-1, 4)[:, 1:]
    triangles = grid.points[cells].reshape(len(macro.cells), -1, 3, 3)
    segments = []
    for cell, faces in enumerate(triangles):
        vertices = macro.points[macro.cells[cell]]
        for left, right in ((0, 1), (1, 2), (2, 0)):
            endpoints = faces[:, [left, right]]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                tangent = vertices[b] - vertices[a]
                offset = endpoints[..., :2] - vertices[a]
                cross = offset[..., 0] * tangent[1] - offset[..., 1] * tangent[0]
                on_face = np.all(abs(cross) < 1e-12 * np.linalg.norm(tangent), axis=1)
                segments.extend(endpoints[on_face])
    coordinates = np.asarray(segments).reshape(-1, 3)
    lines = np.column_stack(
        (np.full(len(coordinates) // 2, 2), np.arange(len(coordinates)).reshape(-1, 2))
    )
    return pv.PolyData(coordinates, lines=lines)


def render_elevation(grids: list[pv.UnstructuredGrid], macro: TriangleMesh, radius: float) -> dict:
    """Render matched views with physical flux labels and an explicit height factor."""
    selected = []
    for grid in grids:
        triangles = grid.points[grid.cells.reshape(-1, 4)[:, 1:], :2]
        segments = np.stack((triangles, np.roll(triangles, -1, axis=1)), axis=2)
        distance = corner_well_distance(segments.reshape(-1, 2, 2)).reshape(-1, 3)
        selected.append(np.flatnonzero(np.all(distance >= radius, axis=1)))
    peak = max(
        float(grid.extract_cells(indices)["Flux magnitude"].max())
        for grid, indices in zip(grids, selected, strict=True)
    )
    height_factor = 0.9 / peak
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2400, 1300))
    for column, (grid, indices, name) in enumerate(
        zip(grids, selected, ("Primal P2", "Mixed RT0/P0"), strict=True)
    ):
        plotter.subplot(0, column)
        plotter.set_background("white")
        raised = grid.copy()
        raised.points[:, 2] = height_factor * raised["Flux magnitude"]
        surface = raised.extract_cells(indices)
        actor = plotter.add_mesh(
            surface,
            scalars="Flux magnitude",
            clim=(0, peak),
            cmap="viridis",
            show_edges=False,
            lighting=False,
            smooth_shading=False,
            show_scalar_bar=False,
        )
        edges = elevated_macro_edges(raised, macro)
        if radius:
            endpoints = edges.points.reshape(-1, 2, 3)[..., :2]
            distance = corner_well_distance(endpoints)
            edges = edges.extract_cells(np.flatnonzero(distance >= radius))
        plotter.add_mesh(edges, color="#252b30", line_width=0.6, lighting=False)
        plotter.add_mesh(macro_edges(macro), color="#777777", line_width=0.45, lighting=False)
        plotter.add_mesh(pv.Line((0, 0.484375, 0), (1, 0.484375, 0)), color="#c53b27", line_width=3)
        region = "Complete domain" if not radius else f"Outside well neighborhoods: radius {radius}"
        heading = plotter.add_text(
            f"{name}: flux magnitude elevation\n2048 macrotriangles; local r=2; trace P0\n{region}",
            position=(0.04, 0.98),
            viewport=True,
            font_size=18,
            color="black",
        )
        heading.GetTextProperty().SetVerticalJustificationToTop()
        caption = plotter.add_text(
            f"Unit corner wells; height = {height_factor:.4g} norm(q)\n"
            "Common height and color scales; no trace averaging",
            position=(0.5, 0.18),
            viewport=True,
            font_size=15,
            color="black",
        )
        caption.GetTextProperty().SetJustificationToCentered()
        caption.GetTextProperty().SetVerticalJustificationToBottom()
        horizontal_color_scale(plotter, actor, "Flux magnitude", scientific=False)
        plotter.add_axes(viewport=(0.82, 0.22, 0.98, 0.36), color="black")
        plotter.camera_position = [(2.3, -2.8, 2.0), (0.5, 0.5, 0.38), (0, 0, 1)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 1.05
        plotter.camera.SetWindowCenter(0.0, -0.14)
    filename = "point-flux-elevation" + ("-detail" if radius else "") + ".png"
    plotter.screenshot(FIGURES / filename)
    plotter.close()
    return {
        "file": filename,
        "excluded_well_radius": radius,
        "selection": "Whole display triangles outside both open well disks; exact edge distances",
        "retained_display_triangles": [len(indices) for indices in selected],
        "height_factor": height_factor,
        "physical_color_limits": [0.0, peak],
    }


def main() -> None:
    """Read original numerical archives and write the two elevation views."""
    with np.load(OUTPUT / "macro.npz") as data:
        macro = TriangleMesh(data["points"], data["cells"])
    paths = [OUTPUT / f"layer-offset-{method}.vtu" for method in ("primal", "mixed")]
    grids = [pv.read(path) for path in paths]
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = {
        "problem": "Unit corner point wells; K=1000 below y=0.484375, K=1 above",
        "formulations": ["Primal P2", "Mixed RT0/P0"],
        "macro_triangles": len(macro.cells),
        "local_refinement": 2,
        "trace_degree": 0,
        "display_sampling": "Existing one-sided nodal display fields; independent cell vertices",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "sampled_flux_peaks": [float(grid["Flux magnitude"].max()) for grid in grids],
        "views": [render_elevation(grids, macro, radius) for radius in (0.0, 0.125)],
    }
    (FIGURES / "point-flux-elevation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
