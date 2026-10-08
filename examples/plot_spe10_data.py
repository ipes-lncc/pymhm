"""Acquire pinned SPE10 data and render unchanged Cartesian cells with PyVista."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
import pyvista as pv

from examples.plot_pyvista_layout import horizontal_color_scale
from pymhm import TriangleMesh
from pymhm.io.datasets.spe10 import (
    SPE10_FILES,
    SPE10_REVISION,
    download_spe10_model2,
    load_spe10_model2,
)
from pymhm.io.reservoir import ReservoirData
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.visualization import macro_edges, structured_cell_grid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/spe10"
FIGURES = ROOT / "docs/figures/spe10"
CACHE = ROOT / "build/data/spe10"


def macro_mesh() -> CartesianMacroMesh:
    """Return the 66-square Darcy macrogrid shown in Paredes et al., Figure 6."""
    return CartesianMacroMesh(6, 11, bounds=(0, 1200, 0, 2200))


def data_grid(layer: int) -> pv.ImageData:
    """Load an archived layer without requiring a full-volume download."""
    with np.load(OUTPUT / f"layer-{layer}.npz") as data:
        return structured_cell_grid(
            (60, 220),
            cell_data={
                "ln(Kx / mD)": np.log(data["permeability"][..., 0]),
                "Porosity": data["porosity"],
            },
            spacing=(20, 10),
        )


def panel(
    plotter: pv.Plotter,
    grid: pv.DataSet,
    scalar: str,
    title: str,
    *,
    limits: tuple[float, float] | None = None,
    cmap: str = "jet",
    mesh: CartesianMacroMesh | TriangleMesh | None = None,
    color_labels: dict[float, str] | None = None,
) -> None:
    """Render an unaveraged field with a reserved right-hand colorbar gutter.

    The camera offset keeps the data rectangle and axis labels out of the
    scalar-bar region. Multi-line scalar titles stay inside that region at a
    readable font size; their width is not allowed to grow into the field.
    Optional color labels express transformed display values in physical units.
    """
    plotter.set_background("white")
    actor = plotter.add_mesh(
        grid,
        scalars=scalar,
        cmap=cmap,
        n_colors=257,
        clim=limits,
        lighting=False,
        smooth_shading=False,
        preference="cell",
        show_scalar_bar=False,
    )
    bar = plotter.add_scalar_bar(
        title=f"{scalar}:{plotter.renderer.GetViewport()}",
        mapper=actor.mapper,
        title_font_size=34,
        label_font_size=32,
        unconstrained_font_size=True,
        n_labels=5 if color_labels is None else 0,
        vertical=True,
        position_x=0.76,
        position_y=0.17,
        height=0.6,
        width=0.06,
        fmt="%.3g",
        color="black",
    )
    bar.SetTitle(textwrap.fill(scalar, width=12, break_long_words=False))
    if color_labels is not None:
        bar.SetTitle("")
        bar.SetWidth(0.025)
        bar.SetBarRatio(1.0)
        heading = plotter.add_text(
            scalar, position=(0.84, 0.81), viewport=True, font_size=18, color="black"
        )
        heading.GetTextProperty().SetJustificationToCentered()
        heading.GetTextProperty().SetVerticalJustificationToBottom()
        lower, upper = actor.mapper.scalar_range
        for value, label in color_labels.items():
            height = 0.17 + 0.6 * (value - lower) / (upper - lower)
            text = plotter.add_text(
                label,
                position=(0.80, height),
                viewport=True,
                font_size=16,
                color="black",
            )
            text.GetTextProperty().SetJustificationToLeft()
            text.GetTextProperty().SetVerticalJustificationToCentered()
    if mesh is not None:
        edges = macro_edges(mesh)
        edges.points[:, 2] = grid.bounds[5] + 1e-4
        plotter.add_mesh(edges, color="white", line_width=2.4, lighting=False)
        ink = edges.copy()
        ink.points[:, 2] += 1e-5
        plotter.add_mesh(ink, color="#23323a", line_width=1.2, lighting=False)
    heading = "\n".join(
        textwrap.fill(line, width=32, break_long_words=False) for line in title.splitlines()
    )
    plotter.add_text(heading, position="upper_left", font_size=20, color="black")
    plotter.show_bounds(
        show_zaxis=False,
        show_zlabels=False,
        xtitle="x [ft]",
        ytitle="y [ft]",
        font_size=18,
        fmt="%.0f",
        location="outer",
        ticks="outside",
        color="black",
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.0)
    plotter.camera.SetWindowCenter(0.21, 0.0)


def save_layers(data: ReservoirData) -> None:
    """Archive only the three relevant slices with source checksums and units."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "model": 2,
        "shape": list(data.porosity.shape),
        "spacing_ft": list(data.spacing),
        "permeability_unit": data.permeability_unit,
        "axis_order": "x,y,z,component; original ECLIPSE I/x index fastest",
        "revision": SPE10_REVISION,
        "repository": "https://github.com/OPM/opm-data",
        "source_checksums": SPE10_FILES,
        "porosity_convention": (
            "OPM published values preserved; header records zero-to-1e-7 replacement"
        ),
        "layers": [],
    }
    for layer in (1, 36, 85):
        k, phi = data.permeability[:, :, layer - 1], data.porosity[:, :, layer - 1]
        path = OUTPUT / f"layer-{layer}.npz"
        np.savez_compressed(path, permeability=k, porosity=phi, spacing=np.array(data.spacing[:2]))
        report["layers"].append(
            {
                "layer_one_based": layer,
                "array_index": layer - 1,
                "depth_interval_ft": [2 * (layer - 1), 2 * layer],
                "kx_range_md": [float(k[..., 0].min()), float(k[..., 0].max())],
                "phi_range": [float(phi.min()), float(phi.max())],
                "kx_equals_ky": bool(np.array_equal(k[..., 0], k[..., 1])),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "file": path.name,
            }
        )
    (OUTPUT / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")


def plot_layers() -> None:
    """Contrast the three candidate layers using common color scales and macros."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2100, 2000))
    for column, layer in enumerate((1, 36, 85)):
        grid = data_grid(layer)
        for row, scalar in enumerate(("ln(Kx / mD)", "Porosity")):
            plotter.subplot(row, column)
            panel(
                plotter,
                grid,
                scalar,
                f"Model 2 / layer {layer} / {'permeability' if row == 0 else 'porosity'}",
                limits=(-6.2, 10) if row == 0 else (0, 0.45),
                cmap="jet" if row == 0 else "viridis",
                mesh=macro_mesh(),
            )
    plotter.screenshot(FIGURES / "layers.png")
    plotter.close()


def plot_volume(data: ReservoirData) -> None:
    """Show the complete reservoir and layer 36 in their physical geometry."""
    volume = structured_cell_grid(
        data.porosity.shape,
        cell_data={"ln(Kx / mD)": np.log(data.permeability[..., 0]), "Porosity": data.porosity},
        spacing=data.spacing,
    )
    volume.save(ROOT / "build/data/spe10/model2.vti")
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 1000))
    plotter.subplot(0, 0)
    plotter.set_background("white")
    actor = plotter.add_mesh(
        volume,
        scalars="ln(Kx / mD)",
        cmap="jet",
        clim=(-6.2, 10),
        lighting=False,
        show_scalar_bar=False,
    )
    plotter.add_mesh(volume.outline(), color="black")
    plotter.add_text("Full 60 x 220 x 85 reservoir\nDimensions: 1200 x 2200 x 170 ft", font_size=20)
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.92)
    plotter.add_axes(viewport=(0, 0.16, 0.16, 0.30))
    horizontal_color_scale(plotter, actor, "ln(Kx / mD)", scientific=False)
    grid = data_grid(36)
    for column, scalar in enumerate(("ln(Kx / mD)", "Porosity"), 1):
        plotter.subplot(0, column)
        panel(
            plotter,
            grid,
            scalar,
            "Layer 36: z in [70,72] ft\n66-square Darcy macrogrid",
            limits=(-6.2, 10) if column == 1 else (0, 0.4),
            cmap="jet" if column == 1 else "viridis",
            mesh=macro_mesh(),
        )
    plotter.screenshot(FIGURES / "volume-and-slice.png")
    plotter.close()


def main() -> None:
    """Regenerate data records and VTK screenshots; network use is explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="fetch the pinned full-volume properties"
    )
    parser.add_argument(
        "--layers-only", action="store_true", help="redraw archived 2D layers offline"
    )
    args = parser.parse_args()
    if not args.layers_only:
        if args.download:
            download_spe10_model2(CACHE)
        data = load_spe10_model2(CACHE)
        save_layers(data)
        FIGURES.mkdir(parents=True, exist_ok=True)
        plot_volume(data)
    plot_layers()


if __name__ == "__main__":
    main()
