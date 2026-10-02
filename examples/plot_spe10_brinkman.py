"""Render the declared 2017 SPE10 USFEM spaces and material-cut integration check."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from plot_mesh import mark_macro_interfaces
from plot_spe10_data import FIGURES, OUTPUT, data_grid, panel

from pymhm import TriangleMesh
from pymhm.visualization import broken_triangle_grid


def run(layer: int = 1) -> None:
    """Display each broken P3 field with the actual crisscross macro edges."""
    records = sorted(
        [
            json.loads(path.read_text())
            for path in OUTPUT.glob(f"flow-layer{layer}-n6x11-p3-r10-s10-q*-pointwise-2017.json")
        ],
        key=lambda row: row["quadrature_order"],
    )
    if not records:
        raise ValueError("Run the declared 2017 P3/P3, r=10, ten-segment SPE10 configuration first")
    row = next(record for record in records if record["quadrature_order"] == 5)
    with np.load(OUTPUT / row["archive"]) as data:
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        meshes = tuple(
            TriangleMesh(points, cells)
            for points, cells in zip(data["local_points"], data["local_cells"], strict=True)
        )
        state = np.concatenate((data["local_velocity"], data["local_pressure"][..., None]), axis=-1)
        # A multiple of three includes every P3 interpolation node in the view.
        grid = broken_triangle_grid(meshes, state, degree=3, name="State", subdivision=6)
        grid.point_data["Velocity magnitude"] = np.linalg.norm(grid["State"][:, :2], axis=1)
        grid.point_data["Pressure"] = grid["State"][:, 2]
        bounds = data["profile_breaks"].copy()
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 1000))
    for column, (field, scalar, title) in enumerate(
        (
            (data_grid(layer), "ln(Kx / mD)", f"Model 2, layer {layer} / material"),
            (grid, "Velocity magnitude", "2017 USFEM P3/P3: velocity"),
            (grid, "Pressure", "2017 USFEM P3/P3: pressure"),
        )
    ):
        plotter.subplot(0, column)
        panel(
            plotter,
            field,
            scalar,
            title
            + f"\n{len(macro.cells)} macros; 100 cells/macro"
            + "\nDG P1, 10 segments/face"
            + "\nPointwise 2017 parameter",
            mesh=macro,
            cmap="jet",
        )
    plotter.screenshot(FIGURES / f"brinkman-layer{layer}.png")
    plotter.close()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for record in records:
        with np.load(OUTPUT / record["archive"]) as data:
            for component, axis in enumerate(axes):
                values = (
                    data["profile_pressure"]
                    if component == 0
                    else np.linalg.norm(data["profile_velocity"], axis=-1)
                )
                for index in range(len(values)):
                    axis.plot(
                        data["profile_points"][index, :, 1],
                        values[index],
                        color=plt.cm.viridis((record["quadrature_order"] - 5) / 20),
                        lw=1,
                        label=f"Quadrature order {record['quadrature_order']}"
                        if index == 0
                        else None,
                    )
    for axis, variable in zip(axes, ("Pressure", "Velocity magnitude"), strict=True):
        mark_macro_interfaces(axis, bounds * 2200)
        axis.set(xlabel="y at x=199 [ft]", ylabel=variable)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    fig.suptitle(
        "Exact material intersections on the same broken P3/P3 space\n"
        "Diagnostic cut x=199 chosen here; not a pressure profile specified in the 2017 article"
    )
    fig.savefig(FIGURES / f"brinkman-layer{layer}-profiles.svg")
    fig.savefig(FIGURES / f"brinkman-layer{layer}-profiles.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
