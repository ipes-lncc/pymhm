"""Show a centered material interface cutting the actual macrotriangle interiors."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from examples.plot_quarter_reference import FIGURES, geometry_record, problem_geometry
from examples.quarter_spot_problem import OBSTACLE_LOWER, OBSTACLE_UPPER, macro_mesh


def geometry_detail() -> None:
    """Contrast the unfitted macro mesh and fitted fine mesh without averaging K."""
    record = geometry_record()
    macro = macro_mesh()
    fine_edges = record["points"][record["cells"][:, [0, 1, 1, 2, 2, 0]]].reshape(-1, 2, 2)
    macro_segments = macro.points[macro.faces]
    boundary = np.array(
        [
            [OBSTACLE_LOWER, OBSTACLE_LOWER],
            [OBSTACLE_UPPER, OBSTACLE_LOWER],
            [OBSTACLE_UPPER, OBSTACLE_UPPER],
            [OBSTACLE_LOWER, OBSTACLE_UPPER],
            [OBSTACLE_LOWER, OBSTACLE_LOWER],
        ]
    )
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "savefig.dpi": 220}):
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 6.8))
        figure.subplots_adjust(left=0.065, right=0.97, top=0.90, bottom=0.27, wspace=0.25)
        for index, axis in enumerate(axes):
            artist = axis.tripcolor(
                record["points"][:, 0],
                record["points"][:, 1],
                record["cells"],
                facecolors=record["permeability"],
                cmap="viridis",
                norm=LogNorm(1e-4, 1),
                rasterized=True,
            )
            if index == 1:
                axis.add_collection(
                    LineCollection(fine_edges, colors="#667788", linewidths=0.5, alpha=0.6)
                )
            axis.add_collection(LineCollection(macro_segments, colors="#17212b", linewidths=1.05))
            line = axis.plot(*boundary.T, color="white", linewidth=2.0)[0]
            line.set_path_effects(
                [effects.Stroke(linewidth=3.8, foreground="#b81825"), effects.Normal()]
            )
            axis.set(aspect="equal", xlabel="$x$", ylabel="$y$")
        axes[0].set(xlim=(0, 1), ylim=(0, 1), title="(a) Centered square: 25% of the domain")
        axes[0].add_patch(
            Rectangle((0.19, 0.39), 0.12, 0.12, fill=False, edgecolor="#e33b26", linewidth=2)
        )
        axes[1].set(
            xlim=(0.19, 0.31),
            ylim=(0.39, 0.51),
            xticks=[0.20, 0.25, 0.30],
            yticks=[0.40, 0.45, 0.50],
            title="(b) Material interface inside two macrotriangles",
        )
        figure.legend(
            handles=[
                Line2D([], [], color="#17212b", linewidth=1.5, label="Macro edges"),
                Line2D([], [], color="#667788", linewidth=0.8, label="Fine edges ($r=4$)"),
                Line2D([], [], color="#b81825", linewidth=2.5, label="Material interface"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.125),
            ncols=3,
            frameon=False,
        )
        color_axis = figure.add_axes((0.25, 0.085, 0.50, 0.023))
        colorbar = figure.colorbar(artist, cax=color_axis, orientation="horizontal")
        colorbar.set_ticks([1e-4, 1e-2, 1])
        colorbar.set_label("Permeability $K$ (logarithmic scale)", labelpad=3)
        FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            figure.savefig(FIGURES / f"obstacle-macro-cut.{suffix}")
        plt.close(figure)


def main() -> None:
    """Render the benchmark geometry without loading or solving numerical fields."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    problem_geometry()
    geometry_detail()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_quarter_geometry").main()
