"""Render archived independent comparisons of the oscillatory elasticity case."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]


def field_panels(
    data: dict[str, np.ndarray], fields: list[tuple[str, str, tuple[int, ...]]], name: str
) -> None:
    """Draw independent one-sided polynomials with separate physical color scales."""
    points = data["points"]
    cells = data["triangles"][None, :, :] + np.arange(len(points))[:, None, None] * points.shape[1]
    triangulation = mtri.Triangulation(*points.reshape(-1, 2).T, cells.reshape(-1, 3))
    macro_cells = data["macro_cells"]
    edges = np.unique(
        np.sort(
            np.concatenate(
                [macro_cells[:, [0, 1]], macro_cells[:, [1, 2]], macro_cells[:, [2, 0]]]
            ),
            axis=1,
        ),
        axis=0,
    )
    segments = data["macro_points"][edges]
    fig = plt.figure(figsize=(13.8, 4.4 * len(fields)), layout="constrained")
    grid = fig.add_gridspec(
        2 * len(fields), 3, height_ratios=[1, 0.055] * len(fields), hspace=0.16, wspace=0.12
    )
    for row, (key, label, component) in enumerate(fields):
        selector = (slice(None), slice(None), *component)
        native, current, delta = [
            data[prefix + key][selector].ravel() for prefix in ("", "pymhm_", "difference_")
        ]
        physical_limit = max(float(np.max(abs(native))), float(np.max(abs(current))))
        difference_limit = max(float(np.max(abs(delta))), np.finfo(float).tiny)
        for column, (values, title) in enumerate(
            (
                (native, "MHM via trace restriction · UFL"),
                (current, "PyMHM"),
                (-delta, "PyMHM − restricted UFL"),
            )
        ):
            ax = fig.add_subplot(grid[2 * row, column])
            limit = difference_limit if column == 2 else physical_limit
            artist = ax.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap="bwr",
                norm=TwoSlopeNorm(0, vmin=-limit, vmax=limit),
                rasterized=True,
            )
            ax.add_collection(LineCollection(segments, colors="black", linewidths=0.35, alpha=0.6))
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="$x$",
                ylabel="$y$",
                title=f"{title}\n{label}",
            )
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            scale = fig.add_subplot(grid[2 * row + 1, column])
            bar = fig.colorbar(
                artist, cax=scale, orientation="horizontal", ticks=[-limit, 0, limit]
            )
            bar.ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.2e}")
            )
            bar.ax.tick_params(labelsize=9)
    output = ROOT / "docs/figures/mixed-elasticity"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=240)
    plt.close(fig)


def main() -> None:
    """Render field comparisons from numerical archives without executing reference code."""
    with np.load(
        ROOT / "examples/results/mixed-elasticity/native-comparison-fields.npz"
    ) as archive:
        data = {key: archive[key] for key in archive.files}
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11}):
        field_panels(
            data,
            [
                ("displacement", "$u_x$", (0,)),
                ("displacement", "$u_y$", (1,)),
                ("rotation", "$q$", ()),
            ],
            "native-displacement-rotation",
        )
        field_panels(
            data,
            [
                ("stress", "$\\sigma_{xx}$", (0, 0)),
                ("stress", "$\\sigma_{xy}$", (0, 1)),
                ("stress", "$\\sigma_{yx}$", (1, 0)),
                ("stress", "$\\sigma_{yy}$", (1, 1)),
            ],
            "native-stress",
        )


if __name__ == "__main__":
    main()
