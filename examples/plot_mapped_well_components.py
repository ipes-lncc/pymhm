"""Plot signed horizontal flux components for the oscillatory producing well.

The physical fields are sampled independently on the reference-cell centroids.
An odd asinh color normalization displays their full range without clipping;
reference and MHM share each component's scale. Macro edges are the actual
intersections with the upper incident side of the horizontal plane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from pymhm.io.workspace import read_resource_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import AsinhNorm

from examples.plot_mapped_oscillatory_well import (
    DATA,
    OUTPUT,
    macro_segments,
    plane_values,
    read,
    slice_values,
)


def flux_components(reference_name: str, *, fine: int, output: Path = OUTPUT) -> None:
    """Compare qx and qy on matched physical samples with explicit signed differences."""
    reference, reference_record, _ = read(reference_name)
    field, record, arrays = read(f"fine{fine}-macro4-s1-q40z10")
    _, polygons, _, _, reference_flux = slice_values(reference)
    _, mhm_flux = plane_values(field, reference)
    segments = macro_segments(arrays)
    figure = plt.figure(figsize=(15.6, 10.2), layout="constrained")
    grid = figure.add_gridspec(2, 5, width_ratios=(1, 1, 0.055, 1, 0.055))
    scales = []
    for row, component in enumerate(("x", "y")):
        values = (
            reference_flux[:, row],
            mhm_flux[:, row],
            mhm_flux[:, row] - reference_flux[:, row],
        )
        extent = max(float(np.abs(values[0]).max()), float(np.abs(values[1]).max()))
        difference_extent = float(np.abs(values[2]).max())
        if extent == 0 or difference_extent == 0:
            raise ValueError(
                "this heterogeneous field comparison requires nonzero component ranges"
            )
        for column, (value, index) in enumerate(zip(values, (0, 1, 3), strict=True)):
            axis = figure.add_subplot(grid[row, index])
            limit = extent if column < 2 else difference_extent
            artist = PolyCollection(
                polygons,
                array=value,
                cmap="RdBu_r",
                norm=AsinhNorm(linear_width=limit / 10_000, vmin=-limit, vmax=limit),
                edgecolors="none",
                antialiased=False,
                rasterized=True,
            )
            axis.add_collection(artist)
            axis.add_collection(LineCollection(segments, colors="white", linewidths=0.9))
            axis.add_collection(LineCollection(segments, colors="black", linewidths=0.3))
            axis.set(
                xlim=(-50, 50),
                ylim=(-50, 50),
                aspect="equal",
                xlabel="x [m]",
                ylabel="y [m]",
                title=("Classical RT1", f"MHM RT1 · F{fine}", "MHM − reference")[column]
                + rf" · $q_{component}$",
            )
            if column:
                color_axis = figure.add_subplot(grid[row, 2 if column == 1 else 4])
                ticks = limit * np.array([-1, -0.01, -0.0001, 0, 0.0001, 0.01, 1])
                colorbar = figure.colorbar(artist, cax=color_axis, ticks=ticks)
                colorbar.ax.set_yticklabels(
                    ["0" if value == 0 else f"{value:.1e}" for value in ticks]
                )
                colorbar.set_label(
                    rf"{'Flux' if column == 1 else 'Flux difference'} $q_{component}$ [m/s] · asinh"
                )
        scales.append(
            {"component": component, "field_extent": extent, "difference_extent": difference_extent}
        )
    figure.suptitle(
        f"Signed flux components · {record['macro_cells']:,} actual macro hexahedra\n"
        "Upper incident side at z=0 · full ranges · shared reference/MHM scales",
        fontsize=13,
    )
    output.mkdir(parents=True, exist_ok=True)
    name = f"flux-components-fine{fine}"
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180)
    plt.close(figure)
    (output / f"{name}.json").write_text(
        json.dumps(
            {
                "reference": reference_name,
                "reference_archive_sha256": reference_record["sha256"],
                "candidate_archive_sha256": record["sha256"],
                "sampling": "Reference fine-cell centroids, upper incident side z=0; no averaging",
                "color_normalization": "asinh with linear width extent/10000; full signed ranges",
                "scales": scales,
                "maximum_sampled_qz": {
                    "reference": float(np.abs(reference_flux[:, 2]).max()),
                    "mhm": float(np.abs(mhm_flux[:, 2]).max()),
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Render both archived MHM local resolutions against the selected reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--fine", type=int, nargs="+", choices=(8, 16), default=[8, 16])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    reference = (
        args.reference or json.loads(read_resource_text(DATA / "comparisons.json"))["reference"]
    )
    for fine in args.fine:
        flux_components(reference, fine=fine, output=args.output)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_mapped_well_components").main()
