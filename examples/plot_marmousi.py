"""Render primary material and archived complex acoustic fields without new solves."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm
from matplotlib.ticker import FuncFormatter

from examples.marmousi_data import load_marmousi_crop
from examples.marmousi_fields import load_reference
from examples.plot_mesh import draw_macro_mesh
from pymhm.quadrilateral import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = (0, 10240, 0, 2560)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 12})


def panel(
    axis: Any,
    values: np.ndarray,
    title: str,
    label: str,
    macro: CartesianMacroMesh,
    *,
    norm: AsinhNorm | None = None,
    limits: tuple[float, float] | None = None,
    nodal: bool = False,
) -> None:
    """Reserve separate horizontal color scales and retain the declared comparison grid."""
    dx, dy = (10240 / (values.shape[0] - 1), 2560 / (values.shape[1] - 1)) if nodal else (0, 0)
    artist = axis.imshow(
        np.asarray(values).T,
        origin="upper",
        extent=(-dx / 2, 10240 + dx / 2, 2560 + dy / 2, -dy / 2),
        interpolation="nearest",
        cmap="RdBu_r" if norm is not None else "viridis",
        norm=norm,
        vmin=None if norm is not None else limits[0],
        vmax=None if norm is not None else limits[1],
        rasterized=True,
    )
    lines = draw_macro_mesh(axis, macro)
    lines.set_path_effects([])
    lines.set_rasterized(True)
    lines.set_linewidth(0.07 if macro.nx > 128 else 0.17)
    lines.set_alpha(0.25 if macro.nx > 128 else 0.4)
    axis.plot(5000, 50, "*", color="black", markeredgecolor="white", markersize=6, zorder=4)
    axis.set(title=title, xlabel="Horizontal distance (km)", ylabel="Depth (km)")
    axis.set_xlim(0, 10240)
    axis.set_ylim(2560, 0)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axis.set_xticks([0, 5000, 10000])
    axis.set_yticks([0, 1000, 2000])
    bar = axis.figure.colorbar(
        artist, ax=axis, location="bottom", pad=0.25, fraction=0.075, aspect=25
    )
    bar.set_label(label)
    if norm is not None:
        # Equal display-space separation keeps physical-value labels apart
        # for both nearly linear small ranges and logarithmic large ranges.
        bar.set_ticks(norm.inverse(np.linspace(0.0, 1.0, 5)))
        bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))


def save(figure: plt.Figure, directory: Path, name: str) -> None:
    """Export both a publication raster and a vector file with rasterized dense maps."""
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Plot complete declared material and compare acquired classical pressure levels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "examples/results/marmousi")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/marmousi")
    parser.add_argument(
        "--degrees",
        type=int,
        nargs=2,
        metavar=("COARSE", "FINE"),
        help="Two acquired degrees to display (default: the two highest complete levels)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    material = load_marmousi_crop(args.data)
    available = [
        degree for degree in range(1, 5) if (args.source / f"classical-p{degree}.json").exists()
    ]
    degrees = args.degrees or available[-2:]
    if len(degrees) != 2 or degrees[0] >= degrees[1] or any(d not in available for d in degrees):
        parser.error("two increasing acquired polynomial degrees are required")
    records = [args.source / f"classical-p{degree}.json" for degree in degrees]
    fields = [load_reference(path) for path in records]
    # Retain every archived node, including the source node absent from the
    # article's coarser sampling grid. Evaluate differences on the finer grid.
    values = [field.nodes for field in fields]
    difference = np.empty_like(values[1])
    for start in range(0, difference.size, 65536):
        index = np.arange(start, min(start + 65536, difference.size))
        coordinates = np.column_stack((index // difference.shape[1], index % difference.shape[1]))
        points = coordinates * (fields[1].spacing / fields[1].degree)
        difference.ravel()[index] = values[1].ravel()[index] - fields[0].sample(points)
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 6.8), layout="constrained")
    for row, H in enumerate((20, 80)):
        macro = CartesianMacroMesh(10240 // H, 2560 // H, BOUNDS)
        for column, (data, title, label) in enumerate(
            (
                (material.velocity.values, "P-wave velocity", "m/s"),
                (material.density.values, "Density", "kg/m³"),
            )
        ):
            panel(
                axes[row, column],
                data,
                f"{title}; comparison grid H={H} m",
                label,
                macro,
                limits=(float(data.min()), float(data.max())),
            )
    save(figure, args.output, "material")
    for H in (20, 80):
        macro = CartesianMacroMesh(10240 // H, 2560 // H, BOUNDS)
        figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.8), layout="constrained")
        for row, component in enumerate((np.real, np.imag)):
            parts = [component(value) for value in values]
            parts.append(component(difference))
            common = max(float(np.max(abs(value))) for value in parts[:2])
            for column, value in enumerate(parts):
                extent = common if column < 2 else float(np.max(abs(value)))
                norm = AsinhNorm(linear_width=1.0, vmin=-extent, vmax=extent)
                panel(
                    axes[row, column],
                    value,
                    (
                        f"Classical P{degrees[0]}",
                        f"Classical P{degrees[1]}",
                        f"P{degrees[1]} − P{degrees[0]}",
                    )[column],
                    ("Re p", "Im p")[row] + "; asinh display",
                    macro,
                    norm=norm,
                    nodal=True,
                )
        figure.suptitle(f"Declared Marmousi crop; H={H} m comparison partition", fontsize=15)
        save(figure, args.output, f"classical-fields-H{H}")
    x = np.linspace(0, 10240, 4097)
    profile_points = np.column_stack((x, np.full_like(x, 500)))
    profiles = [field.sample(profile_points) for field in fields]
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), layout="constrained")
    for axis, component, label in zip(axes, (np.real, np.imag), ("Re p", "Im p"), strict=True):
        for degree, values_on_line in zip(degrees, profiles, strict=True):
            axis.plot(x / 1000, component(values_on_line), label=f"Classical P{degree}", lw=1)
        for crossing in np.arange(80, 10240, 80):
            axis.axvline(crossing / 1000, color="#66737a", lw=0.4, alpha=0.22, zorder=0)
        axis.set(
            xlabel="Horizontal distance (km)",
            ylabel=label,
            title="Depth 500 m; H=80 m comparison interfaces",
        )
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(figure, args.output, "classical-profiles")
    provenance = {
        "material": material.provenance,
        "reference_records": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in records
        },
        "display_grid": (
            f"All native nodal pressure coordinates; differences on the P{degrees[1]} nodal grid"
        ),
        "displayed_degrees": degrees,
        "sampling_norm_grid": [513, 129],
        "displayed_nodal_max_abs": [float(np.max(abs(value))) for value in values],
        "pressure_display": "Signed asinh transform, linear_width=1; physical-value color ticks",
        "macro_overlay": "Declared H20/H80 comparison partitions, not additional numerical solves",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {
            f"examples/{name}.py": hashlib.sha256(
                Path(__file__).with_name(f"{name}.py").read_bytes()
            ).hexdigest()
            for name in ("marmousi_data", "marmousi_fields", "plot_mesh")
        },
    }
    (args.output / "display.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for source in [
        *args.source.glob("classical-p[1-4].json"),
        args.source / "classical-convergence.json",
    ]:
        if source.exists():
            shutil.copyfile(source, args.output / source.name)


if __name__ == "__main__":
    main()
