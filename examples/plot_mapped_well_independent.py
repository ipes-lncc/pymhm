"""Render archived independent UFL and PyMHM fields for the same anisotropic well."""

from __future__ import annotations

import hashlib
import json

import matplotlib

from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import AsinhNorm, Normalize

ROOT = case_workspace()
DATA = ROOT / "examples/results/mapped-well-oscillatory"
OUTPUT = ROOT / "docs/figures/mapped-well-oscillatory"


def main() -> None:
    """Preserve independent point samples, signed differences and actual macro boundaries."""
    report = json.loads(read_resource_text(DATA / "native-discrete-verification.json"))
    display = report["display"]
    path = DATA / display["archive"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != display["sha256"]:
        raise ValueError("independent well sample archive digest mismatch")
    with np.load(local_resource(path)) as archive:
        arrays = dict(archive)
    polygons, segments = arrays["polygons"], arrays["macro_segments"]
    datum = report["physical_case"]["pressure_datum_pa"]
    fields = (
        ((arrays["pressure"] - datum) / 1e6, (arrays["pymhm_pressure"] - datum) / 1e6),
        (arrays["flux"][:, 0], arrays["pymhm_flux"][:, 0]),
        (arrays["flux"][:, 1], arrays["pymhm_flux"][:, 1]),
    )
    labels = (r"Pressure increment $p-p_e$ [MPa]", r"Flux $q_x$ [m/s]", r"Flux $q_y$ [m/s]")
    difference_labels = (r"$\Delta p$ [MPa]", r"$\Delta q_x$ [m/s]", r"$\Delta q_y$ [m/s]")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 14.8), layout="constrained")
    for row, ((native, mhm), label) in enumerate(zip(fields, labels, strict=True)):
        limits = min(native.min(), mhm.min()), max(native.max(), mhm.max())
        magnitude = max(abs(limits[0]), abs(limits[1]))
        common_norm = (
            Normalize(*limits)
            if row == 0
            else AsinhNorm(linear_width=magnitude / 10000, vmin=-magnitude, vmax=magnitude)
        )
        for column, value in enumerate((native, mhm, mhm - native)):
            ax = axes[row, column]
            error_limit = max(float(np.max(abs(value))), np.finfo(float).tiny)
            norm = (
                AsinhNorm(linear_width=error_limit / 10000, vmin=-error_limit, vmax=error_limit)
                if column == 2
                else common_norm
            )
            artist = PolyCollection(
                polygons,
                array=value,
                norm=norm,
                cmap="viridis" if row == 0 and column < 2 else "bwr",
                rasterized=True,
                antialiased=False,
                edgecolors="none",
            )
            ax.add_collection(artist)
            ax.add_collection(LineCollection(segments, colors="white", linewidths=0.8))
            ax.add_collection(LineCollection(segments, colors="black", linewidths=0.3))
            ax.set(
                xlim=(-50, 50),
                ylim=(-50, 50),
                aspect="equal",
                xlabel="x [m]",
                ylabel="y [m]",
                title=(
                    "MHM via trace restriction\nDOLFINx/UFL · RT1/Q1",
                    "PyMHM · MHM RT1/Q1",
                    "PyMHM − restricted UFL",
                )[column],
            )
            ax.set_xticks([-50, -25, 0, 25, 50])
            ax.set_yticks([-50, -25, 0, 25, 50])
            bar = fig.colorbar(artist, ax=ax, location="bottom", fraction=0.055, pad=0.075)
            bar.set_label(difference_labels[row] if column == 2 else label)
            if column == 2 or row:
                ticks = np.array([-1, -0.001, 0, 0.001, 1]) * (
                    error_limit if column == 2 else magnitude
                )
                bar.set_ticks(ticks)
                bar.set_ticklabels([f"{tick:.1e}" if tick else "0" for tick in ticks])
            else:
                bar.set_ticks(np.linspace(*limits, 3))
            bar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Same oscillatory anisotropic well · independent assembly\n"
        f"F8 · 2,048 three-dimensional macros · z = {display['plane_z_m']:g} m\n"
        "Pressure colors: linear scale; flux and differences: signed asinh scale",
        fontsize=14,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"native-comparison.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_mapped_well_independent").main()
