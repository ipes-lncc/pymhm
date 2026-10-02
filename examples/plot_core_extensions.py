"""Publication figures for exact anisotropic elasticity, MsHHO3D and enriched mixed Darcy."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from examples.plot_mesh import mark_macro_interfaces

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/core-extensions"
OUTPUT = ROOT / "docs/figures/core-extensions"
NAMES = {
    "triangle-bdm2": "Triangles · BDM2 / P1 / P1",
    "rectangle-rt1": "Rectangles · RT1 / Q1 / P1",
    "polygon-bdm2": "Nonconvex polygons · BDM2 / P1 / P1",
    "tetra-p0": "Tetrahedra · face P0 / local P2",
    "cube-p0": "Cubes · face P0 / local P2",
    "tetra-p1-k1": "Tetrahedra · normal P1 / pressure P1",
    "tetra-p2-k2": "Tetrahedra · normal P2 / pressure P2",
    "tetra-p3-k1": "Tetrahedra · normal P1 / pressure P3",
    "prism-p2-k2": "Prisms · normal P2–Q2 / pressure W2,2",
}


def save(figure: Any, name: str) -> None:
    """Export the identical scientific layout in raster and vector formats."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def convergence(suite: str, record: dict[str, Any]) -> None:
    """Show all five measured mesh levels and the observed final refinement rate."""
    keys = (
        ("displacement_l2", "stress_l2", "rotation_l2", "divergence_l2")
        if suite == "elasticity"
        else ("pressure_l2", "flux_l2")
    )
    cases = list(dict.fromkeys(row["case"] for row in record["rows"]))
    legend_height = 0.35 + 0.48 * len(cases)
    figure = plt.figure(figsize=(6 * len(keys), 5 + legend_height), layout="constrained")
    grid = figure.add_gridspec(2, len(keys), height_ratios=[5, legend_height])
    for column, key in enumerate(keys):
        axis = figure.add_subplot(grid[0, column])
        for name in cases:
            rows = [r for r in record["rows"] if r["case"] == name]
            h, errors = np.array([(1 / row["resolution"], row[key]) for row in rows]).T
            rate = np.log(errors[-2] / errors[-1]) / np.log(h[-2] / h[-1])
            axis.loglog(h, errors, "o-", label=f"{NAMES[name]}\nFinal rate {rate:.2f}")
        axis.set(
            xlabel="$H=1/n$",
            ylabel="Absolute $L^2$ error",
            title=key.replace("_l2", "").replace("_", " ").capitalize(),
        )
        axis.set_xticks(h, [f"1/{r['resolution']}" for r in rows])
        axis.minorticks_off()
        axis.grid(alpha=0.25)
        legend_axis = figure.add_subplot(grid[1, column])
        legend_axis.set_axis_off()
        handles, labels = axis.get_legend_handles_labels()
        legend_axis.legend(handles, labels, fontsize=10, loc="center", frameon=False)
    save(figure, f"{suite}-convergence")


def fields(name: str, entry: dict[str, str], suite: str) -> None:
    """Compare exact, one-sided computed and signed-error samples on identical polygons."""
    with np.load(DATA / entry["archive"]) as archive:
        data = {key: archive[key] for key in archive.files}
    polygons = data.get("polygons")
    if polygons is None and "vertices" in data:
        polygons = [
            data["vertices"][a:b]
            for a, b in zip(data["offsets"][:-1], data["offsets"][1:], strict=True)
        ]
    groups = (
        (
            ("displacement", (0, 1), ("$u_x$", "$u_y$")),
            (
                "stress",
                (2, 3, 4, 5),
                (r"$\sigma_{xx}$", r"$\sigma_{xy}$", r"$\sigma_{yx}$", r"$\sigma_{yy}$"),
            ),
        )
        if suite == "elasticity"
        else (("fields", (0, 1, 2, 3), ("$p$", "$q_x$", "$q_y$", "$q_z$")),)
    )
    for suffix, components, labels in groups:
        figure = plt.figure(figsize=(13, 4.65 * len(components)), layout="constrained")
        grid = figure.add_gridspec(
            2 * len(components),
            3,
            height_ratios=[1, 0.055] * len(components),
            hspace=0.12,
            wspace=0.11,
        )
        for row, (component, label) in enumerate(zip(components, labels, strict=True)):
            exact, actual = data["exact"][:, component], data["actual"][:, component]
            extent = float(max(abs(exact).max(), abs(actual).max()))
            signed = float(exact.min()) < 0 < float(exact.max())
            norm = (
                Normalize(-extent, extent)
                if signed
                else Normalize(
                    min(0.0, float(exact.min()), float(actual.min())),
                    max(0.0, float(exact.max()), float(actual.max())),
                )
            )
            for column, (values, title) in enumerate(
                ((exact, "Exact (sampled)"), (actual, "PyMHM"), (actual - exact, "PyMHM − exact"))
            ):
                axis, scale = (
                    figure.add_subplot(grid[2 * row, column]),
                    figure.add_subplot(grid[2 * row + 1, column]),
                )
                if column == 2:
                    maximum = max(float(abs(values).max()), np.finfo(float).tiny)
                    local_norm = Normalize(-maximum, maximum)
                else:
                    local_norm = norm
                if "cells" in data:
                    artist = axis.tripcolor(
                        mtri.Triangulation(
                            data["points"][:, 0], data["points"][:, 1], data["cells"]
                        ),
                        values,
                        shading="gouraud",
                        cmap="seismic" if signed or column == 2 else "viridis",
                        norm=local_norm,
                        rasterized=True,
                    )
                else:
                    artist = PolyCollection(
                        polygons,
                        array=values,
                        cmap="seismic" if signed or column == 2 else "viridis",
                        norm=local_norm,
                        edgecolors="none",
                        rasterized=True,
                    )
                    axis.add_collection(artist)
                axis.add_collection(
                    LineCollection(data["macro_edges"], color="#303030", linewidth=0.45, alpha=0.55)
                )
                axis.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    aspect="equal",
                    xlabel="$x$",
                    ylabel="$y$",
                    title=f"{title} · {label}",
                )
                axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
                axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
                bar = figure.colorbar(artist, cax=scale, orientation="horizontal")
                bar.locator = MaxNLocator(nbins=3)
                bar.update_ticks()
                bar.ax.tick_params(labelsize=10)
        figure.suptitle(
            NAMES[name] + ("" if suite == "elasticity" else " · section $z=0.37$"), fontsize=15
        )
        save(figure, f"{name}-{suffix}")


def run(suite: str) -> None:
    """Render only completed, source-guarded acquisitions without rerunning numerical solvers."""
    record = json.loads((DATA / f"{suite}.json").read_text())
    if record.get("source_changed") is not False:
        raise ValueError("the scientific acquisition must complete before plotting")
    convergence(suite, record)
    sampling = json.loads((DATA / f"{suite}-field-sampling.json").read_text())
    if sampling.get("source_changed", False):
        raise ValueError("polynomial display fields require a completed source-guarded replay")
    archives = sampling["cases"]
    for name, entry in archives.items():
        fields(name, entry, suite)
        if suite == "elasticity":
            profiles(name, entry)


def profiles(name: str, entry: dict[str, str]) -> None:
    """Render exact and broken displacement/stress profiles with separate physical error panels."""
    with np.load(DATA / entry["archive"]) as archive:
        data = {key: archive[key] for key in archive.files if key.startswith("profile_")}
    for category, components, symbols in (
        ("displacement", (0, 1), ("$u_x$", "$u_y$")),
        (
            "stress",
            (2, 3, 4, 5),
            (r"$\sigma_{xx}$", r"$\sigma_{xy}$", r"$\sigma_{yx}$", r"$\sigma_{yy}$"),
        ),
    ):
        figure = plt.figure(figsize=(13, 5.3 * len(components)), layout="constrained")
        grid = figure.add_gridspec(
            2 * len(components),
            2,
            height_ratios=[1, 0.55] * len(components),
            hspace=0.08,
            wspace=0.08,
        )
        for row, (component, symbol) in enumerate(zip(components, symbols, strict=True)):
            for column, (cut, description, coordinate) in enumerate(
                (("vertical", "$x=0.43$", "$y$"), ("horizontal", "$y=0.37$", "$x$"))
            ):
                prefix = f"profile_{cut}_"
                parameter = data[prefix + "parameter"]
                actual, exact = (
                    data[prefix + "actual"][..., component],
                    data[prefix + "exact"][..., component],
                )
                axis, error_axis = (
                    figure.add_subplot(grid[2 * row, column]),
                    figure.add_subplot(grid[2 * row + 1, column]),
                )
                for i, t in enumerate(parameter):
                    axis.plot(
                        t,
                        exact[i],
                        color="black",
                        linewidth=1.5,
                        label="Exact" if i == 0 else "_nolegend_",
                    )
                    axis.plot(
                        t,
                        actual[i],
                        color="#0072b2",
                        linestyle="--",
                        linewidth=1.25,
                        label="PyMHM" if i == 0 else "_nolegend_",
                    )
                    error_axis.plot(t, actual[i] - exact[i], color="#a53516", linewidth=1.0)
                for ax in (axis, error_axis):
                    mark_macro_interfaces(ax, data[prefix + "macro_breaks"][1:-1])
                    ax.set_xlim(0, 1)
                    ax.grid(axis="y", alpha=0.2)
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                axis.set(title=f"{symbol} · {description}", ylabel=symbol)
                axis.legend(loc="best", fontsize=10)
                error_axis.set(xlabel=coordinate, ylabel="Signed error")
                error_axis.ticklabel_format(
                    axis="y", style="sci", scilimits=(0, 0), useMathText=True
                )
        figure.suptitle(NAMES[name], fontsize=14)
        save(figure, f"{name}-{category}-profiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("elasticity", "mshho3d", "hdiv3d"))
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    run(parser.parse_args().suite)
