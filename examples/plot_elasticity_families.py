"""Replay mixed-elasticity family fields, convergence and modulus sweeps from archives."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import hashlib
import json

import matplotlib

from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
    resource_glob,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from matplotlib.tri import Triangulation

from examples.plot_mesh import draw_macro_mesh
from pymhm import TriangleMesh

ROOT = case_workspace()
DATA = ROOT / "examples/results/elasticity-families"
FIGURES = ROOT / "docs/figures/elasticity-families"


def save(figure: Figure, name: str) -> None:
    """Export the same laid-out figure in PNG and SVG with rasterized dense fields."""
    for extension in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def fields(record: dict, name: str) -> None:
    """Use shared exact/numerical limits and preserve independent values per fine cell."""
    path = DATA / record["archive"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != record["sha256"]:
        raise ValueError("Mixed-elasticity field archive checksum mismatch")
    with np.load(local_resource(path)) as data:
        triangulation = Triangulation(*data["points"].T, data["cells"])
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
        for row, (key, component, label) in enumerate(
            (("displacement", (0,), "Displacement uₓ"), ("stress", (0, 0), "Cauchy stress σₓₓ"))
        ):
            numerical = data[key][(slice(None), *component)]
            exact = data["exact_" + key][(slice(None), *component)]
            difference = numerical - exact
            extent = max(np.max(np.abs(difference)), np.finfo(float).eps)
            for column, values in enumerate((exact, numerical, difference)):
                options = (
                    dict(
                        cmap="viridis",
                        vmin=min(exact.min(), numerical.min()),
                        vmax=max(exact.max(), numerical.max()),
                    )
                    if column < 2
                    else dict(cmap="RdBu_r", vmin=-extent, vmax=extent)
                )
                artist = axes[row, column].tripcolor(
                    triangulation, values, shading="gouraud", rasterized=True, **options
                )
                edges = draw_macro_mesh(axes[row, column], macro)
                edges.set_linewidth(0.3)
                edges.set_alpha(0.4)
                axes[row, column].set(
                    aspect="equal",
                    xlabel="x",
                    ylabel="y",
                    title=f"{label}: {('exact', 'MHM', 'MHM − exact')[column]}",
                )
                figure.colorbar(artist, ax=axes[row, column], shrink=0.8, pad=0.02)
        last = record["rows"][-1]
        figure.suptitle(
            f"BDM normal degree {last['stress_degree']}, interior enrichment {last['enrichment']}\n"
            f"L² displacement error {last['displacement_l2']:.3e}; "
            f"stress error {last['stress_l2']:.3e}"
        )
        save(figure, name + "-fields")


def studies(records: list[dict]) -> None:
    """Compare each family's refinement and finite/infinite-modulus behavior separately."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    sweep, saxes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    quantities = [
        ("displacement_l2", "Displacement L² error"),
        ("stress_l2", "Stress L² error"),
        ("rotation_l2", "Rotation L² error"),
    ]
    for record in records:
        rows = record["rows"]
        k, n = rows[0]["stress_degree"], rows[0]["enrichment"]
        label = f"BDM{k}" + ("+" * n)
        h = 1 / np.array([r["resolution"] for r in rows])
        for axis, (key, title) in zip(axes, quantities, strict=True):
            axis.loglog(h, [r[key] for r in rows], "o-", label=label)
            axis.set(xlabel="Macro spacing 1/n", ylabel=title)
        for axis, (key, title) in zip(saxes, quantities, strict=True):
            axis.semilogy(
                np.arange(len(record["sweep"])),
                [r[key] for r in record["sweep"]],
                "o-",
                label=label,
            )
            axis.set(
                xlabel="First Lamé modulus λ (μ=1)",
                ylabel=title,
                xticks=np.arange(5),
                xticklabels=["1", "10²", "10⁴", "10⁸", "∞"],
            )
    for axis in (*axes, *saxes):
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend()
    saxes[0].legend()
    figure.suptitle(
        "Original bounded-force polynomial problem — each family uses its declared trace degree"
    )
    sweep.suptitle("Fixed mesh: eight macrotriangles — finite and exact incompressible solutions")
    save(figure, "convergence")
    save(sweep, "incompressible-sweep")


def oscillatory_studies() -> None:
    """Display the nominal four-level L18 problem without fitting historical table entries."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    quantities = [
        ("displacement_l2", "Displacement L² error"),
        ("stress_l2", "Stress L² error"),
        ("stress_divergence_l2", "Stress divergence L² error"),
    ]
    for path in sorted(resource_glob(DATA, "l18-*.json")):
        record = json.loads(read_resource_text(path))
        rows = record["rows"]
        k = rows[0]["trace_degree"]
        degree = rows[0]["local_normal_degree"]
        enrichment = rows[0]["interior_degree"] - degree
        label = f"Trace P{k}, BDM{degree}" + ("+" * enrichment)
        for axis, (key, title) in zip(axes, quantities, strict=True):
            axis.loglog(
                [row["h_sk"] for row in rows], [row[key] for row in rows], "o-", label=label
            )
            axis.set(
                xlabel="Nominal skeletal spacing h_sk",
                ylabel=title,
                xticks=[1 / 32, 1 / 16, 1 / 8, 1 / 4],
                xticklabels=["1/32", "1/16", "1/8", "1/4"],
            )
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("2021 oscillatory-modulus problem — 32 diagonal macrotriangles, h_in = h_sk/2")
    save(figure, "oscillatory-convergence")


def main() -> None:
    """Render completed polynomial campaign records without executing a solver."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(resource_glob(DATA, "bdm*.json")):
        record = json.loads(read_resource_text(path))
        records.append(record)
        fields(record, path.stem)
    studies(records)
    oscillatory_studies()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_elasticity_families").main()
