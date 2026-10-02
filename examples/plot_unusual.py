"""Render archived scalar UNUSUAL evidence with exact fields and real macrofaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from plot_mesh import draw_macro_mesh, mark_macro_interfaces
from plot_style import set_refinement_ticks

from pymhm import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/results/unusual"
TARGET = ROOT / "docs/figures/unusual"
COLORS = {"galerkin": "#a94929", "unusual": "#146d83"}


def save(figure: plt.Figure, name: str) -> None:
    """Export identical readable raster and vector figures."""
    figure.savefig(TARGET / f"{name}.png", dpi=190)
    figure.savefig(TARGET / f"{name}.svg")
    plt.close(figure)


def read_field(row: dict) -> dict[str, np.ndarray]:
    """Verify a field archive before displaying its unmodified data."""
    path = SOURCE / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError(f"archive checksum mismatch: {path.name}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def convergence(rows: list[dict], high_order: list[dict]) -> None:
    """Compare absolute physical errors over five actual macro refinements."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout="constrained")
    for column, (case, title) in enumerate(
        (
            ("smooth", "Smooth: epsilon=1, P1/P0"),
            ("layer", "Layer: epsilon=0.001, P1/P0"),
            ("tensor", "Variable anisotropic tensor: P2/P1"),
        )
    ):
        for method in COLORS:
            selected = [row for row in rows if row["case"] == case and row["method"] == method]
            n = [row["n"] for row in selected]
            for index, metric in enumerate(("scalar_l2", "flux_l2")):
                axes[index, column].loglog(
                    n,
                    [row[metric] for row in selected],
                    "o-",
                    color=COLORS[method],
                    label="MHM-UNUSUAL" if method == "unusual" else "MHM-Galerkin",
                )
                set_refinement_ticks(axes[index, column], n)
        axes[0, column].set_title(title, fontsize=11)
        for index, ylabel in enumerate((r"$\|u-u_h\|_{L^2}$", r"$\|q-q_h\|_{L^2}$")):
            axes[index, column].set_ylabel(ylabel)
            axes[index, column].set_xlabel("Macro divisions per side")
            axes[index, column].grid(alpha=0.25, which="major")
    for index, metric in enumerate(("scalar_l2", "flux_l2")):
        axes[index, 1].loglog(
            [row["n"] for row in high_order],
            [row[metric] for row in high_order],
            "s--",
            color="#794d9a",
            label="MHM-UNUSUAL P3/P2",
        )
    axes[0, 1].legend(fontsize=8)
    axes[0, 0].legend(fontsize=9)
    save(figure, "convergence")


def spatial(rows: list[dict], case: str, quantity: str, name: str) -> None:
    """Show exact, numerical and signed-error fields with independent color scales."""
    selected = [row for row in rows if row["case"] == case and "archive" in row]
    figure, axes = plt.subplots(
        len(selected), 3, figsize=(12, 3.6 * len(selected)), layout="constrained", squeeze=False
    )
    for index, row in enumerate(selected):
        data = read_field(row)
        triangulation = mtri.Triangulation(*data["points"].T, data["cells"])
        mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
        if quantity == "scalar":
            exact, numerical = data["exact"], data["values"]
            label = "Scalar u"
        else:
            exact, numerical = data["exact_flux"][:, 0], data["flux"][:, 0]
            label = "Physical flux q_x"
        lo, hi = min(exact.min(), numerical.min()), max(exact.max(), numerical.max())
        difference = numerical - exact
        bound = float(abs(difference).max())
        for column, (field, title) in enumerate(
            ((exact, "Analytical"), (numerical, "PyMHM"), (difference, "Signed difference"))
        ):
            axis = axes[index, column]
            artist = axis.tripcolor(
                triangulation,
                field,
                shading="gouraud",
                cmap="RdBu_r" if column == 2 else "viridis",
                vmin=-bound if column == 2 else lo,
                vmax=bound if column == 2 else hi,
                rasterized=True,
            )
            draw_macro_mesh(axis, mesh)
            axis.set(xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal")
            method = "MHM-UNUSUAL" if row["method"] == "unusual" else "MHM-Galerkin"
            axis.set_title(
                f"{method} P{row['local_degree']}/P{row['trace_degree']}\n{title}", fontsize=10
            )
            figure.colorbar(
                artist, ax=axis, shrink=0.85, label=label if column < 2 else f"Delta {label}"
            )
    save(figure, name)


def profiles(rows: list[dict], high_order: list[dict]) -> None:
    """Preserve broken one-sided profiles and compare nodal layer errors without smoothing."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), layout="constrained")
    for column, case in enumerate(("layer", "reaction-sweep")):
        selected = [row for row in rows if row["case"] == case and "archive" in row]
        for row in selected:
            data = read_field(row)
            for index, (x, value, exact) in enumerate(
                zip(
                    data["profile_parameter"],
                    data["profile_values"],
                    data["profile_exact"],
                    strict=True,
                )
            ):
                label = "MHM-UNUSUAL" if row["method"] == "unusual" else "MHM-Galerkin"
                axes[column].plot(
                    x,
                    value,
                    color=COLORS[row["method"]],
                    lw=1.2,
                    label=label if index == 0 else None,
                )
                if row["method"] == "unusual":
                    axes[column].plot(
                        x,
                        exact,
                        color="#222222",
                        ls="--",
                        lw=1.2,
                        label="Analytical" if index == 0 else None,
                    )
            mark_macro_interfaces(axes[column], data["profile_breaks"])
        if case == "layer":
            field = read_field(next(row for row in high_order if "archive" in row))
            for index, (x, value) in enumerate(
                zip(field["profile_parameter"], field["profile_values"], strict=True)
            ):
                axes[column].plot(
                    x,
                    value,
                    color="#794d9a",
                    lw=1.2,
                    label="MHM-UNUSUAL P3/P2" if index == 0 else None,
                )
        epsilon = selected[0]["epsilon"]
        axes[column].set(
            title=f"One-sided profile: epsilon={epsilon:g}", xlabel="x at y=0.37", ylabel="Scalar u"
        )
        axes[column].legend(fontsize=8)
    for method in COLORS:
        selected = [
            row for row in rows if row["case"] == "reaction-sweep" and row["method"] == method
        ]
        epsilon = [row["epsilon"] for row in selected]
        axes[2].loglog(
            epsilon,
            [row["nodal_error"] for row in selected],
            "o-",
            color=COLORS[method],
            label=method,
        )
    axes[2].set(
        xlabel="Diffusivity epsilon",
        ylabel="Maximum error at local nodes",
        title="Fixed macro/local spaces; reaction=1",
    )
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    save(figure, "profiles")


def native() -> None:
    """Display independent-code agreement separately from discretization accuracy."""
    record = json.loads((SOURCE / "native-comparison.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    for degree in (2, 3):
        rows = [row for row in record["rows"] if row["local_degree"] == degree]
        n = [row["n"] for row in rows]
        for axis, metric, label in zip(
            axes,
            ("pressure_relative_difference", "flux_relative_difference"),
            ("Relative scalar L2 difference", "Relative physical-flux L2 difference"),
            strict=True,
        ):
            axis.loglog(
                n,
                [row[metric] for row in rows],
                "o-",
                label=f"Local P{degree}, trace P{degree - 2}",
            )
            set_refinement_ticks(axis, n)
            axis.set(xlabel="Macro divisions per side", ylabel=label)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=9)
    figure.suptitle("PyMHM versus MHMUN-RAD_Parallel / FreeFem++", fontsize=12)
    save(figure, "native-comparison")


def main() -> None:
    """Render reproducible figures from verified numerical archives only."""
    TARGET.mkdir(parents=True, exist_ok=True)
    rows = json.loads((SOURCE / "analytical.json").read_text())["rows"]
    high_order = json.loads((SOURCE / "resolution-control.json").read_text())["rows"]
    convergence(rows, high_order)
    spatial(rows, "layer", "scalar", "layer-fields")
    spatial(rows, "tensor", "flux", "tensor-flux")
    spatial(high_order, "layer-high-order", "flux", "resolved-layer-flux")
    profiles(rows, high_order)
    native()
    for name in ("analytical.json", "native-comparison.json", "resolution-control.json"):
        (TARGET / name).write_bytes((SOURCE / name).read_bytes())


if __name__ == "__main__":
    main()
