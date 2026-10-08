"""Render pressure jumps and five-level Stokes refinement without smoothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

from pymhm.io.workspace import read_resource_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from examples.plot_mesh import macro_profile_breaks, mark_macro_interfaces
from pymhm import TriangleMesh

LABELS = {"usfem": "USFEM P1/P1", "taylor-hood": "Taylor–Hood P2/P1"}
COLORS = {"usfem": "#c85710", "taylor-hood": "#006a91"}


def save(figure: Figure, name: str) -> None:
    """Write a vector figure and matching raster preview."""
    directory = Path("docs/figures/flow-audit")
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.svg")
    figure.savefig(directory / f"{name}.png", dpi=170)
    plt.close(figure)


def convergence(data: dict[str, Any]) -> None:
    """Identify PyMHM and DOLFINx explicitly in the conforming-reference comparison."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for method, label in LABELS.items():
        rows = [r for r in data["H_refinement"] if r["formulation"] == method]
        h = 1 / np.array([r["n"] for r in rows])
        for axis, key in zip(
            axes, ("velocity_l2", "pressure_l2", "pressure_jump_max_sampled"), strict=True
        ):
            axis.loglog(
                h, [r[key] for r in rows], "o-", color=COLORS[method], label="PyMHM " + label
            )
        reference = [r for r in data["conforming_reference"] if r["formulation"] == method]
        if reference:
            for axis, key in zip(axes[:2], ("velocity_l2", "pressure_l2"), strict=True):
                axis.loglog(
                    h,
                    [r[key] for r in reference],
                    ":",
                    color=COLORS[method],
                    alpha=0.8,
                    label="DOLFINx conforming " + label,
                )
    for axis, title, ylabel in zip(
        axes,
        ("Velocity error", "Pressure error", "Pressure jump on skeleton"),
        (r"$\|u_h-u\|_{L^2}$", r"$\|p_h-p\|_{L^2}$", r"$\max |[p_h]|$ (sampled)"),
        strict=True,
    ):
        axis.set(title=title, xlabel="Macro grid spacing 1/n", ylabel=ylabel)
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7)
    figure.suptitle("Stokes: five macro refinements, four local subdivisions, linear trace")
    save(figure, "convergence")


def main() -> None:
    """Plot the recorded measurements with explicit finite element conventions."""
    data = json.loads(read_resource_text(Path("examples/results/flow-audit.json")))
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    convergence(data)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, label in LABELS.items():
        rows = [r for r in data["local_refinement"] if r["formulation"] == method]
        axes[0].loglog(
            [1 / r["local_refinement"] for r in rows],
            [r["pressure_l2"] for r in rows],
            "o-",
            color=COLORS[method],
            label=label,
        )
        rows = [r for r in data["trace_enrichment"] if r["formulation"] == method]
        axes[1].semilogy(
            range(len(rows)),
            [r.get("pressure_l2", np.nan) for r in rows],
            "o-",
            color=COLORS[method],
            label=label,
        )
    axes[0].set(
        title="Local refinement, fixed linear trace",
        xlabel="1 / local subdivisions",
        ylabel=r"$\|p_h-p\|_{L^2}$",
    )
    axes[1].set(
        title="Trace enrichment, fixed local subdivisions = 8",
        ylabel=r"$\|p_h-p\|_{L^2}$",
        xticks=range(6),
        xticklabels=["P0×1", "P1×1", "P2×1", "P0×2", "P1×2", "P0×4"],
    )
    axes[1].set_xlabel("Degree × number of subfaces (separate configurations)")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    figure.suptitle("Stokes: local refinement reaches a limit set by the trace space (n = 4)")
    save(figure, "trace-local")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True)
    x = np.linspace(0, 1, 301)
    exact = 150 * (x - 0.5) * (0.37 - 0.5)
    for column, (key, title) in enumerate(
        (
            ("H_refinement", "n = 4, local = 4, trace P1"),
            ("refined_profiles", "n = 8, local = 8, trace P2"),
        )
    ):
        resolutions = {row["n"] for row in data[key] if key != "H_refinement" or row["n"] == 4}
        if len(resolutions) != 1:
            raise ValueError("Each profile panel must have one recorded macro resolution")
        macro_mesh = TriangleMesh.unit_square(resolutions.pop())
        axes[0, column].plot(x, exact, "k-", linewidth=1.8, label="Exact")
        axes[1, column].axhline(0, color="black", linewidth=1)
        for method, label in LABELS.items():
            rows = [
                r
                for r in data[key]
                if r["formulation"] == method and (key != "H_refinement" or r["n"] == 4)
            ]
            for row in rows:
                for i, segment in enumerate(row["profile"]):
                    xx, pp = np.array(segment["x"]), np.array(segment["p"])
                    for axis, yy in (
                        (axes[0, column], pp),
                        (axes[1, column], pp - 150 * (xx - 0.5) * (0.37 - 0.5)),
                    ):
                        axis.plot(
                            xx,
                            yy,
                            "--",
                            color=COLORS[method],
                            linewidth=1.2,
                            label=label if i == 0 else None,
                        )
                        axis.plot(
                            xx[[0, -1]],
                            yy[[0, -1]],
                            "o",
                            color=COLORS[method],
                            markersize=2,
                            markerfacecolor="white",
                            markeredgewidth=0.6,
                        )
                axes[0, column].text(
                    0.02,
                    0.08 if method == "usfem" else 0.16,
                    f"{label}: L²(p) = {row['pressure_l2']:.3g}",
                    transform=axes[0, column].transAxes,
                    color=COLORS[method],
                    fontsize=8,
                )
        axes[0, column].set(title=title, ylabel="p(x, 0.37)")
        axes[1, column].set(xlabel="x", ylabel="Pressure error on profile")
        crossings = macro_profile_breaks(macro_mesh, np.array([0.0, 0.37]), np.array([1.0, 0.37]))
        for axis in axes[:, column]:
            mark_macro_interfaces(axis, crossings[1:-1], label=True)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    axes[0, 0].set_ylim(-11, 11)
    axes[0, 1].set_ylim(-11, 11)
    figure.suptitle(
        "Stokes pressure: actual local polynomials and both interface limits\n"
        "Open markers denote segment endpoints; no averaging across macrofaces"
    )
    save(figure, "pressure-profiles")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_flow_audit").main()
