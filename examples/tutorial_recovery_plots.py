"""Publication figures for measured recovery and adaptive-study records.

These functions display supplied arrays and integrated diagnostics. They do
not select PDE data, manufacture numerical fields or execute solver algorithms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from examples.plot_style import set_refinement_ticks


def plot_recovery_fields(samples: dict[str, Any], *, reconstruction_name: str = "RT2") -> Any:
    """Display one-sided physical fields and errors with independent colorbars."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.8), layout="constrained")
    names = (
        ("analytical_pressure", "Analytical pressure", "coolwarm"),
        ("pressure", "MHM pressure", "coolwarm"),
        ("pressure_error", "Signed pressure error", "coolwarm"),
        ("analytical_flux_magnitude", "Analytical\nDarcy flux magnitude", "viridis"),
        (
            "recovered_flux_magnitude",
            f"Recovered {reconstruction_name}\nDarcy flux magnitude",
            "viridis",
        ),
        ("recovered_flux_error", "Recovered\nvector-flux error", "magma"),
    )
    macro_points = np.asarray(samples["macro_points"])
    macro_faces = np.asarray(samples["macro_faces"])
    for axis, (name, title, cmap) in zip(axes.flat, names, strict=True):
        maximum = max(float(np.max(abs(np.asarray(panel[name])))) for panel in samples["panels"])
        minimum = -maximum if "pressure" in name else 0
        artist = None
        for panel in samples["panels"]:
            points, cells = np.asarray(panel["points"]), np.asarray(panel["cells"])
            artist = axis.tripcolor(
                *points.T,
                cells,
                facecolors=np.asarray(panel[name]),
                vmin=minimum,
                vmax=maximum,
                cmap=cmap,
            )
        for face in macro_faces:
            axis.plot(*macro_points[face].T, color="black", lw=0.28, alpha=0.65)
        axis.set(xlabel="x", ylabel="y", title=title, aspect="equal")
        axis.set_xticks(np.linspace(macro_points[:, 0].min(), macro_points[:, 0].max(), 3))
        axis.set_yticks(np.linspace(macro_points[:, 1].min(), macro_points[:, 1].max(), 3))
        figure.colorbar(artist, ax=axis, shrink=0.83, pad=0.03)
    return figure


def plot_classical_fields(samples: dict[str, Any]) -> Any:
    """Display independently assembled conforming reference fields and errors."""
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.7), layout="constrained")
    points, cells = np.asarray(samples["points"]), np.asarray(samples["cells"])
    macro_points = np.asarray(samples["macro_points"])
    macro_faces = np.asarray(samples["macro_faces"])
    names = (
        ("pressure", "Fine classical P3\npressure", "coolwarm"),
        ("pressure_error", "Classical signed\npressure error", "coolwarm"),
        ("flux_magnitude", "Classical\nDarcy flux magnitude", "viridis"),
        ("flux_error", "Classical\nvector-flux error", "magma"),
    )
    for axis, (name, title, cmap) in zip(axes, names, strict=True):
        values = np.asarray(samples[name])
        limit = float(np.max(abs(values)))
        artist = axis.tripcolor(
            *points.T,
            cells,
            facecolors=values,
            cmap=cmap,
            vmin=-limit if "pressure" in name else 0,
            vmax=limit,
        )
        for face in macro_faces:
            axis.plot(*macro_points[face].T, color="black", lw=0.25, alpha=0.55)
        axis.set(xlabel="x", ylabel="y", title=title, aspect="equal")
        axis.set_xticks(np.linspace(macro_points[:, 0].min(), macro_points[:, 0].max(), 3))
        axis.set_yticks(np.linspace(macro_points[:, 1].min(), macro_points[:, 1].max(), 3))
        figure.colorbar(artist, ax=axis, shrink=0.82, pad=0.03)
    return figure


def plot_estimator_study(rows: Sequence[dict[str, Any]]) -> Any:
    """Compare estimator convergence, its terms, and measured effectivity."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ell, column in ((0, 0), (1, 1)):
        selected = [row for row in rows if row["trace_degree"] == ell]
        h = np.asarray([row["macro_diameter"] for row in selected])
        for name, label in (("energy_error", "Broken energy error"), ("estimator", "Estimator")):
            axes[column, 0].loglog(h, [row[name] for row in selected], "o-", label=label)
        guide = selected[-1]["energy_error"] * (h / h[-1]) ** (ell + 1)
        axes[column, 0].loglog(h, guide, "k--", label=f"Order {ell + 1}")
        for name, label in (
            ("eta_1", "Flux defect"),
            ("eta_2", "Nonconformity"),
            ("eta_3", "Divergence defect"),
            ("eta_osc", "Source oscillation"),
        ):
            axes[column, 1].loglog(h, [row[name] for row in selected], "o-", label=label)
        axes[column, 2].semilogx(h, [row["effectivity"] for row in selected], "o-")
        axes[column, 2].axhline(1, color="black", linestyle=":", label="Reliability threshold")
        for j, axis in enumerate(axes[column]):
            axis.invert_xaxis()
            set_refinement_ticks(axis, h, [f"{value:.3g}" for value in h], max_labels=3)
            axis.set_xlabel("Macro diameter H")
            axis.grid(True, which="major", alpha=0.25)
            axis.set_title(
                ("Error and estimator", "Estimator contributions", "Effectivity")[j]
                + f"\nTrace degree {ell}"
            )
            axis.set_ylabel("Integrated energy quantity" if j < 2 else "Estimator / energy error")
            axis.legend(fontsize=8)
    return figure


def plot_adaptive_study(
    adaptive: Sequence[dict[str, Any]],
    uniform: Sequence[dict[str, Any]],
    mesh: Any,
    indicators: np.ndarray,
) -> Any:
    """Compare measured error/work and display the final conforming adaptive mesh."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    for rows, label, style in (
        (adaptive, "Adaptive energy error", "o-"),
        (uniform, "Uniform energy error", "s--"),
    ):
        work = [row["global_unknowns"] for row in rows]
        axes[0].loglog(work, [row["energy_error"] for row in rows], style, label=label)
    work = np.asarray([row["global_unknowns"] for row in adaptive])
    axes[0].loglog(work, [row["estimator"] for row in adaptive], "^-", label="Adaptive estimator")
    axes[0].set(
        xlabel="Global trace and\nretained unknowns",
        ylabel="Integrated energy quantity",
        title="Localized smooth solution\nError and work",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.2)
    axes[1].plot(np.arange(len(adaptive)), [row["effectivity"] for row in adaptive], "o-")
    axes[1].axhline(1, color="black", linestyle=":")
    axes[1].set(
        xlabel="Adaptive cycle",
        ylabel="Estimator / energy error",
        title="Reliability and effectivity",
    )
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axes[1].grid(True, alpha=0.2)
    artist = axes[2].tripcolor(
        *mesh.points.T,
        mesh.cells,
        facecolors=indicators,
        edgecolors="black",
        linewidth=0.25,
        cmap="viridis",
    )
    axes[2].set(
        xlabel="x",
        ylabel="y",
        title=f"Final adaptive macro mesh\n{len(mesh.cells)} cells",
        aspect="equal",
    )
    figure.colorbar(artist, ax=axes[2], shrink=0.83, label="Squared local indicator")
    return figure
