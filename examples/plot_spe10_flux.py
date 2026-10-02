"""Render archived SPE10 Darcy flux comparisons and reference refinement studies.

This program reads physical fields and measured norms; it does not execute
reference codes. Pixel-center maps are sampling visualizations, distinct from
the integrated finite-element norms recorded with each comparison.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from plot_pyvista_layout import signed_asinh_scale
from plot_spe10_data import FIGURES, OUTPUT, macro_mesh, panel

from pymhm.visualization import structured_cell_grid


def archive(record: dict) -> dict[str, np.ndarray]:
    """Read a physical-field archive only after checking its recorded digest."""
    path = OUTPUT / record["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Archive checksum differs: {path.name}")
    with np.load(path) as data:
        values = {name: data[name] for name in ("points", "pressure", "flux")}
    if values["flux"].shape != (60, 220, 2) or not np.isfinite(values["flux"]).all():
        raise ValueError("Flux archive must contain finite 60-by-220 vector samples")
    return values


def field_panel(
    plotter: pv.Plotter,
    values: np.ndarray,
    title: str,
    scalar: str,
    limits: tuple[float, float],
    *,
    signed: bool = False,
) -> None:
    """Show unaveraged pixel samples and the actual 66-square MHM partition."""
    labels = None
    if signed:
        values, labels = signed_asinh_scale(values, limits[1])
        limits = (-1.0, 1.0)
    grid = structured_cell_grid((60, 220), cell_data={scalar: values}, spacing=(20, 10))
    panel(
        plotter,
        grid,
        scalar,
        title,
        limits=limits,
        cmap="RdBu_r" if signed else "viridis",
        mesh=macro_mesh(),
        color_labels=labels,
    )


def compare(mhm: dict, reference: dict, suffix: str) -> None:
    """Plot magnitude, signed components, and vector differences at common points."""
    numerical, baseline = archive(mhm), archive(reference)
    np.testing.assert_array_equal(numerical["points"], baseline["points"])
    q, qref = numerical["flux"], baseline["flux"]
    magnitude = np.linalg.norm(q, axis=-1)
    reference_magnitude = np.linalg.norm(qref, axis=-1)
    difference = np.linalg.norm(q - qref, axis=-1)
    limit = float(max(magnitude.max(), reference_magnitude.max()))
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 1000))
    for column, (values, title, scalar, bounds) in enumerate(
        (
            (reference_magnitude, reference["label"], "Flux magnitude", (0.0, limit)),
            (magnitude, mhm["label"], "Flux magnitude", (0.0, limit)),
            (
                difference,
                "Norm of vector difference\nMHM minus reference",
                "Flux difference",
                (0.0, float(difference.max())),
            ),
        )
    ):
        plotter.subplot(0, column)
        field_panel(plotter, values, title + "\nMaterial-pixel center samples", scalar, bounds)
    plotter.screenshot(FIGURES / f"darcy-flux-{suffix}.png")
    plotter.close()

    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2400, 2000))
    for component, name in enumerate(("qx", "qy")):
        bound = float(max(abs(q[..., component]).max(), abs(qref[..., component]).max()))
        delta = q[..., component] - qref[..., component]
        for column, (values, title, extent) in enumerate(
            (
                (qref[..., component], reference["label"], bound),
                (q[..., component], mhm["label"], bound),
                (delta, "MHM minus reference", float(abs(delta).max())),
            )
        ):
            plotter.subplot(component, column)
            field_panel(
                plotter,
                values,
                title + f"\nSigned {name}; asinh color scale",
                name if column < 2 else f"Delta {name}",
                (-extent, extent),
                signed=True,
            )
    plotter.screenshot(FIGURES / f"darcy-flux-{suffix}-components.png")
    plotter.close()


def reference_maps(references: dict[str, dict]) -> None:
    """Compare classical references using one common flux-magnitude scale."""
    selected = [references[key] for key in ("q3", "msl", "neopz")]
    data = [archive(row) for row in selected]
    for values in data[1:]:
        np.testing.assert_array_equal(values["points"], data[0]["points"])
    magnitudes = [np.linalg.norm(row["flux"], axis=-1) for row in data]
    limit = float(max(values.max() for values in magnitudes))
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 1000))
    for column, (values, record) in enumerate(zip(magnitudes, selected, strict=True)):
        plotter.subplot(0, column)
        field_panel(
            plotter,
            values,
            record["label"] + "\nMaterial-pixel center samples",
            "Flux magnitude",
            (0.0, limit),
        )
    plotter.screenshot(FIGURES / "darcy-flux-classical-references.png")
    plotter.close()


def refinement(report: dict) -> None:
    """Display independently integrated changes between reference mesh levels."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for key, label, marker in (
        ("q3", "Q3", "o"),
        ("msl", "MSL CG P1", "s"),
        ("neopz", "NeoPZ RT0/P0", "^"),
    ):
        rows = report["refinement"][key]
        x = [row["cells"] for row in rows]
        axes[0].plot(
            x[1:], [100 * row["flux_relative"] for row in rows[1:]], marker=marker, label=label
        )
        axes[1].plot(
            x[1:], [100 * row["pressure_relative"] for row in rows[1:]], marker=marker, label=label
        )
        axes[2].plot(x, [row["inlet_flux"] for row in rows], marker=marker, label=label)
    for axis, title, ylabel in zip(
        axes,
        ("Vector-flux refinement change", "Pressure refinement change", "Total inflow"),
        (
            "Successive relative L2 difference [%]",
            "Successive relative L2 difference [%]",
            "Integrated inlet flux",
        ),
        strict=True,
    ):
        axis.set(title=title, xlabel="Reference mesh cells", ylabel=ylabel, xscale="log")
        if axis is not axes[2]:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[2].axhline(report["mhm"]["inlet_flux"], color="black", ls=":", label="MHM skeleton")
    axes[2].legend(fontsize=8)
    figure.suptitle(
        "SPE10 layer 36: same permeability and boundary conditions; distinct approximation spaces"
    )
    figure.savefig(FIGURES / "darcy-flux-reference-refinement.svg")
    figure.savefig(FIGURES / "darcy-flux-reference-refinement.png", dpi=200)
    plt.close(figure)


def main() -> None:
    """Replay the completed flux study without compiling or running reference codes."""
    report = json.loads((OUTPUT / "darcy-flux-comparison.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, reference in report["references"].items():
        compare(report["mhm"], reference, name)
    reference_maps(report["references"])
    refinement(report)
    (FIGURES / "darcy-flux-comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
