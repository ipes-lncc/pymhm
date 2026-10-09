"""Replay separated local/material/skeleton controls with physical raw flux components."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import SymLogNorm

from examples.plot_pgmhm_spe10 import (
    FIGURES,
    macro_overlay,
    pixel_flux,
    save,
)
from examples.solve_pgmhm_spe10 import OUTPUT, PGMHMField
from examples.spe10_adaptive import StructuredRT
from examples.spe10_plot_records import checked_comparison, comparison_reference, digest
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.io.workspace import local_resource, read_resource_text, resource_glob

DATA = OUTPUT / "kx"


def records() -> list[tuple[Path, dict, dict]]:
    """Accept only complete two-rule comparisons of the same physical reference."""
    paths = [DATA / "pgmhm-s16-q5.json"]
    paths += sorted(resource_glob(DATA / "controls", "pgmhm-*-q5.json"))
    result = []
    for path in paths:
        comparison = path.with_name(path.stem + "-comparison.json")
        if not local_resource(comparison).exists():
            continue
        row, norms = json.loads(read_resource_text(path)), checked_comparison(comparison, DATA)
        identity = digest(path.with_suffix(".npz"))
        if identity != row["archive_sha256"] or identity != norms["mhm_sha256"]:
            raise ValueError("PGMHM control and physical norms must identify the same field")
        result.append((path, row, norms))
    if result:
        comparison_reference(DATA, [record for _, _, record in result])
    return result


def centroid_flux(field: PGMHMField) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate P2 gradients in actual fine cells, retaining their one-sided material values."""
    _, derivative, _ = reference_basis(2, np.full((1, 3), 1 / 3))
    points, cells, centers, values = [], [], [], []
    offset = 0
    for i, mesh in enumerate(field.meshes):
        coordinates = mesh.points[mesh.cells].mean(axis=1)
        coefficients = field.fields["pressure"][i][field.dofs[i]]
        gradient = np.einsum("in,tna,ti->ta", derivative[0], field.geometry[i], coefficients)
        flux = -field.material(coordinates)[:, None] * gradient
        points.append(mesh.points)
        cells.append(mesh.cells + offset)
        centers.append(coordinates)
        values.append(flux)
        offset += len(mesh.points)
    return tuple(np.concatenate(items) for items in (points, cells, centers, values))


def main() -> None:
    """Render the finest accepted control and retain all completed norm comparisons."""
    data = records()
    if len(data) < 2:
        raise ValueError("at least a nominal and a completed fitted control are required")
    data[1:] = sorted(
        data[1:],
        key=lambda item: (item[1]["local_refinement"], item[1]["segments"], item[1]["global_dofs"]),
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    nominal = PGMHMField(data[0][0].with_suffix(".npz"))
    fitted = PGMHMField(data[-1][0].with_suffix(".npz"))
    reference = StructuredRT.load(comparison_reference(DATA, [record for _, _, record in data]))
    points, cells, centers, fit_flux = centroid_flux(fitted)
    fields = (reference.evaluate(centers)[1], pixel_flux(nominal, centers), fit_flux)
    tri = mtri.Triangulation(*points.T, triangles=cells)
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 8.3), layout="constrained")
    for component, name in enumerate(("qₓ", "qᵧ")):
        bound = max(float(np.max(abs(values[:, component]))) for values in fields)
        powers = 10.0 ** np.arange(np.ceil(np.log10(bound / 100)), np.floor(np.log10(bound)) + 1)
        ticks = np.r_[-powers[::-1], 0.0, powers]
        for column, values in enumerate(fields):
            image = axes[component, column].tripcolor(
                tri,
                facecolors=np.asarray(values[:, component], dtype=float),
                cmap="RdBu_r",
                norm=SymLogNorm(bound / 100, vmin=-bound, vmax=bound),
                rasterized=True,
            )
            fig.colorbar(
                image,
                ax=axes[component, column],
                label=f"{'Flux' if column == 0 else 'Raw flux'} {name} (symmetric log)",
                shrink=0.8,
                ticks=ticks,
            )
            macro_overlay(axes[component, column])
            if column:
                axes[component, column].set_ylabel("")
    row = data[-1][1]
    titles = (
        "Classical RT2/P2 reference",
        "Unfitted P2, r32 / s16",
        f"Material-fitted P2, r{row['local_refinement']} / s{row['segments']}"
        + ("\nMaterial-fitted P0 trace" if row.get("material_fitted_trace") else ""),
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    save(fig, FIGURES, "control-flux-components")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    labels = ["Unfitted\nr32 / s16"] + [
        f"Fitted\nr{row['local_refinement']} / s{row['segments']}"
        + ("*" if row.get("material_fitted_trace") else "")
        for _, row, _ in data[1:]
    ]
    for axis, key, title in zip(
        axes,
        ("pressure_l2", "flux_l2", "flux_energy"),
        ("Pressure L²", "Raw flux L²", "Weighted raw flux"),
        strict=True,
    ):
        values = [item[2]["norms"][-1][f"base_{key}_relative_difference"] for item in data]
        axis.plot(np.arange(len(data)), values, "o-")
        axis.set(xticks=np.arange(len(data)), xticklabels=labels, title=title, yscale="log")
        axis.set_ylabel("Relative difference from RT2")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(alpha=0.2)
    fig.supxlabel("* Trace partitions include material-interface intersections", fontsize=9)
    save(fig, FIGURES, "resolution-controls")
    for path, _, _ in data[1:]:
        for report in (path, path.with_name(path.stem + "-comparison.json")):
            shutil.copy2(report, FIGURES / report.name)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_pgmhm_spe10_controls").main()
