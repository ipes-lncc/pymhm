"""Visualize adaptive SPE10 fields and independently refined RT2 reference records."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.ticker import MaxNLocator
from threadpoolctl import threadpool_limits

from examples.plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from examples.plot_style import set_refinement_ticks
from examples.spe10_adaptive import DATA, DOMAIN, ROOT, StructuredRT
from examples.spe10_adaptive_norms import BrokenP2
from pymhm.fem.hdiv.rt_forms import pressure_basis
from pymhm.fem.quadrature.material import fit_material_mesh
from pymhm.meshes.triangle import TriangleMesh

FIGURES = ROOT / "docs/figures/spe10-adaptive"


def save(figure, name: str, directory: Path = FIGURES) -> None:
    """Export readable vector text and rasterized dense maps in both formats."""
    for extension in ("png", "svg"):
        figure.savefig(
            directory / f"{name}.{extension}", dpi=180, bbox_inches="tight", pad_inches=0.12
        )
    plt.close(figure)


def display_fields(mhm: BrokenP2, *, material_fitted: bool = False) -> dict[str, np.ndarray]:
    """Sample fine/material pieces; fitted archives supply their exact incident-cell owners.

    Set ``material_fitted`` only for a verified fitted approximation mesh. This
    avoids constructing the same partition again or locating its known centroids.
    """
    points, cells, values = [], [], []
    offset = 0
    for cell, fine in enumerate(mhm.meshes):
        display = fine if material_fitted else fit_material_mesh(fine, mhm.material)
        physical = display.points[display.cells].mean(axis=1)
        owners = np.arange(len(fine.cells)) if material_fitted else None
        p, raw, reconstructed = mhm.evaluate_local(cell, physical, owners=owners)
        points.append(display.points)
        cells.append(display.cells + offset)
        values.append(np.column_stack((p, raw, reconstructed)))
        offset += len(display.points)
    return {
        "points": np.concatenate(points),
        "cells": np.concatenate(cells),
        "values": np.concatenate(values),
    }


def reference_display(reference: StructuredRT) -> dict[str, np.ndarray]:
    """Use one physical centroid sample per reference triangle without projection."""
    values = []
    vertices = reference.mesh.points[reference.mesh.cells]
    for first in range(0, len(vertices), 4096):
        p, q, _ = reference.evaluate(vertices[first : first + 4096].mean(axis=1))
        values.append(np.column_stack((p, q)))
    return {
        "points": reference.mesh.points,
        "cells": reference.mesh.cells,
        "values": np.concatenate(values),
    }


def reference_pressure_profile(reference: StructuredRT) -> tuple[np.ndarray, np.ndarray]:
    """Keep separate DG2 limits on every triangle intersected by the domain diagonal."""
    counts = (reference.nx, reference.ny, abs(reference.ny - reference.nx))
    breaks = np.unique(np.concatenate([np.linspace(0, 1, n + 1) for n in counts if n]))
    parameters, values = [], []
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        midpoint = (left + right) / 2 * np.array([reference.nx, reference.ny])
        ij = np.minimum(np.floor(midpoint).astype(int), [reference.nx - 1, reference.ny - 1])
        upper = (midpoint - ij)[1] > (midpoint - ij)[0]
        owner = 2 * (ij[1] * reference.nx + ij[0]) + int(upper)
        parameter = np.linspace(left, right, 5)
        local = parameter[:, None] * np.array([reference.nx, reference.ny]) - ij
        x, y = local.T
        bary = np.column_stack((1 - y, x, y - x) if upper else (1 - x, x - y, y))
        pressure = pressure_basis(2, bary) @ reference.pressure[owner]
        parameters.append(np.r_[parameter, np.nan])
        values.append(np.r_[pressure, np.nan])
    return np.concatenate(parameters), np.concatenate(values)


def fine_mesh_quality(path: Path) -> tuple[float, float]:
    """Measure local-cell angles and diameter/area ratios from the archived coordinates."""
    with np.load(path) as arrays:
        if "point_offsets" in arrays:
            points, cells = arrays["local_points"], arrays["local_cells"]
            po, co = arrays["point_offsets"], arrays["cell_offsets"]
            vertices = np.concatenate(
                [points[po[k] : po[k + 1]][cells[co[k] : co[k + 1]]] for k in range(len(po) - 1)]
            )
        else:
            vertices = np.concatenate(
                [p[c] for p, c in zip(arrays["local_points"], arrays["local_cells"], strict=True)]
            )
    angles, lengths = [], []
    for index in range(3):
        a = vertices[:, (index + 1) % 3] - vertices[:, index]
        b = vertices[:, (index + 2) % 3] - vertices[:, index]
        la, lb = np.sum(a * a, axis=1), np.sum(b * b, axis=1)
        lengths.append(la)
        cosine = np.sum(a * b, axis=1) / np.sqrt(la * lb)
        angles.append(np.arccos(np.clip(cosine, -1, 1)))
    a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    areas = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    return float(np.min(angles) * 180 / np.pi), float(np.max(np.max(lengths, axis=0) / areas))


def panel(axis, field, values, macro, title, bounds, *, signed=False, asinh=False):
    """Reserve a horizontal scale below each field and display real macro boundaries."""
    normalization = (
        matplotlib.colors.AsinhNorm(
            linear_width=1e-6,
            vmin=bounds[0],
            vmax=bounds[1],
        )
        if asinh
        else matplotlib.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
    )
    artist = axis.tripcolor(
        *field["points"].T,
        field["cells"],
        facecolors=values,
        cmap="RdBu_r" if signed else "viridis",
        norm=normalization,
        rasterized=True,
    )
    macro_artist = draw_macro_mesh(axis, macro)
    macro_artist.set_linewidth(0.1)
    macro_artist.set_alpha(0.32)
    macro_artist.set_path_effects(
        [patheffects.Stroke(linewidth=0.2, foreground="white", alpha=0.15), patheffects.Normal()]
    )
    macro_artist.set_rasterized(True)
    axis.set(title=title, xlabel="x (ft)", ylabel="y (ft)", aspect="equal")
    axis.set_xticks([0, 600, 1200])
    axis.set_yticks([0, 1100, 2200])
    bar = axis.figure.colorbar(artist, ax=axis, orientation="horizontal", pad=0.12, fraction=0.06)
    if asinh:
        ticks = normalization.inverse(np.linspace(0, 1, 5 if signed else 4))
        ticks[2 if signed else 0] = 0.0
        bar.locator = matplotlib.ticker.FixedLocator(ticks)
        bar.formatter = matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:.2g}")
    else:
        bar.locator = MaxNLocator(4)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=13)
    return artist


def plot(data: Path = DATA / "published", output: Path = FIGURES) -> None:
    """Render mesh, estimator, reference refinement and unsmoothed physical fields."""
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 13})
    rows = json.loads((data / "adaptive.json").read_text())
    references = json.loads((DATA / "references-aligned.json").read_text())
    comparison = json.loads((data / "comparison-order4.json").read_text())
    if [row["level"] for row in comparison] != [row["level"] for row in rows]:
        raise ValueError("each adaptive state requires its own integrated comparison record")
    mhm = BrokenP2(data / rows[-1]["archive"])
    reference_names = {row["reference"] for row in comparison}
    if len(reference_names) != 1:
        raise ValueError("every state in a comparison curve must use the same physical reference")
    reference_name = reference_names.pop()
    reference_record = next(row for row in references if row["archive"] == reference_name)
    reference_path = DATA / reference_name
    digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if any(row["reference_sha256"] != digest for row in comparison):
        raise ValueError("comparison reference checksum mismatch")
    reference = StructuredRT.load(reference_path)
    numerical = display_fields(mhm, material_fitted=rows[-1].get("material_fitted", False))
    classical = reference_display(reference)
    published = rows[-1].get("estimator_convention", "energy") == "published"
    dofs = [row["global_dofs"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axes[0].loglog(
        dofs,
        [row["estimator"] for row in rows],
        "o-",
        label="Published indicator" if published else "Weighted estimator",
    )
    if not published:
        for key, label in (
            ("raw_flux_energy_difference", "Raw-flux difference"),
            ("reconstructed_flux_energy_difference", "RT2-flux difference"),
        ):
            axes[0].loglog(dofs, [row[key] for row in comparison], "o-", label=label)
    axes[0].set(
        xlabel="MHM global unknowns before BC elimination",
        ylabel="Published indicator (Eq. 5.6)" if published else "Weighted energy norms",
    )
    for key, label in (
        ("pressure_relative_difference", "Pressure"),
        ("raw_flux_relative_difference", "Raw flux"),
        ("reconstructed_flux_relative_difference", "RT2 flux"),
    ):
        axes[1].semilogx(dofs, [100 * row[key] for row in comparison], "o-", label=label)
    axes[1].set(
        xlabel="MHM global unknowns before BC elimination",
        ylabel="Difference / RT2 reference norm (%)",
    )
    labeled = np.unique(np.linspace(0, len(dofs) - 1, min(4, len(dofs)), dtype=int))
    for axis in axes:
        set_refinement_ticks(axis, np.asarray(dofs)[labeled])
        axis.grid(alpha=0.25)
        axis.legend()
    save(figure, "refinement", output)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.4), layout="constrained")
    for key, label in (
        ("flux_defect", "Flux defect"),
        ("nonconformity", "Nonconformity"),
        ("divergence_defect", "Divergence defect"),
    ):
        axes[0, 0].loglog(dofs, [row[key] for row in rows], "o-", label=label)
    axes[0, 0].set(ylabel="Estimator components")
    axes[0, 1].loglog(
        dofs, [row["local_indicator"] for row in rows], "o-", label="Nested local difference"
    )
    axes[0, 1].loglog(
        dofs,
        [row["raw_flux_energy_difference"] for row in comparison],
        "o-",
        label="Raw flux vs. RT2",
    )
    axes[0, 1].set(ylabel="Weighted energy norms")
    angles = [row["minimum_macro_angle_degrees"] for row in rows]
    fine_quality = [fine_mesh_quality(data / row["archive"]) for row in rows]
    axes[1, 0].semilogx(dofs, angles, "o-", label="Macro minimum angle")
    axes[1, 0].semilogx(
        dofs, [value[0] for value in fine_quality], "s-", label="Local minimum angle"
    )
    if not published:
        axes[1, 0].axhline(
            angles[0] / 2, linestyle="--", color="black", label="Half initial macro minimum"
        )
    axes[1, 0].set(ylabel="Triangle angle (degrees)")
    axes[1, 1].semilogx(dofs, [row["inflow"] for row in rows], "o-", label="MHM trace inflow")
    axes[1, 1].axhline(
        reference_record["inflow"], linestyle="--", color="black", label="Finest classical RT2"
    )
    axes[1, 1].set(ylabel="Integrated bottom inflow")
    for axis in axes.flat:
        axis.set_xlabel("MHM global unknowns before BC elimination")
        set_refinement_ticks(axis, np.asarray(dofs)[labeled])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=11)
    axes[0, 1].legend(loc="lower left", fontsize=11)
    save(figure, "resolution-diagnostics", output)
    for page, first in enumerate(range(0, len(rows), 4)):
        figure, axes = plt.subplots(2, 2, figsize=(8.5, 10.8), layout="constrained")
        for level, axis in zip(range(first, first + 4), axes.flat, strict=True):
            if level >= len(rows):
                axis.axis("off")
                continue
            arrays = np.load(data / rows[level]["archive"])
            mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
            if not published:
                axis.tripcolor(
                    *mesh.points.T,
                    mesh.cells,
                    facecolors=arrays["marked"].astype(float),
                    cmap="Blues",
                    rasterized=True,
                )
            draw_macro_mesh(axis, mesh).set_rasterized(True)
            axis.set(
                title=f"Level {level}: {len(mesh.cells)} cells",
                xlabel="x (ft)",
                ylabel="y (ft)",
                aspect="equal",
            )
            axis.set_xticks([0, 1200])
            axis.set_yticks([0, 1100, 2200])
        save(figure, "meshes" if page == 0 else f"meshes-{page + 1}", output)
    pbound = (
        min(numerical["values"][:, 0].min(), classical["values"][:, 0].min()),
        max(numerical["values"][:, 0].max(), classical["values"][:, 0].max()),
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 6.2), layout="constrained")
    panel(axes[0], numerical, numerical["values"][:, 0], mhm.macro, "Adaptive MHM pressure", pbound)
    panel(
        axes[1], classical, classical["values"][:, 0], mhm.macro, "Classical RT2 pressure", pbound
    )
    save(figure, "pressure", output)
    magnitudes = (
        np.linalg.norm(numerical["values"][:, 1:3], axis=1),
        np.linalg.norm(numerical["values"][:, 3:5], axis=1),
        np.linalg.norm(classical["values"][:, 1:3], axis=1),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 6.3), layout="constrained")
    magnitude_bound = (0.0, max(float(np.max(value)) for value in magnitudes))
    for axis, field, value, name in zip(
        axes,
        (numerical, numerical, classical),
        magnitudes,
        ("MHM raw", "MHM reconstructed RT2", "Classical RT2"),
        strict=True,
    ):
        panel(
            axis,
            field,
            value,
            mhm.macro,
            f"{name}\nFlux magnitude (mD/ft)",
            magnitude_bound,
            asinh=True,
        )
    save(figure, "flux-magnitude", output)
    figure, axes = plt.subplots(2, 3, figsize=(11, 11), layout="constrained")
    for component in (0, 1):
        fields = (numerical, numerical, classical)
        vectors = (
            numerical["values"][:, 1 + component],
            numerical["values"][:, 3 + component],
            classical["values"][:, 1 + component],
        )
        limit = max(np.max(abs(value)) for value in vectors)
        for axis, field, value, name in zip(
            axes[component],
            fields,
            vectors,
            ("MHM raw", "MHM reconstructed RT2", "Classical RT2"),
            strict=True,
        ):
            panel(
                axis,
                field,
                value,
                mhm.macro,
                f"{name}\nq{'xy'[component]} (mD/ft)",
                (-limit, limit),
                signed=True,
                asinh=True,
            )
    save(figure, "flux-components", output)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    unknowns = [row["dofs"] for row in references[1:]]
    for key, label in (
        ("previous_pressure_relative_difference", "Pressure"),
        ("previous_flux_relative_difference", "Flux"),
    ):
        axes[0].loglog(unknowns, [100 * row[key] for row in references[1:]], "o-", label=label)
    axes[0].set(xlabel="Classical RT2 unknowns", ylabel="Successive difference / fine norm (%)")
    axes[1].semilogx(
        [row["dofs"] for row in references],
        [row["inflow"] for row in references],
        "o-",
        label="Classical RT2",
    )
    axes[1].axhline(rows[-1]["inflow"], linestyle="--", label="Final MHM trace inflow")
    axes[1].set(xlabel="Classical RT2 unknowns", ylabel="Integrated bottom inflow")
    for axis, ticks in zip(axes, (unknowns, [row["dofs"] for row in references]), strict=True):
        set_refinement_ticks(axis, ticks, [f"{x / 1e6:.2g}M" for x in ticks])
        axis.grid(alpha=0.25)
        axis.legend()
    save(figure, "reference-refinement", output)
    profile_records = json.loads((DATA / "published-profile-mhm.json").read_text())["curves"]
    reference_samples = json.loads((DATA / "published-profile.json").read_text())["samples"]
    t, reference_pressure = reference_pressure_profile(reference)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained", sharey=True)
    for axis, record in zip(axes, profile_records, strict=True):
        field = BrokenP2(data / rows[0 if record["curve"] == "initial" else -1]["archive"])
        breaks = macro_profile_breaks(field.macro, np.zeros(2), DOMAIN)
        for i, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
            parameter = np.linspace(left, right, 101)
            points = parameter[:, None] * DOMAIN
            cell = int(field.locate(((left + right) / 2 * DOMAIN)[None])[0])
            pressure, _, _ = field.evaluate_local(cell, points)
            axis.plot(
                parameter,
                pressure,
                color="C0",
                label="PyMHM raw pressure" if i == 0 else None,
            )
        axis.plot(t, reference_pressure, color="black", linestyle="--", label="Classical RT2")
        axis.plot(
            [sample["t"] for sample in reference_samples],
            [sample["published_pressure"] for sample in reference_samples],
            ".",
            color="C3",
            markersize=3,
            label="Published reference",
        )
        axis.plot(
            [sample["t"] for sample in record["samples"]],
            [sample["published_pressure"] for sample in record["samples"]],
            ".",
            color="C2" if record["level"] else "C4",
            markersize=3,
            label="Published MHM (digitized)",
        )
        mark_macro_interfaces(axis, breaks[1:-1])
        axis.set(
            title=f"{record['curve'].capitalize()}: {len(field.macro.cells)} macrotriangles",
            xlabel="Diagonal parameter t",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=11)
    axes[0].set_ylabel("Raw pressure")
    figure.supxlabel("Physical profile: (x, y) = t (1200, 2200) ft", fontsize=12)
    save(figure, "diagonal-profile", output)
    quadrature_record = data / "quadrature-check.json"
    if quadrature_record.is_file():
        (output / quadrature_record.name).write_bytes(quadrature_record.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "published")
    parser.add_argument("--output", type=Path, default=FIGURES)
    options = parser.parse_args()
    with threadpool_limits(1):
        plot(options.data, options.output)
