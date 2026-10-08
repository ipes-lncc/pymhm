"""Replay material-fitted MHM/PGMHM fields and independent inclusion references."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

from examples.compare_pgmhm_inclusions import InclusionMHMField
from examples.field_sampling import sample_field
from examples.mh_campaign import l_mesh
from examples.pgmhm_inclusion_data import material_array
from examples.plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from examples.plot_style import set_refinement_ticks
from examples.solve_pgmhm_inclusions_reference import load_field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/pgmhm-inclusions"
FIGURES = ROOT / "docs/figures/pgmhm-inclusions"


def read(path: Path) -> dict:
    """Check the current coefficient archive before using its numerical record."""
    row = json.loads(path.read_text())
    for key in ("archive", "reference"):
        if (
            key in row
            and key + "_sha256" in row
            and hashlib.sha256((path.parent / row[key]).read_bytes()).hexdigest()
            != row[key + "_sha256"]
        ):
            raise ValueError(f"coefficient checksum mismatch in {path.name}")
    return row


def save(figure: plt.Figure, name: str) -> None:
    """Write raster fields and vector annotations at the publication display size."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def profile(field: InclusionMHMField) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Keep independent P2 profile pieces from both sides of every fine and macro edge."""
    coordinates, values = [], []
    for macro, triangles in enumerate(field.vertices):
        distance = triangles[..., 1] - triangles[..., 0]
        candidates = np.flatnonzero(
            (distance.min(axis=1) <= 1e-14) & (distance.max(axis=1) >= -1e-14)
        )
        for cell in candidates:
            vertices = triangles[cell]
            parameters = []
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
                da, db = a[1] - a[0], b[1] - b[0]
                if abs(da) < 1e-14:
                    parameters.append(a[0])
                if da * db < 0:
                    parameters.append(a[0] + da / (da - db) * (b[0] - a[0]))
            if len(parameters) < 2 or max(parameters) - min(parameters) < 1e-14:
                continue
            x = np.linspace(min(parameters), max(parameters), 7)
            points = np.column_stack((x, x))
            coordinates.append(x)
            values.append(field.evaluate(macro, int(cell), points)[0].astype(float))
    return coordinates, values


def main() -> None:
    """Render the final fields and retain the reference's separate refinement uncertainty."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    rows = [read(DATA / f"mhm-pgmhm-factor{factor}-s2.json") for factor in (1, 2, 4)]
    references = [read(DATA / f"classical-cg2-factor{factor}.json") for factor in (1, 2, 4)]
    available = sorted(
        (read(path) for path in DATA.glob("classical-cg2-graded*.json")),
        key=lambda row: row["grading_level"],
    )
    comparisons = []
    graded = []
    for candidate in available:
        level = candidate["grading_level"]
        paths = [
            DATA / f"mhm-pgmhm-factor{factor}-s2-graded{level}-comparison.json"
            for factor in (1, 2, 4)
        ]
        if not all(path.exists() for path in paths):
            continue
        records = [read(path) for path in paths]
        if all(len(record["rows"]) == 2 for record in records):
            comparisons = records
            graded = [row for row in available if row["grading_level"] <= level]
    if len(graded) < 2:
        raise ValueError("a refined graded reference and three complete comparisons are required")
    for row in comparisons:
        if (
            row["reference_sha256"] != graded[-1]["archive_sha256"]
            or len(row["rows"]) != 2
            or row.get("source_changed_during_run", False)
        ):
            raise ValueError("plots require the same graded reference and both norm rules")
    fine = load_field(DATA / graded[-1]["archive"])
    field = InclusionMHMField(DATA / rows[-1]["archive"])
    macro = l_mesh(4)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    image = axes[0].imshow(
        np.log10(material_array().T),
        extent=(0, 1, 0, 1),
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    draw_macro_mesh(axes[0], macro)
    axes[0].set(xlabel="x", ylabel="y", title="729 square annuli; 32 L-shaped macroelements")
    fig.colorbar(image, ax=axes[0], label="log₁₀ K", shrink=0.88)
    x = np.linspace(0, 1, 4001)
    axes[1].plot(
        x, fine.evaluate(np.column_stack((x, x)))[0], color="black", lw=1.8, label="DOLFINx/UFL CG2"
    )
    parameters, values = profile(field)
    for index, name in enumerate(("MHM", "PGMHM", "PGMHM enriched")):
        segments = [np.column_stack((p, v[index])) for p, v in zip(parameters, values, strict=True)]
        axes[1].add_collection(
            LineCollection(segments, colors=f"C{index}", linewidths=1, label=name)
        )
    mark_macro_interfaces(axes[1], macro_profile_breaks(macro, [0, 0], [1, 1]))
    axes[1].set(
        xlabel="x = y",
        ylabel="Pressure",
        title="Independent profile pieces; local factor 4",
        xlim=(0, 1),
    )
    axes[1].autoscale_view(scalex=False)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    save(fig, "material-profile")

    sampled = sample_field(
        field.meshes, tuple(value[1].astype(float) for value in field.coefficients), 2, 2
    )
    reference = fine.evaluate(sampled["points"])[0].astype(float)
    pressure = sampled["values"]
    triangulation = mtri.Triangulation(*sampled["points"].T, triangles=sampled["cells"])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), layout="constrained")
    common = min(reference.min(), pressure.min()), max(reference.max(), pressure.max())
    error = pressure - reference
    for i, (axis, values, title) in enumerate(
        zip(
            axes,
            (reference, pressure, error),
            ("DOLFINx/UFL CG2", "PGMHM P2/P0; local factor 4", "Signed pressure difference"),
            strict=True,
        )
    ):
        options = (
            dict(cmap="viridis", vmin=common[0], vmax=common[1])
            if i < 2
            else dict(cmap="RdBu_r", vmin=-abs(error).max(), vmax=abs(error).max())
        )
        image = axis.tripcolor(triangulation, values, shading="gouraud", rasterized=True, **options)
        draw_macro_mesh(axis, macro)
        axis.set(
            xlabel="x",
            ylabel="y" if i == 0 else "",
            title=title,
            aspect="equal",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        fig.colorbar(image, ax=axis, label="Pressure" if i < 2 else "p_PGMHM − p_CG2", shrink=0.85)
    save(fig, "pressure-fields")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), layout="constrained")
    for index, (name, label) in enumerate(
        (
            ("pressure", "Pressure L²"),
            ("flux", "Raw flux L²"),
            ("energy", "Weighted flux"),
        )
    ):
        axes[0].loglog(
            [row["dofs"] for row in references[1:]],
            [row["norms"][-1][name + "_relative"] for row in references[1:]],
            "o-",
            color=f"C{index}",
            label=label + "; uniform",
        )
        axes[0].loglog(
            [row["dofs"] for row in graded[1:]],
            [row["norms"][-1][name + "_relative"] for row in graded[1:]],
            "^-",
            color=f"C{index}",
            label=label + "; graded",
        )
    axes[0].set(
        title="Classical CG2 successive refinements",
        xlabel="Fine reference unknowns",
        ylabel="Relative successive increment",
    )
    set_refinement_ticks(
        axes[0],
        [references[1]["dofs"], references[2]["dofs"], *(row["dofs"] for row in graded[1:])],
        ["0.191M", "0.762M", *(f"{row['dofs'] / 1e6:.2f}M" for row in graded[1:])],
    )
    for axis, name in zip(axes[1:], ("pressure", "flux"), strict=True):
        for variant, label in (("mhm", "MHM"), ("pgmhm", "PGMHM"), ("enriched", "PGMHM enriched")):
            axis.loglog(
                [1, 2, 4],
                [row["rows"][-1][variant][name + "_relative"] for row in comparisons],
                "o-",
                label=label,
            )
        axis.set(
            title=f"{name.capitalize()} L² difference",
            xlabel="Local subdivision factor",
            ylabel="Relative difference from CG2",
        )
        set_refinement_ticks(axis, [1, 2, 4])
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7.5)
    save(fig, "physical-norms")
    for path in DATA.glob("*.json"):
        shutil.copy2(path, FIGURES / path.name)


if __name__ == "__main__":
    main()
