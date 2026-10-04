"""Render periodic Darcy fields and separate reference, local and trace sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from . import compare_periodic
    from .compare_periodic import load_fields, load_reference
    from .plot_mesh import draw_macro_mesh
    from .plot_style import set_refinement_ticks
    from .verify_periodic import ROOT, material, source
else:
    import compare_periodic
    from compare_periodic import load_fields, load_reference
    from plot_mesh import draw_macro_mesh
    from plot_style import set_refinement_ticks
    from verify_periodic import ROOT, material, source

from pymhm.meshes.cartesian import CartesianMacroMesh

FOLDER = ROOT / "docs/figures/periodic"
COMPARISON = ROOT / "examples/results/periodic-comparison.json"
RECORDS = ROOT / "examples/results"


def save(figure: plt.Figure, name: str) -> None:
    """Save identical publication content as a raster preview and vector figure."""
    FOLDER.mkdir(exist_ok=True, parents=True)
    for extension in ("svg", "png"):
        figure.savefig(FOLDER / f"{name}.{extension}", dpi=190)
    plt.close(figure)


def reference_changes(records: list[dict], comparisons: list[dict], reference: str) -> list[dict]:
    """Collect consecutive reference differences across algebraically equivalent assemblers."""
    specification = reference.split(":")
    selected_degree, selected_n = map(int, specification[:2])
    selected_order = (
        int(specification[2]) if len(specification) >= 3 else max(4, selected_degree + 1)
    )
    curves: dict[tuple[int, int], dict[int, dict]] = {}
    # Separated and elementwise assembly solve the same checked tensor-Gauss
    # operator. Prefer separated direct controls when duplicate levels exist.
    ordered = sorted(
        records,
        key=lambda item: any(row.get("assembly") == "separable" for row in item["reference"]),
    )
    for record in ordered:
        for row in record["reference"]:
            if "difference_to_previous" not in row:
                continue
            degree = row.get("degree", 1)
            order = row.get("quadrature_order", max(4, degree + 1))
            if degree == selected_degree and order != selected_order:
                continue
            curves.setdefault((degree, order), {})[row["n"]] = row["difference_to_previous"]
    preceding: dict[tuple[int, int, int], tuple[int, dict]] = {}
    for row in comparisons:
        if not row["field"].startswith("conforming:"):
            continue
        reference_parts = row["reference"].split(":")
        reference_degree, reference_n = map(int, reference_parts[:2])
        reference_order = (
            int(reference_parts[2]) if len(reference_parts) >= 3 else max(4, reference_degree + 1)
        )
        parts = row["field"].split(":")[1:]
        degree, n = map(int, parts[:2])
        order = int(parts[2]) if len(parts) >= 3 else max(4, degree + 1)
        if (
            degree == reference_degree == selected_degree
            and order == reference_order == selected_order
            and n < reference_n <= selected_n
        ):
            key = (degree, order, reference_n)
            if key not in preceding or n > preceding[key][0]:
                preceding[key] = n, row
    for (degree, order, n), (_, norms) in preceding.items():
        curves.setdefault((degree, order), {})[n] = norms
    return [
        dict(degree=degree, order=order, points=sorted(rows.items()))
        for (degree, order), rows in sorted(curves.items())
    ]


def convergence(reference: str, *, macro: int = 8, refinement: int | None = None) -> int:
    """Separate historical markers from measured baseline and local-grid uncertainty."""
    records = [json.loads(path.read_text()) for path in sorted(RECORDS.glob("periodic*.json"))]
    records = [item for item in records if "mhm" in item and "reference" in item]
    comparisons = json.loads(COMPARISON.read_text())["comparisons"]
    selected = [
        row
        for row in comparisons
        if row["reference"] == reference and "segments" in row and row["macro"] == macro
    ]
    if not selected:
        raise ValueError("selected reference has no recorded MHM norm comparisons")
    finest = max(row["refinement"] for row in selected) if refinement is None else refinement
    if not any(row["refinement"] == finest for row in selected):
        raise ValueError("selected local refinement has no recorded MHM norm comparisons")
    with (ROOT / "examples/results/published/paredes2017_figure6.csv").open() as stream:
        published = [row for row in csv.DictReader(stream) if row["series"] == "face"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    ax = axes[0]
    h = np.array([float(row["effective_h"]) for row in published])
    values = np.array([float(row["relative_h1"]) for row in published])
    limits = np.array([[float(row[key]) for row in published] for key in ("lower", "upper")])
    if macro == 8:
        ax.errorbar(
            h,
            values,
            yerr=np.abs(limits - values),
            color="black",
            marker="x",
            capsize=3,
            ls="--",
            label="Published Figure 6 (digitized; 8×8 macros)",
        )
    rows = sorted(
        [row for row in selected if row["refinement"] == finest], key=lambda r: r["segments"]
    )
    ax.plot(
        [1 / (macro * r["segments"]) for r in rows],
        [r["relative_h1"] for r in rows],
        "s-",
        label=f"MHM / conforming Q{reference.split(':')[0]} {reference.split(':')[1]}²",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Face segment length h",
        ylabel="Relative broken H1 difference",
        title=f"Face enrichment; {macro}×{macro} macros; local r={finest}",
    )
    set_refinement_ticks(ax, h, [f"1/{8 * int(r['subdivisions'])}" for r in published])
    ax.legend(fontsize=8, loc="best")
    ax = axes[1]
    for segments in (1, 8, 32):
        rows = sorted(
            [r for r in selected if r["segments"] == segments], key=lambda r: r["refinement"]
        )
        if rows:
            ax.loglog(
                [r["refinement"] for r in rows],
                [r["relative_h1"] for r in rows],
                "o-",
                label=f"{segments} segment(s) per face",
            )
    levels = sorted({row["refinement"] for row in selected})
    ax.set(
        xlabel="Q1 subdivisions per macro edge",
        ylabel="Relative broken H1 difference",
        title="Local mesh sensitivity; fixed skeleton",
    )
    set_refinement_ticks(ax, levels or [128, 256, 512])
    ax.legend(fontsize=8)
    ax = axes[2]
    for curve in reference_changes(records, comparisons, reference):
        ax.loglog(
            [n for n, _ in curve["points"]],
            [norms["relative_h1"] for _, norms in curve["points"]],
            "s-",
            label=f"Conforming Q{curve['degree']}; quadrature {curve['order']}",
        )
    ax.set(
        xlabel="Reference cells per coordinate",
        ylabel="Relative H1 change to preceding mesh",
        title="Classical reference refinement",
    )
    ax.legend(fontsize=8)
    set_refinement_ticks(ax, [128, 256, 512, 1024, 2048, 4096])
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, "convergence")
    return finest


def field_maps(
    reference_specification: str, *, macro: int = 8, refinement: int = 512, samples: int = 512
) -> None:
    """Sample each broken physical field directly with shared signed component scales."""
    mesh = CartesianMacroMesh(macro)
    reference, _ = load_reference(reference_specification)
    local = [load_fields(macro, refinement, segments)[0] for segments in (1, 32)]
    n = samples
    coordinates = (np.arange(n) + 0.5) / n
    points = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
    owners = (points[:, 0] * macro).astype(int) + macro * (points[:, 1] * macro).astype(int)
    pressure, gradient = reference.evaluate(points)
    values = [np.column_stack((pressure, -material(points)[:, None] * gradient))]
    for collection in local:
        fields = np.empty((len(points), 3))
        for cell, field in enumerate(collection):
            selected = owners == cell
            pressure, gradient = field.evaluate(points[selected])
            fields[selected] = np.column_stack(
                (pressure, -material(points[selected])[:, None] * gradient)
            )
        values.append(fields)
    fig = plt.figure(figsize=(13.5, 11.5), layout="constrained")
    grid = fig.add_gridspec(3, 4, width_ratios=(1, 1, 1, 0.055))
    titles = (
        f"Classical Q{reference.degree} / {reference.mesh.nx}² cells",
        "MHM / 1 segment per face",
        "MHM / 32 segments per face",
    )
    for row, (name, cmap) in enumerate(
        zip(("Pressure", "Flux qx", "Flux qy"), ("viridis", "RdBu_r", "RdBu_r"), strict=True)
    ):
        lower, upper = min(v[:, row].min() for v in values), max(v[:, row].max() for v in values)
        if row:
            upper = max(abs(lower), abs(upper))
            lower = -upper
        for column, (title, field) in enumerate(zip(titles, values, strict=True)):
            ax = fig.add_subplot(grid[row, column])
            artist = ax.imshow(
                field[:, row].reshape(n, n),
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
            draw_macro_mesh(ax, mesh)
            ax.set(title=title if row == 0 else "", xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(artist, cax=fig.add_subplot(grid[row, 3]), label=name)
    save(fig, "fields")


def profile_maps(
    reference_specification: str,
    *,
    macro: int = 8,
    refinement: int = 512,
    samples: int = 512,
    ordinate: float = 0.37,
) -> None:
    """Draw independent one-sided pressure and Darcy-flux limits across macrofaces.

    Each macro interval is a separate line artist with both endpoints retained.
    The classical field is evaluated from inside that interval at its endpoints;
    the plotting coordinates stay on the physical face. The horizontal ordinate
    must avoid macrofaces so each interval has an unambiguous adjacent macrocell.
    Samples describe displayed profiles and never enter the integrated norms.
    """
    if not 0 < ordinate < 1 or np.isclose(
        ordinate * macro, round(ordinate * macro), rtol=0, atol=1e-12
    ):
        raise ValueError("profile ordinate must lie strictly inside a row of macrocells")
    reference, _ = load_reference(reference_specification)
    collections = [load_fields(macro, refinement, segments)[0] for segments in (1, 32)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    labels = ("Pressure", "Darcy flux qx", "Darcy flux qy", "Darcy flux magnitude")
    titles = (f"Classical Q{reference.degree}", "MHM / 1 segment", "MHM / 32 segments")
    styles = (("black", "-"), ("tab:blue", "--"), ("tab:orange", "-"))
    row = int(ordinate * macro)
    for column in range(macro):
        x = np.linspace(column / macro, (column + 1) / macro, max(2, samples // macro + 1))
        points = np.column_stack((x, np.full(len(x), ordinate)))
        inside = points.copy()
        midpoint = (column + 0.5) / macro
        inside[[0, -1], 0] = np.nextafter(inside[[0, -1], 0], midpoint)
        values = []
        for index in range(3):
            field = reference if index == 0 else collections[index - 1][row * macro + column]
            pressure, gradient = field.evaluate(inside if index == 0 else points)
            flux = -material(points)[:, None] * gradient
            values.append(np.column_stack((pressure, flux, np.linalg.norm(flux, axis=1))))
        for ax, component, label in zip(axes.flat, range(4), labels, strict=True):
            for title, (color, linestyle), value in zip(titles, styles, values, strict=True):
                ax.plot(
                    x,
                    value[:, component],
                    color=color,
                    ls=linestyle,
                    label=title if column == 0 else None,
                )
            ax.set(xlabel="x", ylabel=label, title=f"y = {ordinate:g}")
    for ax in axes.flat:
        for face in np.arange(1, macro) / macro:
            ax.axvline(face, color="0.45", lw=0.7, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    save(fig, "profiles")


def problem_maps(*, macro: int = 8) -> None:
    """Show the declared periodic coefficient and source on the actual fixed macrogrid."""
    n = 600
    coordinates = (np.arange(n) + 0.5) / n
    points = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, field, label in zip(
        axes,
        (material(points), source(points)),
        ("Permeability K", "Source sin(x) sin(y)"),
        strict=True,
    ):
        artist = ax.imshow(
            field.reshape(n, n),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="viridis",
            interpolation="nearest",
        )
        draw_macro_mesh(ax, CartesianMacroMesh(macro))
        ax.set(xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(artist, ax=ax, label=label, shrink=0.85)
    save(fig, "problem")


def main() -> None:
    """Render only archived current fields; never run a numerical solve while plotting."""
    global FOLDER, COMPARISON, RECORDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Archived degree:n[:quadrature[:assembly]]")
    parser.add_argument("--artifacts", type=Path, default=compare_periodic.ARTIFACTS)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--output", type=Path, default=FOLDER)
    parser.add_argument("--macro", type=int, default=8)
    parser.add_argument("--local-refinement", type=int)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--profile-y", type=float, default=0.37)
    args = parser.parse_args()
    if (
        args.macro < 1
        or args.samples < 2
        or (args.local_refinement is not None and args.local_refinement < 1)
    ):
        parser.error("macro, local refinement and sample count must be positive")
    FOLDER, COMPARISON, RECORDS = args.output, args.comparison, args.records
    compare_periodic.ARTIFACTS = args.artifacts
    compare_periodic.REFERENCE_RECORDS = args.records
    reference = args.reference or json.loads(COMPARISON.read_text()).get("primary_reference")
    if reference is None:
        parser.error("supply --reference or select primary_reference in the comparison record")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    finest = convergence(reference, macro=args.macro, refinement=args.local_refinement)
    field_maps(reference, macro=args.macro, refinement=finest, samples=args.samples)
    profile_maps(
        reference,
        macro=args.macro,
        refinement=finest,
        samples=args.samples,
        ordinate=args.profile_y,
    )
    problem_maps(macro=args.macro)


if __name__ == "__main__":
    main()
