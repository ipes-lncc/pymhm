"""Plot archived oscillatory MH2M fields, physical norms and independent trace enrichment."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import AsinhNorm, Normalize

from examples.mh2m_heterogeneous import OscillatoryCoefficient, load_field
from examples.plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from examples.plot_style import set_refinement_ticks
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11}
)


def save(figure: plt.Figure, output: Path, name: str) -> None:
    """Save publication PNG/vector assets with dense field artists rasterized."""
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def refinement(record: dict, output: Path) -> None:
    """Keep classical refinement uncertainty separate from MHM approximation differences."""
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
    references = [row for row in record["references"] if "increment" in row]
    for key, label in (
        ("pressure", "Pressure"),
        ("gradient", "Broken gradient"),
        ("flux", "Physical flux"),
    ):
        axes[0].loglog(
            [r["resolution"] for r in references],
            [r["increment_quadrature_check"][f"{key}_relative_difference"] for r in references],
            "o-",
            label=label,
        )
    axes[0].set(
        xlabel="Classical fine resolution",
        ylabel="Relative consecutive difference",
        title=record.get("reference_refinement_label", "Classical P1 refinement"),
    )
    set_refinement_ticks(axes[0], [r["resolution"] for r in references])
    for method in ("MH2M", "MHM"):
        rows = sorted(
            [
                r
                for r in record["cases"]
                if r["family"] == "global-refinement" and r["method"] == method
            ],
            key=lambda r: r["global_dofs_total"],
        )
        if rows:
            axes[1].loglog(
                [r["global_dofs_total"] for r in rows],
                [r["norms_quadrature_check"]["gradient_relative_difference"] for r in rows],
                "o-",
                label=method.replace("2", "²"),
            )
    axes[1].set(
        xlabel="Total global unknowns",
        ylabel="Relative broken H1 difference",
        title="Macro refinement; declared spaces" + record.get("norm_reference_suffix", ""),
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    save(figure, output, "refinement")


def enrichment(record: dict, output: Path) -> None:
    """Compare Lambda partitions with fixed continuous pressure-trace unknowns."""
    rows = [
        r
        for r in record["cases"]
        if r["method"] == "MH2M" and r["macro_resolution"] == 16 and r["Gamma_segments"] == 1
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
    for refinement in sorted({r["local_refinement"] for r in rows}):
        subset = sorted(
            [r for r in rows if r["local_refinement"] == refinement],
            key=lambda r: r["Lambda_segments"],
        )
        for axis, key in zip(axes, ("pressure", "flux"), strict=True):
            axis.loglog(
                [r["Lambda_segments"] for r in subset],
                [r["norms_quadrature_check"][f"{key}_relative_difference"] for r in subset],
                "o-",
                label=f"Local refinement r={refinement}",
            )
            axis.set(xlabel="Lambda segments per macroedge", ylabel=f"Relative {key} difference")
            set_refinement_ticks(axis, [1, 2, 4, 8])
            axis.grid(alpha=0.25, which="both")
    axes[0].legend()
    figure.suptitle(
        "Fixed continuous Gamma: 289 total / 225 free unknowns"
        + record.get("norm_reference_suffix", "")
    )
    save(figure, output, "fixed-gamma")


def profiles(
    record: dict,
    source: Path,
    output: Path,
    *,
    reference_field: Any | None = None,
    reference_label: str | None = None,
) -> None:
    """Preserve upper/lower traces at y=1/2 and mark the actual macro interfaces."""
    target = next(r for r in record["cases"] if r["family"] == "figure-5-discretization")
    macro = TriangleMesh.unit_square(target["macro_resolution"])
    x = np.linspace(0, 1, 2001)
    points = np.column_stack((x, np.full(len(x), 0.5)))
    reference = (
        load_field(source / record["references"][-1]["archive"])
        if reference_field is None
        else reference_field
    )
    numerical = load_field(source / target["archive"])
    figure, axis = plt.subplots(figsize=(9, 4.2), layout="constrained")
    axis.plot(
        x,
        reference.evaluate(points)[0],
        "k-",
        label=reference_label or f"Classical P1 n={reference.resolution}",
    )
    paper_grid = next((r for r in record["references"] if r["resolution"] == 128), None)
    if paper_grid:
        field = load_field(source / paper_grid["archive"])
        axis.plot(x, field.evaluate(points)[0], color="#377eb8", label="Classical P1 n=128")
    n = numerical.resolution
    knots = np.unique(np.r_[np.linspace(0, 1, n + 1), (np.arange(n) + (0.5 * n) % 1) / n])
    endpoints = np.column_stack((knots[:-1], knots[1:]))
    midpoints = endpoints.mean(axis=1)
    for side, style in ((1, "-"), (-1, "--")):
        center_values, gradients = numerical.evaluate(
            np.column_stack((midpoints, np.full(len(midpoints), 0.5))), y_side=side
        )
        traces = center_values[:, None] + gradients[:, :1] * (endpoints - midpoints[:, None])
        separated_x = np.column_stack((endpoints, np.full(len(endpoints), np.nan))).ravel()
        separated_p = np.column_stack((traces, np.full(len(traces), np.nan))).ravel()
        axis.plot(
            separated_x,
            separated_p,
            style,
            label=f"MH²M n=31, r=4, Γ/Λ segments=2; side {side:+d}",
        )
    positions = macro_profile_breaks(macro, np.array([0.0, 0.5]), np.array([1.0, 0.5]))
    mark_macro_interfaces(axis, positions[1:-1])
    axis.set(
        xlabel="x at y=1/2",
        ylabel="Pressure",
        title="Equation (61), declared gamma=1.8 and Section 8.1 source",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=10.5)
    save(figure, output, "profiles")


def fields(
    record: dict,
    source: Path,
    output: Path,
    *,
    field_loader: Callable[[Path], Any] = load_field,
    flux_asinh: bool = False,
    reference_field: Any | None = None,
    reference_label: str | None = None,
) -> None:
    """Show the Figure-5 spatial configuration with raw, independently sampled flux."""
    target = next(r for r in record["cases"] if r["family"] == "figure-5-discretization")
    arrays = dict(np.load(source / target["archive"]))
    macro = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
    vertices = arrays["vertices"]
    points = vertices.reshape(-1, 2)
    cells = np.arange(len(points)).reshape(-1, 3)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    reference = (
        load_field(source / record["references"][-1]["archive"])
        if reference_field is None
        else reference_field
    )
    numerical = field_loader(source / target["archive"])
    pressure = [reference.evaluate(points)[0], arrays["pressure"].ravel()]
    centers = vertices.mean(axis=1)
    material = OscillatoryCoefficient()(centers)
    flux = [-material[:, None] * field.evaluate(centers)[1] for field in (reference, numerical)]
    quantities = (pressure, [q[:, 0] for q in flux], [q[:, 1] for q in flux])
    figure, axes = plt.subplots(3, 3, figsize=(11.6, 8.8), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.08, h_pad=0.06, wspace=0.12, hspace=0.06)
    display = []
    for row, data in enumerate(quantities):
        data = [*data, data[1] - data[0]]
        common = max(float(np.max(abs(values))) for values in data[:2])
        for column, values in enumerate(data):
            axis = axes[row, column]
            extent = common if column < 2 else float(np.max(abs(values)))
            lower = (
                -extent
                if row > 0 or column == 2
                else min(0.0, float(min(data[0].min(), data[1].min())))
            )
            transformed = flux_asinh and row > 0
            normalization = (
                AsinhNorm(linear_width=0.03 * extent, vmin=lower, vmax=extent)
                if transformed
                else Normalize(vmin=lower, vmax=extent)
            )
            artist = axis.tripcolor(
                triangulation,
                values,
                shading="gouraud" if row == 0 else "flat",
                cmap="RdBu_r" if row > 0 or column == 2 else "viridis",
                norm=normalization,
                rasterized=True,
            )
            macro_artist = draw_macro_mesh(axis, macro)
            macro_artist.set_linewidth(0.2)
            macro_artist.set_alpha(0.4)
            macro_artist.set_path_effects(
                [
                    path_effects.Stroke(linewidth=0.45, foreground="white", alpha=0.4),
                    path_effects.Normal(),
                ]
            )
            axis.set(
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=[
                    reference_label or f"Classical P1 n={reference.resolution}",
                    "MH²M n=31, r=4",
                    "MH²M − classical",
                ][column],
            )
            label = ("Pressure", r"Raw flux $q_x$", r"Raw flux $q_y$")[row]
            colorbar = figure.colorbar(
                artist,
                ax=axis,
                fraction=0.045,
                pad=0.03,
                label=label + (" (asinh)" if transformed else ""),
            )
            if transformed:
                ticks = extent * np.array([-1, -0.1, 0, 0.1, 1])
                colorbar.set_ticks(ticks, labels=[f"{value:.2g}" for value in ticks])
                colorbar.minorticks_off()
            elif row == 0 and column < 2 and lower < 0:
                ticks = np.linspace(lower, extent, 5)
                colorbar.set_ticks(ticks, labels=[f"{value:.3g}" for value in ticks])
            display.append(
                {
                    "row": row,
                    "column": column,
                    "quantity": ("pressure", "raw_flux_x", "raw_flux_y")[row],
                    "scale": "asinh" if transformed else "linear",
                    "linear_width": 0.03 * extent if transformed else None,
                    "limits": [lower, extent],
                    "data_range": [float(values.min()), float(values.max())],
                    "clipped": False,
                }
            )
    name = "fields-asinh" if flux_asinh else "fields"
    save(figure, output, name)
    (output / f"{name}-display.json").write_text(
        json.dumps(
            {
                "panels": display,
                "difference": "MH2M minus classical",
                "reference": reference_label or f"Classical P1 n={reference.resolution}",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Render only archived results; never execute a finite-element solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "examples/results/mh2m-heterogeneous")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/mh2m-heterogeneous")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads((args.source / "comparison.json").read_text())
    for row in (*record["references"], *record["cases"]):
        archive = args.source / row["archive"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != row["archive_sha256"]:
            raise ValueError(f"field archive does not match its numerical record: {archive.name}")
    refinement(record, args.output)
    enrichment(record, args.output)
    profiles(record, args.source, args.output)
    fields(record, args.source, args.output)
    shutil.copy2(args.source / "comparison.json", args.output / "comparison.json")
    equivalence = args.source / "norm-equivalence.json"
    if equivalence.exists():
        shutil.copy2(equivalence, args.output / equivalence.name)


if __name__ == "__main__":
    main()
