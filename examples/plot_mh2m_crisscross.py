"""Plot recovered MH²M crisscross fields and independently integrated comparisons."""

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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.mh2m_crisscross_norms import CrossedP1
from examples.mh2m_heterogeneous import load_field
from examples.plot_mesh import macro_profile_breaks, mark_macro_interfaces
from examples.plot_mh2m_heterogeneous import fields, save
from examples.plot_style import set_refinement_ticks
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]


def load_crossed(path: Path) -> CrossedP1:
    """Restore canonical fine triangles without averaging independent macro values."""
    with np.load(path) as arrays:
        return CrossedP1.from_arrays(arrays["vertices"], arrays["pressure"])


def profile_segments(field: CrossedP1, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep distinct endpoint values on every triangle intersecting y=1/2."""
    n = field.resolution
    fraction = (0.5 * n) % 1
    knots = np.unique(
        np.r_[
            np.linspace(0, 1, n + 1),
            (np.arange(n) + fraction) / n,
            (np.arange(n) + 1 - fraction) / n,
        ]
    )
    endpoints = np.column_stack((knots[:-1], knots[1:]))
    midpoints = endpoints.mean(axis=1)
    values, gradient = field.evaluate(
        np.column_stack((midpoints, np.full(len(midpoints), 0.5))), y_side=side
    )
    values = values[:, None] + gradient[:, :1] * (endpoints - midpoints[:, None])
    return (
        np.column_stack((endpoints, np.full(len(endpoints), np.nan))).ravel(),
        np.column_stack((values, np.full(len(values), np.nan))).ravel(),
    )


def profiles(
    record: dict,
    source: Path,
    output: Path,
    *,
    reference_field: Any | None = None,
    reference_label: str | None = None,
) -> None:
    """Compare the Figure-5 profile with two classical resolutions and both traces."""
    target = next(row for row in record["cases"] if row["name"] == "figure-5")
    with np.load(source / target["archive"]) as arrays:
        mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
    numerical = load_crossed(source / target["archive"])
    x = np.linspace(0, 1, 3001)
    points = np.column_stack((x, np.full(len(x), 0.5)))
    figure, axis = plt.subplots(figsize=(10.8, 4.6), layout="constrained")
    for n, style in ((128, "--"), (1024, "-")):
        if n == 1024 and reference_field is not None:
            axis.plot(x, reference_field.evaluate(points)[0], style, label=reference_label)
            continue
        row = next(r for r in record["references"] if r["resolution"] == n)
        reference = load_field(source / row["archive"])
        axis.plot(x, reference.evaluate(points)[0], style, label=f"Classical P1 n={n}")
    for side, style in ((1, "-"), (-1, ":")):
        axis.plot(
            *profile_segments(numerical, side),
            style,
            label=f"MH²M crisscross, incident y-side {side:+d}",
        )
    intersections = macro_profile_breaks(mesh, np.array([0.0, 0.5]), np.array([1.0, 0.5]))
    mark_macro_interfaces(axis, intersections[1:-1])
    axis.set(
        xlabel="x at y=1/2",
        ylabel="Pressure",
        title="Figure-5 spaces; recovered crisscross mesh, ε=1/14, γ=1.8",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    save(figure, output, "profiles")


def published_profiles(record: dict, source: Path, output: Path) -> None:
    """Compare visible published raster spans without inferring occluded curves."""
    digitized = source / "published-profiles.json"
    if not digitized.exists():
        return
    publication = json.loads(digitized.read_text())
    for printed in publication["figures"]:
        figure, axes = plt.subplots(3, 2, figsize=(10.8, 10.6), layout="constrained")
        figure.get_layout_engine().set(hspace=0.09, wspace=0.08)
        for axis, curve in zip(axes.flat, printed["curves"], strict=True):
            columns = curve["visible_columns"]
            pixels = np.array([c["x_pixel"] for c in columns])
            groups = np.split(np.arange(len(columns)), np.flatnonzero(np.diff(pixels) > 1) + 1)
            uncertainty = printed["coordinate_uncertainty"]["pressure"]
            for index, group in enumerate(groups):
                x = np.array([columns[i]["x"] for i in group])
                low = np.array([columns[i]["pressure_low"] for i in group]) - uncertainty
                high = np.array([columns[i]["pressure_high"] for i in group]) + uncertainty
                axis.fill_between(
                    x,
                    low,
                    high,
                    color="#444444",
                    alpha=0.45,
                    label="Published visible raster ±1 pixel" if index == 0 else None,
                )
            if curve["method"] == "reference":
                row = next(r for r in record["references"] if r["resolution"] == 128)
                reference = load_field(source / row["archive"])
                x = np.linspace(0, 1, 3001)
                axis.plot(
                    x,
                    reference.evaluate(np.column_stack((x, np.full(len(x), 0.5))))[0],
                    color="#0072B2",
                    label="Classical P1 n=128",
                )
                title = "Classical reference"
            else:
                name = f"figure-{printed['figure']}-n{curve['macro_resolution']}"
                matches = [r for r in record["cases"] if r["name"] == name]
                if not matches:
                    axis.set_visible(False)
                    continue
                row = matches[0]
                field = load_crossed(source / row["archive"])
                for side, style, color in ((1, "-", "#0072B2"), (-1, ":", "#D55E00")):
                    axis.plot(
                        *profile_segments(field, side),
                        style,
                        color=color,
                        label=f"PyMHM, incident y-side {side:+d}",
                    )
                with np.load(source / row["archive"]) as arrays:
                    mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
                positions = macro_profile_breaks(mesh, np.array([0.0, 0.5]), np.array([1.0, 0.5]))
                mark_macro_interfaces(axis, positions[1:-1])
                title = f"{row['method'].replace('2', '²')}, n={row['macro_resolution']}"
            axis.set(
                xlabel="x at y=1/2",
                ylabel="Pressure",
                title=f"{title}\n{len(columns)} visible raster columns",
            )
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        handles, labels = axes.flat[1].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=3,
            fontsize=10,
        )
        figure.suptitle(
            f"Article Figure {printed['figure']}: pressure profiles, without amplitude rescaling",
            fontsize=13,
        )
        save(figure, output, f"published-figure-{printed['figure']}-comparison")


def figure6_profiles(
    record: dict,
    source: Path,
    output: Path,
    *,
    reference_field: Any | None = None,
    reference_label: str | None = None,
) -> None:
    """Show the declared Figure-6 configurations, retaining every incident value."""
    x = np.linspace(0, 1, 3001)
    points = np.column_stack((x, np.full(len(x), 0.5)))
    reference_row = next(row for row in record["references"] if row["resolution"] == 1024)
    reference = (
        load_field(source / reference_row["archive"])
        if reference_field is None
        else reference_field
    )
    specifications = {
        "upper": (
            ("figure-6-upper-MHM-n8", "figure-6-upper-MH2M-n16"),
            ("figure-6-upper-MHM-n30", "figure-6-upper-MH2M-n65"),
        ),
        "lower": (
            (
                "figure-6-lower-dofs-MHM-n5-lambda4",
                "figure-6-lower-dofs-MH2M-n5-lambda4",
                "figure-6-lower-dofs-MH2M-n5-lambda8",
            ),
            (
                "figure-6-lower-dofs-MHM-n9-lambda4",
                "figure-6-lower-dofs-MH2M-n10-lambda4",
                "figure-6-lower-dofs-MH2M-n10-lambda8",
            ),
        ),
    }
    for name, panels in specifications.items():
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), layout="constrained")
        for axis, names in zip(axes, panels, strict=True):
            axis.plot(
                x,
                reference.evaluate(points)[0],
                color="#444444",
                label=reference_label or "Classical P1 n=1024",
            )
            intersections = []
            for case, color in zip(names, ("#D55E00", "#0072B2", "#009E73"), strict=False):
                row = next(row for row in record["cases"] if row["name"] == case)
                field = load_crossed(source / row["archive"])
                with np.load(source / row["archive"]) as arrays:
                    mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
                intersections.extend(
                    macro_profile_breaks(mesh, np.array([0.0, 0.5]), np.array([1.0, 0.5]))[1:-1]
                )
                label = (
                    f"{row['method'].replace('2', '²')}, "
                    f"sΛ={row['Lambda_segments']}, {row['global_dofs_total']} DOFs"
                )
                for side, style in ((1, "-"), (-1, ":")):
                    axis.plot(
                        *profile_segments(field, side),
                        style,
                        color=color,
                        label=label if side == 1 else None,
                    )
            for position in np.unique(intersections):
                axis.axvline(
                    position, color="#66737a", linewidth=0.45, linestyle=":", alpha=0.3, zorder=0
                )
            axis.set(xlabel="x at y=1/2", ylabel="Pressure")
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=10)
        title = (
            "Figure-6 upper-row configurations: crisscross r=4, sΓ=sΛ=1"
            if name == "upper"
            else "Figure-6 lower-row dimension controls: crisscross r=8, sΓ=4"
        )
        figure.suptitle(title + "\nSolid/dotted: independent incident y-sides +1/−1", fontsize=13)
        save(figure, output, f"figure-6-{name}-profiles")


def norms(record: dict, output: Path) -> None:
    """Keep macro refinement and fixed-Gamma conormal enrichment distinct."""
    rows = [r for r in record["cases"] if "norms_quadrature_check" in r]
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), layout="constrained")
    for method, prefix in (("MHM", "figure-7-"), ("MH²M", "figure-8-")):
        selected = [r for r in rows if r["name"].startswith(prefix)]
        for axis, quantity in zip(axes, ("gradient", "flux"), strict=True):
            if selected:
                axis.loglog(
                    [r["global_dofs_total"] for r in selected],
                    [
                        r["norms_quadrature_check"][quantity + "_relative_difference"]
                        for r in selected
                    ],
                    "o-",
                    label=method,
                )
                axis.legend()
            axis.set(
                xlabel="Total global unknowns",
                ylabel=f"Relative {quantity} difference",
                title=f"Crisscross P1 locals, r=8; {quantity}"
                + record.get("norm_reference_suffix", ""),
            )
            axis.grid(alpha=0.25, which="both")
    measured_dofs = sorted(
        {r["global_dofs_total"] for r in rows if r["name"].startswith("figure-7-")}
    )
    if len(measured_dofs) == 5:
        measured_dofs = [measured_dofs[i] for i in (0, 1, 2, 4)]
    for axis in axes:
        set_refinement_ticks(axis, measured_dofs)
    save(figure, output, "refinement")
    selected = sorted(
        [r for r in rows if r["name"].startswith("fixed-Gamma-") or r["name"] == "figure-8-n16"],
        key=lambda r: r["Lambda_segments"],
    )
    if selected:
        figure, axis = plt.subplots(figsize=(7.8, 4.2), layout="constrained")
        for key, label in (("pressure", "Pressure"), ("gradient", "Gradient"), ("flux", "Flux")):
            axis.loglog(
                [r["Lambda_segments"] for r in selected],
                [r["norms_quadrature_check"][key + "_relative_difference"] for r in selected],
                "o-",
                label=label,
            )
        set_refinement_ticks(axis, [r["Lambda_segments"] for r in selected])
        axis.set(
            xlabel="Constant conormal segments per macroedge",
            ylabel="Relative difference",
            title="Fixed Γ: 289 total / 225 free unknowns; crisscross local r=8"
            + record.get("norm_reference_suffix", ""),
        )
        axis.legend()
        axis.grid(alpha=0.25, which="both")
        save(figure, output, "fixed-gamma")


def main() -> None:
    """Render only completed archived fields and copy their current scientific record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "examples/results/mh2m-heterogeneous/crisscross"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/figures/mh2m-heterogeneous/crisscross"
    )
    args = parser.parse_args()
    record = json.loads((args.source / "comparison.json").read_text())
    for row in (*record["cases"], *record["references"]):
        archive = args.source / row["archive"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != row["archive_sha256"]:
            raise ValueError("archive digest differs from the acquired numerical record")
    args.output.mkdir(parents=True, exist_ok=True)
    adapted = {
        **record,
        "cases": [
            {**r, "family": "figure-5-discretization"}
            for r in record["cases"]
            if r["name"] == "figure-5"
        ],
    }
    fields(adapted, args.source, args.output, field_loader=load_crossed)
    fields(adapted, args.source, args.output, field_loader=load_crossed, flux_asinh=True)
    profiles(record, args.source, args.output)
    published_profiles(record, args.source, args.output)
    figure6_profiles(record, args.source, args.output)
    norms(record, args.output)
    shutil.copy2(args.source / "comparison.json", args.output / "comparison.json")


if __name__ == "__main__":
    main()
