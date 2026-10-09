"""Replay the stated PGMHM SPE10 spaces and independently refined classical fields."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import cKDTree

from examples.field_sampling import sample_field
from examples.plot_style import set_refinement_ticks
from examples.solve_pgmhm_spe10 import OUTPUT, PGMHMField, load_material, macro_mesh
from examples.spe10_adaptive import ROOT, StructuredRT
from examples.spe10_plot_records import checked_comparison, comparison_reference, digest
from pymhm.io.workspace import local_resource, read_resource_text, resource_glob

FIGURES = ROOT / "docs/figures/pgmhm-spe10"


def read(path: Path) -> dict:
    """Verify the archived physical coefficients before plotting a numerical record."""
    row = json.loads(read_resource_text(path))
    if "archive" in row:
        actual = digest(path.parent / row["archive"])
        if actual != row["archive_sha256"]:
            raise ValueError(f"archive checksum mismatch for {path.name}")
    return row


def save(figure: plt.Figure, output: Path, name: str) -> None:
    """Save reproducible publication figures with vector axes and rasterized dense fields."""
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def macro_overlay(axis: plt.Axes) -> None:
    """Highlight the real two-cell northwest--southeast macro partition."""
    mesh = macro_mesh()
    for face in mesh.faces:
        points = mesh.points[face]
        axis.plot(points[:, 0], points[:, 1], color="black", lw=0.8)
    axis.set(xlim=(0, 1200), ylim=(0, 2200), xlabel="x (ft)", ylabel="y (ft)", aspect="equal")


def pixel_flux(field: PGMHMField, points: np.ndarray) -> np.ndarray:
    """Evaluate raw flux at geological cell centers using each actual containing fine cell."""
    macros = (points[:, 0] / 1200 + points[:, 1] / 2200 > 1).astype(int)
    values = np.zeros((len(points), 2))
    for macro, mesh in enumerate(field.meshes):
        selected = np.flatnonzero(macros == macro)
        vertices = mesh.points[mesh.cells]
        tree = cKDTree(vertices.mean(axis=1))
        _, candidate = tree.query(points[selected], k=16)
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        coordinate = np.einsum(
            "nlab,nlb->nla", inverse[candidate], points[selected, None] - vertices[candidate, 0]
        )
        score = np.minimum(coordinate.min(axis=2), 1 - coordinate.sum(axis=2))
        best = np.argmax(score, axis=1)
        if np.any(score[np.arange(len(best)), best] < -1e-11):
            raise ValueError("a pixel sample lies outside its selected local triangle")
        owners = candidate[np.arange(len(best)), best]
        for cell in np.unique(owners):
            ids = selected[owners == cell]
            values[ids] = field.evaluate(macro, int(cell), points[ids])[1][0]
    return values


def plot(output: Path, data: Path, reference_path: Path | None = None) -> None:
    """Keep geological sampling, broken physical fields and integrated norms distinct."""
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    rows = [read(data / f"pgmhm-s{s}-q5.json") for s in (1, 2, 4, 8, 16)]
    comparisons = [
        checked_comparison(data / f"pgmhm-s{row['segments']}-q5-comparison.json", data)
        for row in rows
    ]
    if any(
        row["archive"] != comparison["mhm"] or row["archive_sha256"] != comparison["mhm_sha256"]
        for row, comparison in zip(rows, comparisons, strict=True)
    ):
        raise ValueError("acquisitions and displayed comparisons identify different fields")
    controls = [
        checked_comparison(path, data)
        for path in sorted(resource_glob(data / "controls", "pgmhm-*-comparison.json"))
    ]
    selected_reference = comparison_reference(data, [*comparisons, *controls])
    if reference_path is not None and reference_path.resolve() != selected_reference.resolve():
        raise ValueError("the requested reference differs from the displayed physical norms")
    reference_path = selected_reference
    reference_record = read(reference_path.with_suffix(".json"))
    if (
        reference_record["archive"] != reference_path.name
        or reference_record["archive_sha256"] != comparisons[0]["reference_sha256"]
    ):
        raise ValueError("reference metadata and chosen archive disagree")
    reference = StructuredRT.load(reference_path)
    fig = plt.figure(figsize=(12.4, 5.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(4.3, 0.8))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    legend_axis = fig.add_subplot(grid[1, 1:])
    legend_axis.set_axis_off()
    material = load_material(rows[0].get("component", "kz"))
    image = axes[0].pcolormesh(
        np.arange(61) * 20,
        np.arange(221) * 10,
        np.log10(material.values.T),
        cmap="viridis",
        rasterized=True,
    )
    macro_overlay(axes[0])
    axes[0].set_title(f"SPE10 layer 1, scalar {rows[0].get('component', 'kz').upper()}")
    fig.colorbar(image, ax=axes[0], label="log₁₀(K / mD)", shrink=0.85)
    y = np.linspace(0, 2200, 1401)
    rp = reference.evaluate(np.column_stack((np.full(len(y), 600), y)))[0]
    for axis in axes[1:]:
        axis.plot(y, rp, color="black", lw=2, label=f"Classical RT2 {reference.nx}×{reference.ny}")
        axis.axvline(1100, color="0.55", ls=":", lw=1)
        axis.set(xlabel="y (ft), x = 600 ft", ylabel="Pressure", xlim=(0, 2200))
        axis.grid(alpha=0.2)
    digitization = OUTPUT / "profile-digitization.json"
    if local_resource(digitization).exists():
        points = json.loads(read_resource_text(digitization))["rows"]
        for axis in axes[1:]:
            axis.errorbar(
                [point["y"] for point in points],
                [point["pressure"] for point in points],
                yerr=[point["pressure_pixel_uncertainty"] for point in points],
                fmt=".",
                color="0.35",
                ms=4,
                capsize=2,
                label="Published reference (digitized)",
            )
    for color_index, row in enumerate(rows):
        with np.load(local_resource(data / row["archive"])) as arrays:
            profile_points = arrays["profile_points"]
            profiles = [arrays[key] for key in ("profile_pressure", "profile_enriched_pressure")]
            for index in range(2):
                for axis, values in zip(axes[1:], profiles, strict=True):
                    axis.plot(
                        profile_points[index, :, 1],
                        values[index],
                        label=f"P0, s={row['segments']} ({row['free_global_dofs']} free)"
                        if index == 0
                        else None,
                        lw=1.25,
                        color=f"C{color_index}",
                    )
    axes[1].set_title("Base pressure; independent macro traces")
    axes[2].set_title("Residual-enriched pressure")
    handles, labels = axes[2].get_legend_handles_labels()
    legend_axis.legend(handles, labels, fontsize=8.5, loc="center", ncols=3, frameon=False)
    save(fig, output, "material-profiles")

    field = PGMHMField(data / "pgmhm-s4-q5.npz")
    sampled = sample_field(field.meshes, tuple(field.fields["pressure"]), 2, 3)
    pressure = sampled["values"].astype(float)
    rp = reference.evaluate(sampled["points"])[0]
    tri = mtri.Triangulation(*sampled["points"].T, triangles=sampled["cells"])
    xs, ys = np.meshgrid((np.arange(60) + 0.5) * 20, (np.arange(220) + 0.5) * 10, indexing="ij")
    points = np.column_stack((xs.ravel(), ys.ravel()))
    q, rq = pixel_flux(field, points), reference.evaluate(points)[1]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 8.3), layout="constrained")
    fields = (rp, pressure, pressure - rp)
    limits = (min(rp.min(), pressure.min()), max(rp.max(), pressure.max()))
    for index, axis in enumerate(axes[0]):
        options = (
            dict(cmap="viridis", vmin=limits[0], vmax=limits[1])
            if index < 2
            else dict(cmap="RdBu_r", vmin=-max(abs(fields[2])), vmax=max(abs(fields[2])))
        )
        image = axis.tripcolor(tri, fields[index], shading="gouraud", rasterized=True, **options)
        fig.colorbar(
            image, ax=axis, label="Pressure" if index < 2 else "p_PGMHM − p_RT2", shrink=0.8
        )
    values = (rq[:, 0], q[:, 0], q[:, 0] - rq[:, 0])
    limit = max(abs(values[0]).max(), abs(values[1]).max())
    for index, axis in enumerate(axes[1]):
        bound = limit if index < 2 else abs(values[index]).max()
        image = axis.pcolormesh(
            np.arange(61) * 20,
            np.arange(221) * 10,
            values[index].reshape(60, 220).T,
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            rasterized=True,
        )
        fig.colorbar(
            image, ax=axis, label="Flux qₓ" if index < 2 else "qₓ,PGMHM − qₓ,RT2", shrink=0.8
        )
    for row in axes:
        for axis, title in zip(
            row,
            ("Classical RT2/P2 reference", "PGMHM P2/P0, s=4 (14 free)", "Signed difference"),
            strict=True,
        ):
            axis.set_title(title)
            macro_overlay(axis)
    for axis in axes[:, 1:].ravel():
        axis.set_ylabel("")
    save(fig, output, "fields")

    references = [read(path) for path in sorted(resource_glob(data, "classical-rt2-*.json"))]
    references.sort(key=lambda row: row["nx"])
    fig = plt.figure(figsize=(12, 4.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(3.7, 0.4))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    for name, label in (
        ("pressure", "Pressure L²"),
        ("flux", "Flux L²"),
        ("flux_energy", "Weighted flux"),
    ):
        key = f"{name}_relative_difference"
        axes[0].loglog(
            [row["nx"] for row in references[1:]],
            [row["norms"][-1][key] for row in references[1:]],
            "o-",
            label=label,
        )
    axes[0].set(
        xlabel="Reference cells along x",
        ylabel="Relative successive increment",
        title="Classical reference's own refinement",
    )
    set_refinement_ticks(axes[0], [row["nx"] for row in references[1:]])
    for axis, variant in zip(axes[1:], ("base", "enriched"), strict=True):
        for name, label in (
            ("pressure_l2", "Pressure L²"),
            ("flux_l2", "Flux L²"),
            ("flux_energy", "Weighted flux"),
        ):
            axis.loglog(
                [row["segments"] for row in rows],
                [r["norms"][-1][f"{variant}_{name}_relative_difference"] for r in comparisons],
                "o-",
                label=label,
            )
        axis.set(
            xlabel="P0 segments per macroface",
            ylabel="Relative difference from RT2",
            title=f"{variant.capitalize()} PGMHM fields",
        )
        set_refinement_ticks(axis, [row["segments"] for row in rows])
    for axis in axes:
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, fontsize=9, loc="center", ncols=3, frameon=False)
    save(fig, output, "norms")
    for name in (
        "material-identification.json",
        "profile-digitization.json",
        "profile-component-comparison.json",
        "ufl-verification.json",
        "quadrature-verification.json",
    ):
        path = OUTPUT / name
        if local_resource(path).exists():
            shutil.copy2(path, output / name)
    for path in resource_glob(data, "*.json"):
        shutil.copy2(path, output / path.name)


def main() -> None:
    """Render archived results without launching a PDE solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT / "kx")
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument(
        "--reference", type=Path, help="Require this archive to match every completed comparison"
    )
    args = parser.parse_args()
    plot(args.output, args.data, args.reference)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_pgmhm_spe10").main()
