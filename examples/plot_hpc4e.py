"""Publication views of the heterogeneous mixed-elasticity material and stress fields."""

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FormatStrFormatter, NullFormatter

from examples.hpc4e_data import BOUNDS, LENGTH_SCALE, load_data
from examples.hpc4e_fields import RectangularElasticityField
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/hpc4e"


def overlay(ax, *, alpha: float = 0.4) -> None:
    """Draw actual 16×8 macrorectangle boundaries in physical metres."""
    mesh = CartesianMacroMesh(16, 8, BOUNDS)
    ax.add_collection(
        LineCollection(
            mesh.points[mesh.faces] * LENGTH_SCALE, colors="white", linewidths=0.9, alpha=alpha
        )
    )
    ax.add_collection(
        LineCollection(
            mesh.points[mesh.faces] * LENGTH_SCALE, colors="#222222", linewidths=0.35, alpha=alpha
        )
    )
    ax.set(xlabel="$x$ [m]", ylabel="$z$ [m]", xlim=(0, 10000), ylim=(0, 4500), aspect="equal")


def save(fig, name: str) -> None:
    """Save a matching raster/vector pair with sufficient resolution for fine material pixels."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=220)
    fig.savefig(OUTPUT / f"{name}.svg", dpi=300)
    plt.close(fig)


def materials() -> None:
    """Compare original material samples with the quantities displayed in published Figure 9."""
    data = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout="constrained")
    for ax, values, title, label in zip(
        axes,
        (data.young / 1e6, data.poisson),
        ("Young modulus", "Poisson ratio"),
        ("$E$ [MPa]", "$\\nu$"),
        strict=True,
    ):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=(0, 10000, 0, 4500),
            interpolation="nearest",
            cmap="viridis",
        )
        overlay(ax)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, location="bottom", label=label, shrink=0.85, pad=0.09)
    fig.suptitle("HPC4e material samples · 512 × 256 pixels · 16 × 8 macrorectangles", fontsize=12)
    save(fig, "material")


def profiles(reference_path: Path, segments: list[int]) -> None:
    """Show one-sided stress profiles at the article's z=2250.25 m, without joining jumps."""
    reference = RectangularElasticityField.load(reference_path)
    fine = reference.nx
    x = ((np.arange(fine)[:, None] + np.linspace(1e-8, 1 - 1e-8, 6)) / fine).ravel()
    points = np.column_stack((x, np.full_like(x, 2250.25 / LENGTH_SCALE)))
    target = reference.evaluate(points)[1][:, 0, 0] * 100
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7.8), layout="constrained", sharex=True, sharey=True
    )
    for ax, subdivision in zip(axes.flat, segments, strict=True):
        field = RectangularElasticityField.load(
            ROOT / f"build/results/hpc4e/mhm-s{subdivision}.npz"
        )
        values = field.evaluate(points)[1][:, 0, 0] * 100
        for boundary in np.linspace(0, 10000, 17):
            ax.axvline(boundary, color="0.5", linewidth=0.4, alpha=0.5)
        for samples, color, label, width in (
            (
                target,
                "black",
                f"DOLFINx RT{reference.degree} · {reference.nx} × {reference.ny}",
                1.1,
            ),
            (values, "#d55e00", "PyMHM RT1", 0.85),
        ):
            lines = np.stack((x.reshape(-1, 6) * LENGTH_SCALE, samples.reshape(-1, 6)), axis=-1)
            ax.add_collection(LineCollection(lines, colors=color, linewidths=width, label=label))
        ax.autoscale_view(scalex=False)
        ax.set(
            title=f"$\\ell={int(np.log2(subdivision))}$ · {subdivision} P1 subfaces per macroedge",
            xlabel="$x$ [m]",
            ylabel="$\\sigma_{xx}$ [MPa]",
        )
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Published geometry and spaces · stress at $z=2250.25$ m", fontsize=13)
    save(fig, "stress-profiles")


def published_profiles(segments: list[int]) -> None:
    """Compare signed stress with the article's red curves and pixel uncertainty."""
    metadata = json.loads((ROOT / "examples/results/hpc4e/published-profile.json").read_text())
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8.1), layout="constrained", sharex=True, sharey=True
    )
    x = ((np.arange(512)[:, None] + np.linspace(1e-8, 1 - 1e-8, 6)) / 512).ravel()
    points = np.column_stack((x, np.full_like(x, 2250.25 / LENGTH_SCALE)))
    for ax, subdivision in zip(axes.flat, segments, strict=True):
        field = RectangularElasticityField.load(
            ROOT / f"build/results/hpc4e/mhm-s{subdivision}.npz"
        )
        values = field.evaluate(points)[1][:, 0, 0] * 100
        for boundary in np.linspace(0, 10000, 17):
            ax.axvline(boundary, color="0.5", linewidth=0.4, alpha=0.5)
        lines = np.stack((x.reshape(-1, 6) * LENGTH_SCALE, values.reshape(-1, 6)), axis=-1)
        ax.add_collection(LineCollection(lines, colors="#0072b2", linewidths=1.1, label="PyMHM"))
        curve = next(row for row in metadata["curves"] if row["segments"] == subdivision)
        samples = curve["samples"]
        px = np.array([row["x_m"] for row in samples])
        py = np.array([row["stress_mpa"] for row in samples])
        lower = np.array([row["stress_lower_mpa"] for row in samples])
        upper = np.array([row["stress_upper_mpa"] for row in samples])
        ax.errorbar(
            px,
            py,
            xerr=2 * 10000 / 1170,
            yerr=np.stack((py - lower, upper - py)),
            fmt=".",
            markersize=2,
            color="#d55e00",
            elinewidth=0.45,
            alpha=0.8,
            label="Published NeoPZ MHM: pixel interval",
        )
        ax.set(
            xlim=(0, 10000),
            ylim=(-120, 60),
            xlabel="$x$ [m]",
            ylabel="$\\sigma_{xx}$ [MPa]",
            title=f"$\\ell={int(np.log2(subdivision))}$ · {subdivision} P1 subfaces",
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.12)
    fig.suptitle(
        "Figure 10 comparison · identical RT1 local spaces and material samples", fontsize=12
    )
    save(fig, "published-profile-comparison")


def fields(reference_path: Path) -> None:
    """Render signed stress and displacement with common physical units and macro boundaries."""
    reference = RectangularElasticityField.load(reference_path)
    approximation = RectangularElasticityField.load(ROOT / "build/results/hpc4e/mhm-s8.npz")
    nx, ny = max(reference.nx, approximation.nx), max(reference.ny, approximation.ny)
    x, y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) * 0.45 / ny)
    points = np.column_stack((x.ravel(), y.ravel()))
    u_ref, stress_ref, _, _ = reference.evaluate(points)
    u_mhm, stress_mhm, _, _ = approximation.evaluate(points)
    ref = (stress_ref[:, 0, 0] * 100, u_ref[:, 0] * LENGTH_SCALE, u_ref[:, 1] * LENGTH_SCALE)
    mhm = (stress_mhm[:, 0, 0] * 100, u_mhm[:, 0] * LENGTH_SCALE, u_mhm[:, 1] * LENGTH_SCALE)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08, hspace=0.06)
    for row, (target, actual, label) in enumerate(
        zip(ref, mhm, ("$\\sigma_{xx}$ [MPa]", "$u_x$ [m]", "$u_z$ [m]"), strict=True)
    ):
        limit = max(float(np.max(np.abs(target))), float(np.max(np.abs(actual))))
        for col, values in enumerate((target, actual, actual - target)):
            ax = axes[row, col]
            bound = limit if col < 2 else max(float(np.max(np.abs(values))), np.finfo(float).tiny)
            artist = ax.imshow(
                values.reshape(ny, nx),
                origin="lower",
                extent=(0, 10000, 0, 4500),
                cmap="RdBu_r",
                norm=SymLogNorm(linthresh=0.01 * bound, vmin=-bound, vmax=bound),
                interpolation="nearest",
            )
            overlay(ax, alpha=0.35)
            ax.set_title(
                (
                    f"DOLFINx RT{reference.degree} · {reference.nx} × {reference.ny}",
                    "PyMHM RT1 · $\\ell=3$",
                    "PyMHM − classical",
                )[col],
                fontsize=11,
            )
            fig.colorbar(
                artist,
                ax=ax,
                location="bottom",
                shrink=0.85,
                pad=0.09,
                label=("Difference: " if col == 2 else "") + label,
                ticks=[-bound, -0.1 * bound, 0, 0.1 * bound, bound],
                format=FormatStrFormatter("%.2g"),
            )
    fig.suptitle(
        f"HPC4e · signed physical fields at {nx} × {ny} fine-cell centres\n"
        "Symmetric-log colors; linear region within 1% of each scale limit",
        fontsize=13,
    )
    save(fig, "stress-displacement-fields")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
    qx, qy = np.meshgrid(np.linspace(0.012, 0.988, 29), np.linspace(0.012, 0.438, 13))
    qpoints = np.column_stack((qx.ravel(), qy.ravel()))
    vectors = [field.evaluate(qpoints)[0] * LENGTH_SCALE for field in (reference, approximation)]
    maximum = max(float(np.linalg.norm(value, axis=1).max()) for value in vectors)
    for ax, value, title in zip(
        axes,
        vectors,
        (
            f"DOLFINx RT{reference.degree} · {reference.nx} × {reference.ny}",
            "PyMHM RT1 · $\\ell=3$",
        ),
        strict=True,
    ):
        overlay(ax, alpha=0.25)
        ax.quiver(
            qpoints[:, 0] * LENGTH_SCALE,
            qpoints[:, 1] * LENGTH_SCALE,
            value[:, 0],
            value[:, 1],
            angles="xy",
            scale_units="xy",
            scale=maximum / 260,
            width=0.0028,
            color="#0072b2",
        )
        ax.set_title(title)
    fig.suptitle("Displacement vectors · common physical arrow scale", fontsize=12)
    fig.supxlabel(
        f"Arrow length = {260 / maximum:.3g} × physical displacement in the coordinate units",
        fontsize=10,
    )
    save(fig, "displacement-vectors")


def reference_sensitivity() -> None:
    """Show differences from the fine reference and its own classical h-refinement increment."""
    record_path = ROOT / "examples/results/hpc4e/reference-spatial-refinement.json"
    record = json.loads(record_path.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / record_path.name).write_bytes(record_path.read_bytes())
    degree_record = ROOT / "examples/results/hpc4e/reference-refinement.json"
    (OUTPUT / degree_record.name).write_bytes(degree_record.read_bytes())
    rows = {
        row["approximation"]: row["norms"]
        for row in record["rows"]
        if row["norms"]["quadrature_order"] == 5
    }
    refinement = rows["classical-rt2-512x256-mumps"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, field, title in zip(
        axes.flat,
        ("stress", "displacement", "rotation", "compliance"),
        ("Stress L2", "Displacement L2", "Rotation L2", "Compliance norm"),
        strict=True,
    ):
        values = [100 * rows[f"mhm-s{s}"][f"{field}_relative"] for s in (1, 2, 4, 8)]
        ax.semilogy(range(4), values, "o-", color="#0072b2", label="MHM − RT2 on 1024 × 512")
        ax.axhline(
            100 * refinement[f"{field}_relative"],
            color="#d55e00",
            linestyle="--",
            label="RT2: 512 × 256 → 1024 × 512",
        )
        ax.set(title=title, xlabel="Skeleton level $\\ell$", ylabel="Relative difference [%]")
        ax.set_xticks(range(4))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    fig.suptitle(
        "HPC4e · differences normalized by the RT2 field on 1024 × 512\n"
        "The reference increment is not a continuum error bound",
        fontsize=12,
    )
    save(fig, "reference-sensitivity")


def main() -> None:
    """Render materials and, when requested, the completed reference comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    materials()
    if not args.materials_only:
        if args.reference is None:
            parser.error("--reference is required for the stress comparison")
        profiles(args.reference, [1, 2, 4, 8])
        published_profiles([1, 2, 4, 8])
        fields(args.reference)
        reference_sensitivity()
        inputs = [
            args.reference,
            *(ROOT / f"build/results/hpc4e/mhm-s{s}.npz" for s in (1, 2, 4, 8)),
            ROOT / "examples/results/hpc4e/reference-spatial-refinement.json",
            ROOT / "examples/results/hpc4e/published-profile.json",
        ]
        record = {
            "reference": args.reference.name,
            "field_samples": "Centres of the common fine rectangular partition; no averaging",
            "profile": "Six one-sided points per reference fine interval; z=2250.25 m",
            "macro_mesh": "16 by 8 physical rectangles",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
            },
            "figure_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(OUTPUT.glob("*.png"))
                if path.name
                in {
                    "material.png",
                    "stress-profiles.png",
                    "published-profile-comparison.png",
                    "stress-displacement-fields.png",
                    "displacement-vectors.png",
                    "reference-sensitivity.png",
                }
            },
        }
        payload = json.dumps(record, indent=2) + "\n"
        (ROOT / "examples/results/hpc4e/visualization.json").write_text(payload)
        (OUTPUT / "visualization.json").write_text(payload)


if __name__ == "__main__":
    main()
