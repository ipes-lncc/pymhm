"""Replay oscillatory-well material, fields, profiles and separate error studies."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm, Normalize

from examples.mapped_well_fields import MappedWellField
from examples.plot_style import set_refinement_ticks
from examples.solve_mapped_oscillatory_well import OscillatoryWellData
from pymhm.meshes.hexahedron import hexahedral_mapping

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/mapped-well-oscillatory"
OUTPUT = ROOT / "docs/figures/mapped-well-oscillatory"
CORNERS = np.array([0, 4, 6, 2])


def save(fig: plt.Figure, name: str) -> None:
    """Export publication raster/vector pairs without enormous dense SVG paths."""
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def finest_norm(row: dict) -> dict:
    """Use the highest recorded integration order of the explicitly identified field pair."""
    return row["norms"][str(max(map(int, row["norms"])))]


def read(name: str) -> tuple[MappedWellField, dict, dict]:
    """Verify provenance before opening any field archive for display."""
    field, record = MappedWellField.load(DATA / (name + ".json"))
    with np.load(DATA / record["archive"]) as archive:
        arrays = dict(archive)
    return field, record, arrays


def macro_segments(arrays: dict) -> list:
    """Return actual macro intersections with the upper incident side of z=0."""
    result = []
    for nodes in arrays["macro_points"][arrays["macro_cells"]]:
        if nodes[:, 2].min() <= 0 < nodes[:, 2].max():
            outline = nodes[CORNERS, :2]
            result.extend(zip(outline, np.roll(outline, -1, axis=0), strict=True))
    return result


def slice_values(field: MappedWellField) -> tuple:
    """Evaluate independent fine-cell centroid samples in the plane z=0."""
    nodes = field.vertices
    ids = np.flatnonzero((nodes[:, :, 2].min(axis=1) <= 0) & (nodes[:, :, 2].max(axis=1) > 0))
    zeta = -nodes[ids[0], :, 2].min() / np.ptp(nodes[ids[0], :, 2])
    reference = np.array([[0.5, 0.5, zeta]])
    pressure, flux = field.values(ids, reference)
    points = hexahedral_mapping(nodes[ids], reference)[0][:, 0]
    return ids, nodes[ids][:, CORNERS, :2], points, pressure[:, 0], flux[:, 0]


def plane_values(field: MappedWellField, grid: MappedWellField) -> tuple[np.ndarray, np.ndarray]:
    """Sample each independent display-cell centroid on the matching physical fine cell."""
    ids, _, points, _, _ = slice_values(grid)
    if grid.factor % field.factor:
        raise ValueError("display grid must refine every compared horizontal grid")
    ratio = grid.factor // field.factor
    shape = np.asarray(grid.shape)
    indices = np.array(np.unravel_index(ids % np.prod(shape), shape)).T
    bases = ids // np.prod(shape)
    vertical = field.shape[2]
    selected = (
        ((indices[:, 0] // ratio) * field.factor + indices[:, 1] // ratio) * vertical
        + vertical // 2
        + bases * np.prod(field.shape)
    )
    values = np.empty(len(ids))
    flux = np.empty((len(ids), 3))
    offsets, groups = np.unique(indices[:, :2] % ratio, axis=0, return_inverse=True)
    for number, offset in enumerate(offsets):
        mask = groups == number
        reference = np.array([[*(offset + 0.5) / ratio, 0.5 if vertical % 2 else 0.0]])
        p, q = field.values(selected[mask], reference)
        physical = hexahedral_mapping(field.vertices[selected[mask]], reference)[0][:, 0]
        if not np.allclose(physical, points[mask], rtol=0, atol=2e-12):
            raise ValueError("display grid does not share the physical annular hierarchy")
        values[mask], flux[mask] = p[:, 0], q[:, 0]
    return values, flux


def spatial(macro: int, reference: MappedWellField, norm: dict, *, fine: int = 8) -> None:
    """Compare an MHM field with the refined classical reference at identical points."""
    field, record, arrays = read(f"fine{fine}-macro{macro}-s1-q40z10")
    _, polygons, _, rp, rq = slice_values(reference)
    p, q = plane_values(field, reference)
    segments = macro_segments(arrays)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for row, values, title in (
        (0, (rp / 1e6, p / 1e6, abs(p - rp) / 1e6), "Pressure [MPa]"),
        (
            1,
            (np.linalg.norm(rq, axis=1), np.linalg.norm(q, axis=1), np.linalg.norm(q - rq, axis=1)),
            "Flux magnitude [m/s]",
        ),
    ):
        limits = (min(values[0].min(), values[1].min()), max(values[0].max(), values[1].max()))
        if row:
            limits = (0.0, limits[1])
        for column, value in enumerate(values):
            ax = axes[row, column]
            artist = PolyCollection(
                polygons,
                array=value,
                cmap="viridis" if column < 2 else "magma",
                norm=Normalize(*limits) if column < 2 else Normalize(0, max(value.max(), 1e-30)),
                rasterized=True,
                edgecolors="none",
                antialiased=False,
            )
            ax.add_collection(artist)
            ax.add_collection(LineCollection(segments, colors="white", linewidths=1.1))
            ax.add_collection(LineCollection(segments, colors="black", linewidths=0.35))
            ax.set(
                xlim=(-50, 50),
                ylim=(-50, 50),
                aspect="equal",
                xlabel="x [m]",
                ylabel="y [m]",
                title=(
                    f"{title}: " + ("PyMHM classical RT1", "PyMHM MHM RT1")[column]
                    if column < 2
                    else "Pressure difference [MPa]"
                    if row == 0
                    else "Flux vector difference norm [m/s]"
                ),
            )
            fig.colorbar(artist, ax=ax, shrink=0.8, pad=0.025)
    fig.suptitle(
        f"Oscillatory producing well · {record['macro_cells']} macros · F{fine} · upper-side z=0\n"
        f"Physical L² difference: flux {100 * norm['flux_relative']:.3g}% · "
        f"pressure increment {100 * norm['pressure_increment_relative']:.3g}%",
        fontsize=14,
    )
    save(fig, f"macro{macro}" if fine == 8 else f"macro{macro}-fine{fine}")


def flux_detail(reference_name: str, *, fine: int = 8) -> None:
    """Resolve the flux dynamic range with logarithmic colors and unchanged sampled fields."""
    reference, _, _ = read(reference_name)
    field, _, arrays = read(f"fine{fine}-macro4-s1-q40z10")
    _, polygons, _, _, reference_flux = slice_values(reference)
    _, mhm_flux = plane_values(field, reference)
    values = [
        np.linalg.norm(reference_flux, axis=1),
        np.linalg.norm(mhm_flux, axis=1),
        np.linalg.norm(mhm_flux - reference_flux, axis=1),
    ]
    shared = np.concatenate(values[:2])
    common_limits = (float(shared[shared > 0].min()), float(shared.max()))
    segments = macro_segments(arrays)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained")
    for column, (ax, value) in enumerate(zip(axes, values, strict=True)):
        limits = (
            common_limits if column < 2 else (float(value[value > 0].min()), float(value.max()))
        )
        artist = PolyCollection(
            polygons,
            array=value,
            norm=LogNorm(*limits),
            cmap="viridis" if column < 2 else "magma",
            edgecolors="none",
            antialiased=False,
            rasterized=True,
        )
        ax.add_collection(artist)
        ax.add_collection(LineCollection(segments, colors="white", linewidths=0.9))
        ax.add_collection(LineCollection(segments, colors="black", linewidths=0.3))
        ax.set(
            xlim=(-50, 50),
            ylim=(-50, 50),
            aspect="equal",
            xlabel="x [m]",
            ylabel="y [m]",
            title=("PyMHM classical RT1", "PyMHM MHM RT1", "Flux vector difference norm")[column],
        )
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.82, pad=0.025)
        colorbar.set_label("Flux magnitude [m/s]" if column < 2 else "Flux difference [m/s]")
    fig.suptitle(
        f"Flux dynamic range · F{fine} · logarithmic color scales · "
        "2,048 actual macros · upper-side z=0"
    )
    save(fig, "flux-logarithmic" if fine == 8 else f"flux-logarithmic-fine{fine}")


def material(reference_name: str) -> None:
    """Show both spatial horizontal components and the complete vertical coefficient profile."""
    field, _, _ = read(reference_name)
    _, _, arrays = read("fine8-macro1-s1-q40z10")
    _, polygons, points, _, _ = slice_values(field)
    data = OscillatoryWellData()
    values = np.diagonal(data.tensor(points) * data.viscosity, axis1=1, axis2=2)
    segments = macro_segments(arrays)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for component, ax in enumerate(axes[:2]):
        value = values[:, component]
        artist = PolyCollection(
            polygons,
            array=value,
            cmap="viridis",
            norm=LogNorm(value.min(), value.max()),
            rasterized=True,
            edgecolors="none",
            antialiased=False,
        )
        ax.add_collection(artist)
        ax.add_collection(LineCollection(segments, colors="black", linewidths=0.5))
        ax.set(
            xlim=(-50, 50),
            ylim=(-50, 50),
            aspect="equal",
            xlabel="x [m]",
            ylabel="y [m]",
            title=f"K{'xy'[component]} [m²] · z=0",
        )
        fig.colorbar(artist, ax=ax, shrink=0.8, pad=0.025)
    z = np.linspace(-5, 5, 501)
    kz = (
        data.tensor(np.column_stack((np.ones_like(z), np.ones_like(z), z)))[:, 2, 2]
        * data.viscosity
    )
    axes[2].plot(z, kz)
    axes[2].set(xlabel="z [m]", ylabel="Kz [m²]", title="Vertical permeability")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].grid(alpha=0.25)
    fig.suptitle("Declared uniformly positive tensor · actual 32-cell macro intersections")
    save(fig, "material")


def profiles(reference_name: str) -> None:
    """Preserve one-sided profiles on both diagonal branches without joining across the well."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for level, name in enumerate(
        [reference_name] + [f"fine8-macro{m}-s1-q40z10" for m in (1, 2, 4)]
    ):
        field, record, arrays = read(name)
        nodes = field.vertices
        side = nodes[:, [2, 3, 6, 7]]
        ids = np.flatnonzero(
            np.all(abs(side[:, :, 0] - side[:, :, 1]) < 1e-11, axis=1)
            & (nodes[:, :, 2].min(axis=1) <= 0)
            & (nodes[:, :, 2].max(axis=1) > 0)
        )
        zeta = -nodes[ids[0], :, 2].min() / np.ptp(nodes[ids[0], :, 2])
        reference = np.column_stack((np.linspace(0, 1, 9), np.ones(9), np.full(9, zeta)))
        points = hexahedral_mapping(nodes[ids], reference)[0]
        p, q = field.values(ids, reference)
        signed = np.sign(points[:, :, 0]) * np.linalg.norm(points[:, :, :2], axis=2)
        radial = np.einsum("tqa,tqa->tq", q[:, :, :2], points[:, :, :2]) / np.linalg.norm(
            points[:, :, :2], axis=2
        )
        color = "black" if level == 0 else f"C{level - 1}"
        for i in range(len(ids)):
            label = (
                ("PyMHM classical RT1" if level == 0 else f"PyMHM MHM level {level - 1}")
                if i == 0
                else None
            )
            axes[0].plot(signed[i], p[i] / 1e6, color=color, label=label, lw=1)
            axes[1].semilogy(
                signed[i], np.linalg.norm(q[i], axis=1), color=color, label=label, lw=1
            )
            axes[2].plot(signed[i], radial[i], color=color, label=label, lw=1)
        if level:
            for vertices in arrays["macro_points"][arrays["macro_cells"]]:
                face = vertices[[2, 3, 6, 7]]
                if (
                    np.all(abs(face[:, 0] - face[:, 1]) < 1e-11)
                    and vertices[:, 2].min() <= 0 < vertices[:, 2].max()
                ):
                    endpoints = np.sign(vertices[[2, 6], 0]) * np.linalg.norm(
                        vertices[[2, 6], :2], axis=1
                    )
                    for ax in axes:
                        ax.plot(
                            endpoints,
                            [0.025 + 0.025 * level] * 2,
                            "|",
                            color=color,
                            transform=ax.get_xaxis_transform(),
                        )
    for ax in axes:
        ax.set(xlabel="Signed radius along x=y,z=0 [m]", xlim=(-50, 50))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Pressure [MPa]")
    axes[1].set(ylabel="Flux magnitude [m/s]")
    axes[2].set(ylabel="Radial flux [m/s] · outward positive")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("One-sided diagonal profiles · colored ticks: actual macro intersections")
    save(fig, "profiles")


def convergence(report: dict) -> None:
    """Separate fixed-fine trace restriction, refined-reference differences and integration."""
    reference_factor = json.loads((DATA / (report["reference"] + ".json")).read_text())[
        "fine_factor"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), layout="constrained")
    for ax, kind, title in zip(
        axes.ravel(),
        (
            "macro trace restriction",
            "MHM versus refined reference",
            "classical spatial refinement",
            "material quadrature",
        ),
        (
            "Trace restriction at fixed fine resolution F8",
            f"MHM versus refined F{reference_factor} reference",
            "Classical reference refinement",
            "Material-quadrature sensitivity",
        ),
        strict=True,
    ):
        rows = [r for r in report["rows"] if r["kind"] == kind]
        if not rows:
            raise ValueError(f"missing physical comparison category: {kind}")
        x = np.arange(1, len(rows) + 1)
        for key, label in (
            ("pressure_increment_relative", "Pressure increment"),
            ("flux_relative", "Physical flux"),
        ):
            ax.semilogy(x, [100 * finest_norm(r)[key] for r in rows], "o-", label=label)
        labels = (
            ["0", "1", "2"]
            if kind in ("macro trace restriction", "MHM versus refined reference")
            else [
                f"{json.loads((DATA / (row['candidate'] + '.json')).read_text())['fine_factor']}"
                "→"
                f"{json.loads((DATA / (row['reference'] + '.json')).read_text())['fine_factor']}"
                for row in rows
            ]
            if kind == "classical spatial refinement"
            else [
                f"F{json.loads((DATA / (row['reference'] + '.json')).read_text())['fine_factor']} "
                f"{json.loads((DATA / (row['candidate'] + '.json')).read_text())['quadrature'][0]}"
                "→"
                f"{json.loads((DATA / (row['reference'] + '.json')).read_text())['quadrature'][0]}"
                for row in rows
            ]
        )
        set_refinement_ticks(ax, x, labels=labels)
        ax.set(
            title=title,
            xlabel="Macro level"
            if kind in ("macro trace restriction", "MHM versus refined reference")
            else "Fine factor"
            if kind == "classical spatial refinement"
            else "Horizontal Gauss order",
            ylabel="Relative physical L² difference [%]",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    save(fig, "convergence")


def local_resolution(report: dict) -> None:
    """Separate local enrichment from its matching-fine and refined-reference comparisons."""
    reference_factor = json.loads((DATA / (report["reference"] + ".json")).read_text())[
        "fine_factor"
    ]
    pairs = (
        ("macro trace restriction", "local-control trace restriction"),
        ("MHM versus refined reference", "local-control refined reference"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, kinds, title in zip(
        axes,
        pairs,
        (
            "Trace restriction: matching classical F8/F16",
            f"Total difference: classical F{reference_factor}",
        ),
        strict=True,
    ):
        rows = [
            next(
                row
                for row in report["rows"]
                if row["kind"] == kind and row["candidate"] == f"fine{factor}-macro4-s1-q40z10"
            )
            for factor, kind in zip((8, 16), kinds, strict=True)
        ]
        for key, label in (
            ("pressure_increment_relative", "Pressure increment"),
            ("flux_relative", "Physical flux"),
        ):
            ax.plot((8, 16), [100 * finest_norm(row)[key] for row in rows], "o-", label=label)
        set_refinement_ticks(ax, (8, 16), labels=("F8", "F16"))
        ax.set(
            title=title,
            xlabel="Fine resolution; macro mesh fixed",
            ylabel="Relative physical L² difference [%]",
            ylim=(0, None),
        )
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Local RT1/Q1 refinement · 2,048 macro hexahedra · unchanged coarse traces")
    save(fig, "local-resolution")


def main() -> None:
    """Render verified physical records without rerunning a numerical solve."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((DATA / "comparisons.json").read_text())
    for row in report["rows"]:
        for key in ("reference", "candidate"):
            record = json.loads((DATA / (row[key] + ".json")).read_text())
            if record["sha256"] != row[key + "_sha256"]:
                raise ValueError("physical norm record does not match its current field archive")
    reference, _, _ = read(report["reference"])
    for macro in (1, 2, 4):
        row = next(
            r
            for r in report["rows"]
            if r["kind"] == "MHM versus refined reference"
            and r["candidate"] == f"fine8-macro{macro}-s1-q40z10"
        )
        spatial(macro, reference, finest_norm(row))
    local_control = next(
        (row for row in report["rows"] if row["kind"] == "local-control refined reference"),
        None,
    )
    if local_control is not None:
        spatial(4, reference, finest_norm(local_control), fine=16)
        local_resolution(report)
        flux_detail(report["reference"], fine=16)
    material(report["reference"])
    profiles(report["reference"])
    flux_detail(report["reference"])
    convergence(report)


if __name__ == "__main__":
    main()
