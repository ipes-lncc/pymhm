"""Replay physical mapped-RT well fields, macro refinement and a classical five-grid study."""

from __future__ import annotations

import hashlib
import json

import matplotlib

from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize

from examples.plot_style import set_refinement_ticks
from examples.solve_mapped_well import WellData
from pymhm.fem.hdiv.mapped import mapped_rt_basis, mapped_rt_dofs
from pymhm.meshes.hexahedron import HexMesh

ROOT = case_workspace()
INPUT = ROOT / "examples/results/mapped-well"
OUTPUT = ROOT / "docs/figures/mapped-well"


def load(fine: int, macro: int) -> tuple[dict, dict]:
    """Read one numerical record only after verifying its field archive digest."""
    report = json.loads(read_resource_text(INPUT / f"fine{fine}-macro{macro}-q6.json"))
    path = INPUT / report["archive"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != report["sha256"]:
        raise ValueError("mapped well field archive digest mismatch")
    with np.load(local_resource(path)) as data:
        return report, dict(data)


def sample_slice(archive: dict) -> tuple:
    """Evaluate z=0 from the upper incident cells without averaging pressure or flux."""
    data = WellData()
    polygons, computed, analytical, errors, segments = [], [], [], [], []
    corners = np.array([0, 4, 6, 2])
    for macro, vertices in enumerate(archive["macro_points"][archive["macro_cells"]]):
        if not (vertices[:, 2].min() <= 0 < vertices[:, 2].max()):
            continue
        outline = vertices[corners, :2]
        segments.extend(zip(outline, np.roll(outline, -1, axis=0), strict=True))
        mesh = HexMesh(archive["local_points"][macro], archive["local_cells"][macro])
        nodes = mesh.points[mesh.cells]
        selected = np.flatnonzero(
            (nodes[:, :, 2].min(axis=1) <= 0) & (nodes[:, :, 2].max(axis=1) > 0)
        )
        first = selected[0]
        zeta = -nodes[first, :, 2].min() / np.ptp(nodes[first, :, 2])
        reference = np.array([[0.5, 0.5, zeta]])
        physical = mesh.geometry(reference)[0][:, 0]
        basis, _, pressure = mapped_rt_basis(mesh, 1, reference)
        p = archive["pressure"][macro] @ pressure[0]
        q = np.einsum("tia,ti->ta", basis[:, 0], archive["flux"][macro][mapped_rt_dofs(mesh, 1)])
        pe, qe = data.pressure(physical), data.flux(physical)
        for cell in selected:
            polygons.append(nodes[cell, corners, :2])
            computed.append((p[cell] / 1e6, np.linalg.norm(q[cell])))
            analytical.append((pe[cell] / 1e6, np.linalg.norm(qe[cell])))
            errors.append((abs(p[cell] - pe[cell]) / 1e6, np.linalg.norm(q[cell] - qe[cell])))
    return polygons, np.array(computed), np.array(analytical), np.array(errors), segments


def save(fig: plt.Figure, name: str) -> None:
    """Export matching raster/vector figures with dense field artists rasterized."""
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def spatial(macro: int) -> None:
    """Compare exact and numerical pressure/flux with actual macroface intersections."""
    report, archive = load(8, macro)
    polygons, values, exact, error, segments = sample_slice(archive)
    for limit, label in ((50.0, "reservoir"), (1.3, "well")):
        visible = np.array(
            [
                np.min(abs(p)) < limit
                and np.max(p[:, 0]) >= -limit
                and np.min(p[:, 0]) <= limit
                and np.max(p[:, 1]) >= -limit
                and np.min(p[:, 1]) <= limit
                for p in polygons
            ]
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for row, name in enumerate(("Pressure [MPa]", "Darcy flux magnitude [m/s]")):
            lo = min(values[visible, row].min(), exact[visible, row].min()) if row == 0 else 0.0
            hi = max(values[visible, row].max(), exact[visible, row].max())
            for column, field in enumerate((exact[:, row], values[:, row], error[:, row])):
                ax = axes[row, column]
                artist = PolyCollection(
                    polygons,
                    array=field,
                    cmap="viridis" if column < 2 else "magma",
                    norm=Normalize(lo, hi)
                    if column < 2
                    else Normalize(0, max(error[visible, row].max(), 1e-30)),
                    rasterized=True,
                )
                ax.add_collection(artist)
                ax.add_collection(
                    LineCollection(segments, colors="white", linewidths=1.2 if macro == 1 else 0.55)
                )
                ax.add_collection(
                    LineCollection(segments, colors="black", linewidths=0.45 if macro == 1 else 0.2)
                )
                ax.set(
                    xlim=(-limit, limit),
                    ylim=(-limit, limit),
                    aspect="equal",
                    xlabel="x [m]",
                    ylabel="y [m]",
                    title=(
                        f"{name}: {('exact', 'MHM RT1')[column]}"
                        if column < 2
                        else (
                            "Pressure error magnitude [MPa]"
                            if row == 0
                            else "Flux vector error norm [m/s]"
                        )
                    ),
                )
                fig.colorbar(artist, ax=ax, shrink=0.81, pad=0.025)
        fig.suptitle(
            f"Mapped hexahedral RT1 · z=0 upper-side values · {report['macro_cells']} macros\n"
            f"Flux L² error {100 * report['flux_relative']:.3g}% · "
            f"pressure-increment error {100 * report['pressure_increment_relative']:.3g}%",
            fontsize=15,
        )
        save(fig, f"{label}-macro{macro}")


def profiles() -> None:
    """Plot x=y,z=0 profiles from the incident azimuthal sector below pi/4."""
    data = WellData()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    radius = np.geomspace(0.2, 50, 500)
    points = np.column_stack((radius / np.sqrt(2), radius / np.sqrt(2), np.zeros(len(radius))))
    axes[0].semilogx(radius, data.pressure(points) / 1e6, "k--", label="Exact")
    axes[1].loglog(radius, np.linalg.norm(data.flux(points), axis=1), "k--", label="Exact")
    for level, macro_factor in enumerate((1, 2, 4, 8)):
        _, archive = load(8, macro_factor)
        first = True
        for macro, vertices in enumerate(archive["macro_points"][archive["macro_cells"]]):
            side = vertices[[2, 3, 6, 7]]
            if not (
                np.all(abs(side[:, 0] - side[:, 1]) < 1e-11)
                and np.all(side[:, 0] > 0)
                and vertices[:, 2].min() <= 0 < vertices[:, 2].max()
            ):
                continue
            mesh = HexMesh(archive["local_points"][macro], archive["local_cells"][macro])
            nodes = mesh.points[mesh.cells]
            side = nodes[:, [2, 3, 6, 7]]
            selected = np.flatnonzero(
                np.all(abs(side[:, :, 0] - side[:, :, 1]) < 1e-11, axis=1)
                & (nodes[:, :, 2].min(axis=1) <= 0)
                & (nodes[:, :, 2].max(axis=1) > 0)
            )
            zeta = -nodes[selected[0], :, 2].min() / np.ptp(nodes[selected[0], :, 2])
            reference = np.column_stack((np.linspace(0, 1, 11), np.ones(11), np.full(11, zeta)))
            coordinates = mesh.geometry(reference)[0]
            basis, _, pressure = mapped_rt_basis(mesh, 1, reference)
            p = archive["pressure"][macro] @ pressure.T
            q = np.einsum("tqia,ti->tqa", basis, archive["flux"][macro][mapped_rt_dofs(mesh, 1)])
            for cell in selected:
                radius = np.linalg.norm(coordinates[cell, :, :2], axis=1)
                label = f"Macro level {level}" if first else None
                axes[0].semilogx(radius, p[cell] / 1e6, color=f"C{level}", label=label)
                axes[1].loglog(
                    radius, np.linalg.norm(q[cell], axis=1), color=f"C{level}", label=label
                )
                first = False
            for ax in axes:
                # Small colored ticks identify this curve's actual radial macro boundaries.
                endpoints = np.linalg.norm(vertices[[2, 6], :2], axis=1)
                ax.plot(
                    endpoints,
                    [0.025 + 0.022 * level] * 2,
                    "|",
                    color=f"C{level}",
                    transform=ax.get_xaxis_transform(),
                    markersize=7,
                )
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
    axes[0].set(xlabel="Radius along x=y,z=0 [m]", ylabel="Pressure [MPa]")
    axes[1].set(xlabel="Radius along x=y,z=0 [m]", ylabel="Darcy flux magnitude [m/s]")
    fig.suptitle("One-sided profiles · colored ticks: actual macro intersections")
    save(fig, "profiles")


def main() -> None:
    """Render spatial fields and both discretization studies from archived physical results."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for macro in (1, 8):
        spatial(macro)
    profiles()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for axis, records, title in (
        (
            axes[0],
            [load(8, m)[0] for m in (1, 2, 4, 8)],
            "MHM macro refinement · fixed 16,384 fine hexahedra",
        ),
        (
            axes[1],
            [load(n, n)[0] for n in (1, 2, 3, 4, 8)],
            "Classical conforming RT1 · five fine grids",
        ),
    ):
        x = np.array([r["trace_dofs"] + r["pressure_coarse_dofs"] for r in records])
        for key, name in (
            ("flux_relative", "Physical flux"),
            ("pressure_increment_relative", "Pressure increment p − pₑ"),
        ):
            axis.loglog(x, [r[key] for r in records], "o-", label=name)
        set_refinement_ticks(axis, x, labels=[str(v) for v in x])
        axis.tick_params(axis="x", labelsize=8, rotation=20)
        axis.set(
            xlabel="Trace and coarse unknowns before boundary elimination",
            ylabel="Relative physical L² error",
            title=title,
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    save(fig, "convergence")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_mapped_well").main()
