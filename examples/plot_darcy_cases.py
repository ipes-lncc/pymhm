"""Render Darcy solutions, analytical comparisons and conservation diagnostics.

Run with ``pixi run -e notebooks python examples/plot_darcy_cases.py``.
The figures verify specified analytical problems; they do not reproduce a
published numerical table. All reported integrals use explicit quadrature.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from field_sampling import sample_darcy_pressure_profile
from manufactured import darcy_flux, darcy_pressure, darcy_source
from numpy.typing import NDArray
from plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import DarcySolution, solve_darcy
from pymhm.fem.scalar.operators import rt0_evaluate, rt0_operators, triangle_quadrature
from pymhm.recovery.equilibrated import EquilibratedFlux, equilibrate_flux

Array = NDArray[np.float64]
Field = Callable[[Array], Array]


@dataclass
class Snapshot:
    """Plot-ready broken fields with independent elementwise RMS errors."""

    triangulation: Triangulation
    pressure: Array
    shading: str
    centers: Array
    flux: Array
    pressure_rms: Array
    flux_rms: Array
    macro_mesh: TriangleMesh


def snapshot(solution: DarcySolution, exact_pressure: Field, exact_flux: Field) -> Snapshot:
    """Keep macro traces broken while evaluating errors with Duffy order eight."""
    points, cells, pressures, centers, fluxes, p_errors, q_errors = [], [], [], [], [], [], []
    offset = 0
    bary, weights = triangle_quadrature(8)
    for mesh, pressure, flux in zip(
        solution.local_meshes, solution.pressure, solution.flux, strict=True
    ):
        positions = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        centroid = mesh.points[mesh.cells].mean(axis=1)
        if solution.formulation == "primal":
            p_values = pressure[mesh.cells] @ bary.T
            q_values = np.broadcast_to(flux[:, None], positions.shape)
            q_centers = flux
        else:
            p_values = np.broadcast_to(pressure[:, None], positions.shape[:2])
            q_values = rt0_evaluate(mesh, flux, bary)
            q_centers = rt0_evaluate(mesh, flux, np.full((1, 3), 1 / 3))[:, 0]
        p_difference = p_values - exact_pressure(positions.reshape(-1, 2)).reshape(p_values.shape)
        q_difference = q_values - exact_flux(positions.reshape(-1, 2)).reshape(q_values.shape)
        points.append(mesh.points)
        cells.append(mesh.cells + offset)
        pressures.append(pressure)
        centers.append(centroid)
        fluxes.append(q_centers)
        p_errors.append(np.sqrt(p_difference**2 @ weights))
        q_errors.append(np.sqrt(np.sum(q_difference**2, axis=2) @ weights))
        offset += len(mesh.points)
    coordinates = np.concatenate(points)
    return Snapshot(
        Triangulation(coordinates[:, 0], coordinates[:, 1], np.concatenate(cells)),
        np.concatenate(pressures),
        "gouraud" if solution.formulation == "primal" else "flat",
        np.concatenate(centers),
        np.concatenate(fluxes),
        np.concatenate(p_errors),
        np.concatenate(q_errors),
        solution.skeleton.mesh,
    )


def axis_geometry(axis: Any, title: str) -> None:
    """Apply identical physical limits and aspect ratio to a solution panel."""
    axis.set(title=title, xlabel="$x$", ylabel="$y$", xlim=(0, 1), ylim=(0, 1))
    axis.set_aspect("equal")


def cell_map(
    axis: Any, data: Snapshot, values: Array, title: str, norm: Normalize, *, shading: str = "flat"
) -> Any:
    """Plot actual P1 nodal or cellwise values without smoothing P0 fields."""
    cmap = "coolwarm" if norm.vmin < 0 else "viridis"
    if shading == "gouraud":
        artist = axis.tripcolor(
            data.triangulation, values, shading=shading, norm=norm, cmap=cmap, rasterized=True
        )
    else:
        artist = axis.tripcolor(
            data.triangulation, facecolors=values, norm=norm, cmap=cmap, rasterized=True
        )
    axis_geometry(axis, title)
    draw_macro_mesh(axis, data.macro_mesh)
    return artist


def exact_map(
    axis: Any, field: Field, title: str, norm: Normalize, macro_mesh: TriangleMesh
) -> Any:
    """Sample an analytical field on a display grid independent of the FEM mesh."""
    x, y = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    values = field(np.column_stack((x.ravel(), y.ravel()))).reshape(x.shape)
    artist = axis.pcolormesh(
        x,
        y,
        values,
        shading="auto",
        norm=norm,
        cmap="coolwarm" if norm.vmin < 0 else "viridis",
        rasterized=True,
    )
    axis_geometry(axis, title)
    draw_macro_mesh(axis, macro_mesh)
    return artist


def save_figure(figure: Any, directory: Path, name: str) -> None:
    """Write vector labels with rasterized dense fields, and a high-resolution PNG."""
    figure.savefig(directory / f"{name}.svg", dpi=220)
    figure.savefig(directory / f"{name}.png", dpi=180)
    plt.close(figure)


def pressure_comparison(
    directory: Path,
    name: str,
    data: tuple[Snapshot, Snapshot],
    exact: Field,
    limits: tuple[float, float],
) -> None:
    """Compare exact, primal and mixed pressure with a shared physical scale."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    norm = Normalize(*limits)
    exact_map(axes[0], exact, "Analytical pressure", norm, data[0].macro_mesh)
    for axis, item, label in zip(axes[1:], data, ("Primal P1", "Mixed RT0/P0"), strict=True):
        artist = cell_map(axis, item, item.pressure, label, norm, shading=item.shading)
    figure.colorbar(artist, ax=axes, label="Pressure $p$", shrink=0.82)
    save_figure(figure, directory, name)


def cosine_figures(
    directory: Path,
    data: tuple[Snapshot, Snapshot],
    *,
    name: str,
    resolution: str,
    relative_flux_errors: tuple[float, float],
) -> None:
    """Render pressure, physical flux and independent local RMS errors."""
    pressure_comparison(directory, f"{name}-pressure", data, darcy_pressure, (-1, 1))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    maximum = max(np.pi, *(np.linalg.norm(item.flux, axis=1).max() for item in data))
    norm = Normalize(0, maximum)
    exact_map(
        axes[0],
        lambda x: np.linalg.norm(darcy_flux(x), axis=1),
        "Analytical Darcy flux",
        norm,
        data[0].macro_mesh,
    )
    x, y = np.meshgrid(np.linspace(0.04, 0.96, 11), np.linspace(0.04, 0.96, 11))
    positions = np.column_stack((x.ravel(), y.ravel()))
    arrows = darcy_flux(positions)
    axes[0].quiver(*positions.T, *arrows.T, scale=48, width=0.004, color="black")
    for axis, item, label, error in zip(
        axes[1:], data, ("Raw primal flux", "Mixed RT0 flux"), relative_flux_errors, strict=True
    ):
        title = f"{label}\nRelative $L^2$ error: {100 * error:.2f}%"
        artist = cell_map(axis, item, np.linalg.norm(item.flux, axis=1), title, norm)
        stride = max(1, len(item.centers) // 150)
        axis.quiver(
            *item.centers[::stride].T,
            *item.flux[::stride].T,
            scale=48,
            width=0.004,
            color="black",
        )
    figure.colorbar(artist, ax=axes, label="$|q|$ at cell centroids", shrink=0.82)
    figure.suptitle(resolution)
    save_figure(figure, directory, f"{name}-flux")

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, field, label in ((0, "pressure_rms", "$p_h-p$"), (1, "flux_rms", "$q_h-q$")):
        norm = Normalize(0, max(getattr(item, field).max() for item in data))
        for axis, item, method in zip(axes[row], data, ("Primal", "Mixed"), strict=True):
            artist = cell_map(
                axis, item, getattr(item, field), f"{method}: element RMS error", norm
            )
        figure.colorbar(artist, ax=axes[row], label=f"Element RMS of {label}", shrink=0.82)
    figure.suptitle(resolution)
    save_figure(figure, directory, f"{name}-errors")


def layered_permeability(points: Array) -> Array:
    """Return the two scalar permeabilities separated by the fitted vertical interface."""
    return np.where(points[:, 0] < 0.5, 1.0, 1000.0)


def layered_pressure(points: Array) -> Array:
    """Return continuous pressure with physical flux exactly equal to (1,0)."""
    x = points[:, 0]
    return np.where(x <= 0.5, 1 - x, 0.5 - (x - 0.5) / 1000)


def layered_flux(points: Array) -> Array:
    """Return the analytical normal-continuous horizontal Darcy flux."""
    return np.tile([1.0, 0.0], (len(points), 1))


def layered_figures(
    directory: Path, solutions: tuple[DarcySolution, DarcySolution], data: tuple[Snapshot, Snapshot]
) -> None:
    """Show the fitted contrast problem and resolve its much smaller right-hand slope."""
    pressure_comparison(directory, "layered-pressure", data, layered_pressure, (0.4995, 1))
    x = np.linspace(1e-5, 1 - 1e-5, 601)
    positions = np.column_stack((x, np.full(len(x), 0.37)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    crossings = macro_profile_breaks(solutions[0].skeleton.mesh, (0, 0.37), (1, 0.37))[1:-1]
    for axis in axes:
        mark_macro_interfaces(axis, crossings, label=True)
        axis.plot(x, layered_pressure(positions), "k-", linewidth=2, label="Analytical")
        primal = sample_darcy_pressure_profile(solutions[0], (0, 0.37), (1, 0.37))
        for segment, (points, pressure) in enumerate(
            zip(primal["profile_points"], primal["profile_values"], strict=True)
        ):
            axis.plot(
                points[:, 0],
                pressure,
                "o-",
                markersize=3,
                markerfacecolor="none",
                linewidth=1,
                color="tab:blue",
                label="Primal P1" if segment == 0 else "_nolegend_",
            )
        profile = sample_darcy_pressure_profile(solutions[1], (0, 0.37), (1, 0.37))
        for segment, (points, pressure) in enumerate(
            zip(profile["profile_points"], profile["profile_values"], strict=True)
        ):
            axis.plot(
                points[:, 0],
                pressure,
                color="tab:orange",
                linewidth=1.2,
                label="Mixed P0" if segment == 0 else "_nolegend_",
            )
        axis.axvline(0.5, color="grey", linestyle="--", linewidth=1)
        axis.set(xlabel="$x$ at $y=0.37$", ylabel="Pressure $p$")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set(title="Pressure drop across both materials", xlim=(0, 1))
    axes[1].set(
        title="High-permeability region: magnified pressure scale",
        xlim=(0.5, 1),
        ylim=(0.49948, 0.50002),
    )
    save_figure(figure, directory, "layered-profile")


def raw_balances(solution: DarcySolution) -> tuple[Array, Array]:
    """Integrate the unit-permeability raw P1 flux on fine and macro boundaries."""
    macro, fine_defects = [], []
    for mesh, flux in zip(solution.local_meshes, solution.flux, strict=True):
        source = rt0_operators(mesh, 1.0, solution.source, solution.quadrature_order)[2]
        outward = mesh.normals[mesh.cell_faces] * mesh.signs[:, :, None]
        fine_outflow = np.sum(
            np.sum(flux[:, None] * outward, axis=2) * mesh.lengths[mesh.cell_faces], axis=1
        )
        fine_defects.append(fine_outflow - source)
        boundary = mesh.boundary_faces
        q = flux[mesh.face_cells[boundary, 0]]
        macro.append(
            np.sum(np.sum(q * mesh.normals[boundary], axis=1) * mesh.lengths[boundary])
            - source.sum()
        )
    return np.asarray(macro), np.concatenate(fine_defects)


def reconstruction_errors(recovered: EquilibratedFlux) -> Array:
    """Compute independent elementwise RMS errors of the recovered RT0 field."""
    bary, weights = triangle_quadrature(8)
    errors = []
    for mesh, flux in zip(recovered.meshes, recovered.coefficients, strict=True):
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        difference = rt0_evaluate(mesh, flux, bary) - darcy_flux(points.reshape(-1, 2)).reshape(
            len(mesh.cells), len(weights), 2
        )
        errors.append(np.sqrt(np.sum(difference**2, axis=2) @ weights))
    return np.concatenate(errors)


def conservation_figures(
    directory: Path,
    solutions: tuple[DarcySolution, DarcySolution],
    data: tuple[Snapshot, Snapshot],
    recovered: EquilibratedFlux,
) -> dict[str, float]:
    """Compare balances at both scales and separate conservation from flux accuracy."""
    primal, mixed = solutions
    raw_macro, raw_fine = raw_balances(primal)
    recovered_fine = recovered.conservation_residuals()

    def maximum(values: Any) -> float:
        """Return an absolute maximum without rounding small conservation defects."""
        return float(np.max(np.abs(values)))

    metrics = {
        "raw_macro_max": maximum(raw_macro),
        "raw_fine_max": maximum(raw_fine),
        "primal_skeleton_macro_max": maximum(primal.conservation_residuals()),
        "recovered_macro_max": maximum([values.sum() for values in recovered_fine]),
        "recovered_fine_max": maximum(np.concatenate(recovered_fine)),
        "mixed_macro_max": maximum(mixed.conservation_residuals()),
        "mixed_fine_max": maximum(np.concatenate(mixed.fine_conservation_residuals())),
        "recovered_flux_l2": recovered.l2_error(darcy_flux, order=8),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    macro = [
        metrics[key]
        for key in (
            "raw_macro_max",
            "primal_skeleton_macro_max",
            "recovered_macro_max",
            "mixed_macro_max",
        )
    ]
    fine = [
        metrics["raw_fine_max"],
        np.nan,
        metrics["recovered_fine_max"],
        metrics["mixed_fine_max"],
    ]
    positions = np.arange(4)
    axes[0].bar(positions - 0.18, np.maximum(macro, 1e-16), width=0.36, label="Macrocell")
    axes[0].bar(positions + 0.18, np.maximum(fine, 1e-16), width=0.36, label="Fine cell")
    axes[0].set(
        yscale="log",
        ylabel="Max. |integrated outflow − source|",
        title="Conservation depends on the field and spatial scale",
        xticks=positions,
        xticklabels=["Raw P1\ngradient", "Primal\nskeleton", "Equilibrated\nRT0", "Mixed\nRT0"],
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].text(
        0.34,
        0.34,
        "Display floor: $10^{-16}$\nNo fine-cell balance is assigned\nto the skeleton alone.",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    errors = [
        primal.flux_l2_error(darcy_flux, order=8),
        metrics["recovered_flux_l2"],
        mixed.flux_l2_error(darcy_flux, order=8),
    ]
    axes[1].bar(
        ["Raw P1", "Equilibrated RT0", "Mixed RT0"],
        errors,
        color=["tab:blue", "tab:green", "tab:orange"],
    )
    axes[1].set(ylabel=r"$\|q_h-q\|_{L^2(\Omega)}$", title="Flux accuracy is a separate diagnostic")
    axes[1].grid(axis="y", alpha=0.2)
    save_figure(figure, directory, "conservation")

    recovered_errors = reconstruction_errors(recovered)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    errors_by_field = [data[0].flux_rms, recovered_errors, data[1].flux_rms]
    norm = Normalize(0, max(values.max() for values in errors_by_field))
    for axis, item, errors, label in zip(
        axes,
        (data[0], data[0], data[1]),
        errors_by_field,
        ("Raw P1 gradient", "Equilibrated RT0", "Mixed RT0"),
        strict=True,
    ):
        artist = cell_map(axis, item, errors, label, norm)
    figure.colorbar(artist, ax=axes, label="Element RMS physical flux error", shrink=0.82)
    save_figure(figure, directory, "equilibration-errors")
    return metrics


@threadpool_limits.wrap(limits=1)
def main() -> None:
    """Solve declared analytical problems, save ten figure pairs and record metrics."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=root / "docs/figures/darcy")
    parser.add_argument("--macro-resolution", type=int, default=4)
    parser.add_argument("--local-refinement", type=int, default=4)
    args = parser.parse_args()
    if args.macro_resolution < 2 or args.macro_resolution % 2:
        parser.error("macro resolution must be positive and even to fit the material interface")
    if args.local_refinement < 1:
        parser.error("local refinement must be positive")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "svg.fonttype": "none"})
    mesh = TriangleMesh.unit_square(args.macro_resolution)
    cosine = tuple(
        solve_darcy(
            mesh,
            source=darcy_source,
            dirichlet=darcy_pressure,
            formulation=formulation,
            local_refinement=args.local_refinement,
            quadrature_order=6,
        )
        for formulation in ("primal", "mixed")
    )
    cosine_data = tuple(snapshot(solution, darcy_pressure, darcy_flux) for solution in cosine)
    cosine_figures(
        args.output_directory,
        cosine_data,
        name="cosine",
        resolution=(
            f"Coarse diagnostic: macro n={args.macro_resolution}, "
            f"local r={args.local_refinement}, one constant trace per face"
        ),
        relative_flux_errors=tuple(
            item.flux_l2_error(darcy_flux, order=8) / (np.pi / np.sqrt(2)) for item in cosine
        ),
    )
    resolved = tuple(
        solve_darcy(
            mesh,
            source=darcy_source,
            dirichlet=darcy_pressure,
            formulation=formulation,
            skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 6) for _ in mesh.faces)),
            local_refinement=12,
            quadrature_order=6,
        )
        for formulation in ("primal", "mixed")
    )
    cosine_figures(
        args.output_directory,
        tuple(snapshot(solution, darcy_pressure, darcy_flux) for solution in resolved),
        name="cosine-resolved",
        resolution=(
            f"Enriched computation: macro n={args.macro_resolution}, "
            "local r=12, six constant trace segments per face"
        ),
        relative_flux_errors=tuple(
            item.flux_l2_error(darcy_flux, order=8) / (np.pi / np.sqrt(2)) for item in resolved
        ),
    )
    recovered = equilibrate_flux(cosine[0])
    balances = conservation_figures(args.output_directory, cosine, cosine_data, recovered)
    layered = tuple(
        solve_darcy(
            mesh,
            permeability=layered_permeability,
            dirichlet=layered_pressure,
            formulation=formulation,
            local_refinement=args.local_refinement,
            quadrature_order=6,
        )
        for formulation in ("primal", "mixed")
    )
    layer_data = tuple(snapshot(solution, layered_pressure, layered_flux) for solution in layered)
    layered_figures(args.output_directory, layered, layer_data)
    fine_spacing = 1 / (args.macro_resolution * args.local_refinement)
    projected_pressure_error = fine_spacing / 6 * np.sqrt(1 + 1e-6)
    metrics = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "solver": "scipy",
        "macro_resolution": args.macro_resolution,
        "macro_triangles": len(mesh.cells),
        "fine_triangles": sum(len(local.cells) for local in cosine[0].local_meshes),
        "local_refinement": args.local_refinement,
        "trace_degree": 0,
        "assembly_quadrature": 6,
        "error_quadrature": 8,
        "cosine": {
            solution.formulation: {
                "pressure_l2": solution.l2_error(darcy_pressure, order=8),
                "flux_l2": solution.flux_l2_error(darcy_flux, order=8),
            }
            for solution in cosine
        },
        "cosine_resolved": {
            "local_refinement": 12,
            "trace_segments": 6,
            "fine_triangles": sum(len(local.cells) for local in resolved[0].local_meshes),
            "errors": {
                item.formulation: {
                    "pressure_l2": item.l2_error(darcy_pressure, order=8),
                    "flux_l2": item.flux_l2_error(darcy_flux, order=8),
                    "flux_relative_l2": item.flux_l2_error(darcy_flux, order=8)
                    / (np.pi / np.sqrt(2)),
                }
                for item in resolved
            },
        },
        "layered": {
            solution.formulation: {
                "pressure_l2": solution.l2_error(layered_pressure, order=8),
                "flux_l2": solution.flux_l2_error(layered_flux, order=8),
            }
            for solution in layered
        },
        "conservation": balances,
        "layered_p0_projection_l2": projected_pressure_error,
    }
    np.testing.assert_allclose(
        metrics["layered"]["mixed"]["pressure_l2"],
        projected_pressure_error,
        rtol=1e-10,
        atol=1e-14,
        err_msg="mixed layered pressure must match the exact cell-average projection error",
    )
    (args.output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
