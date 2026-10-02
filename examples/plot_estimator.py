"""Measure the unit-diffusion MHM estimator and its conforming potential recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from plot_mesh import draw_macro_mesh
from plot_reconstruction_moments import flux, pressure, source
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.estimator import DarcyEstimator, estimate_darcy_error
from pymhm.lagrange import tabulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/results"
FIGURES = ROOT / "docs/figures/estimator"


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Convert the analytical physical flux into the pressure gradient."""
    return -flux(points)


def solve(resolution: int, trace_degree: int) -> DarcyEstimator:
    """Assemble unit diffusion with k=ell+2, m=2 and conforming submeshes."""
    mesh = TriangleMesh.unit_square(resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(trace_degree) for _ in mesh.faces))
    primal = solve_darcy(
        mesh,
        degree=trace_degree + 2,
        skeleton=skeleton,
        local_refinement=2,
        source=source,
        quadrature_order=10,
    )
    return estimate_darcy_error(primal, homogeneous_dirichlet=True, degree=2, quadrature_order=10)


def record(result: DarcyEstimator, resolution: int, trace_degree: int) -> dict[str, Any]:
    """Record estimator components and independently integrated energy errors."""
    error = result.energy_error(exact_gradient, order=12)
    if result.total < error:
        raise RuntimeError("the measured unit-diffusion estimator is below the energy error")
    return {
        "macro_resolution": resolution,
        "macro_triangles": 2 * resolution**2,
        "macro_diameter": np.sqrt(2) / resolution,
        "local_refinement": 2,
        "local_degree": trace_degree + 2,
        "trace_degree": trace_degree,
        "rt_degree": 2,
        "assembly_order": 10,
        "estimator_order": 10,
        "error_order": 12,
        "energy_error": error,
        "estimator": result.total,
        "effectivity": result.total / error,
        "eta_1": float(np.linalg.norm(result.flux_defect)),
        "eta_2": float(np.linalg.norm(result.nonconformity)),
        "eta_3": float(np.linalg.norm(result.divergence_defect)),
        "eta_osc": float(np.linalg.norm(result.oscillation)),
        "equilibrium_moment_l2_max": float(np.max(result.equilibrium_defect)),
        "primal_pressure_l2": result.solution.l2_error(pressure, order=12),
        "conforming_pressure_l2": result.potential.l2_error(pressure, order=12),
    }


def sample(result: DarcyEstimator) -> dict[str, np.ndarray]:
    """Sample each broken fine element without averaging its display values."""
    reference = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    display = reference.submesh(0, 4)
    bary = np.column_stack((1 - display.points.sum(axis=1), display.points))
    coordinates, cells, primal_values, recovered_values = [], [], [], []
    offset = 0
    solution = result.solution
    for mesh, primal, recovered in zip(
        solution.local_meshes, solution.pressure, result.potential.local_values, strict=True
    ):
        dofs, _, basis, _, _ = tabulate(mesh, solution.degree, bary)
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        coordinates.append(points.reshape(-1, 2))
        primal_values.append((primal[dofs] @ basis.T).ravel())
        recovered_values.append((recovered[dofs] @ basis.T).ravel())
        cells.extend(display.cells + offset + i * len(bary) for i in range(len(mesh.cells)))
        offset += len(mesh.cells) * len(bary)
    points = np.concatenate(coordinates)
    return {
        "points": points,
        "cells": np.concatenate(cells),
        "exact": pressure(points),
        "primal": np.concatenate(primal_values),
        "conforming": np.concatenate(recovered_values),
        "macro_points": solution.skeleton.mesh.points,
        "macro_cells": solution.skeleton.mesh.cells,
        "eta_1": result.flux_defect,
        "eta_2": result.nonconformity,
        "eta_3": result.divergence_defect,
        "eta_osc": result.oscillation,
    }


def save(figure: Any, name: str) -> None:
    """Export scientific PNG and SVG figures with vector text and macro boundaries."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def convergence(rows: list[dict[str, Any]]) -> None:
    """Display actual energy errors, estimator effectivity and all four terms."""
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    for ell, color in ((0, "#27648d"), (1, "#ad4c2d")):
        data = [row for row in rows if row["trace_degree"] == ell]
        h = [row["macro_diameter"] for row in data]
        axes[0].loglog(
            h,
            [row["energy_error"] for row in data],
            "o-",
            color=color,
            label=f"Energy error, k={ell + 2}, trace P{ell}",
        )
        axes[0].loglog(
            h,
            [row["estimator"] for row in data],
            "s--",
            color=color,
            label=f"Estimator, k={ell + 2}, trace P{ell}",
        )
        axes[1].semilogx(
            h,
            [row["effectivity"] for row in data],
            "o-",
            color=color,
            label=f"k={ell + 2}, trace P{ell}",
        )
    axes[1].axhline(1.0, color="black", linestyle=":", label="Estimator = error")
    axes[0].set(title="Energy error and estimator", ylabel="Absolute norm")
    axes[1].set(title="Measured effectivity", ylabel="Estimator / energy error")
    for axis in axes:
        axis.set_xlabel("Macro diameter H")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, "convergence")
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    for ell, axis in enumerate(axes):
        data = [row for row in rows if row["trace_degree"] == ell]
        for name, label in (
            ("eta_1", "Flux defect η1"),
            ("eta_2", "Nonconformity η2"),
            ("eta_3", "Divergence defect η3"),
            ("eta_osc", "Source oscillation"),
        ):
            axis.loglog(
                [row["macro_diameter"] for row in data],
                [row[name] for row in data],
                "o-",
                label=label,
            )
        axis.set(
            xlabel="Macro diameter H",
            ylabel="Global L2 sum of local terms",
            title=f"P{ell + 2} pressure, P{ell} trace, RT2 recovery",
        )
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, "components")


def fields(data: Any) -> None:
    """Compare exact, broken and conforming pressures, with separate error scales."""
    macro = TriangleMesh(data["macro_points"], data["macro_cells"])
    triangulation = mtri.Triangulation(*data["points"].T, triangles=data["cells"])
    exact, primal, recovered = (data[name] for name in ("exact", "primal", "conforming"))
    low, high = (
        min(v.min() for v in (exact, primal, recovered)),
        max(v.max() for v in (exact, primal, recovered)),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.6), layout="constrained")
    values = (exact, primal, recovered, primal - exact, recovered - exact, recovered - primal)
    titles = (
        "Analytical pressure",
        "Broken P3 pressure",
        "Oswald conforming pressure",
        "Broken − analytical",
        "Oswald − analytical",
        "Oswald − broken",
    )
    for index, (axis, value, title) in enumerate(zip(axes.ravel(), values, titles, strict=True)):
        limit = max(float(np.max(abs(value))), 1e-15)
        artist = axis.tripcolor(
            triangulation,
            value,
            shading="gouraud",
            rasterized=True,
            cmap="RdBu_r" if index >= 3 else "viridis",
            vmin=-limit if index >= 3 else low,
            vmax=limit if index >= 3 else high,
        )
        draw_macro_mesh(axis, macro)
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        figure.colorbar(artist, ax=axis, shrink=0.8, format="%.2g")
    figure.suptitle(
        "32 macrotriangles, four fine triangles per macro, P1 trace\n"
        "Averaging defines a separate recovered field; original broken values are preserved"
    )
    save(figure, "potential")
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.6), layout="constrained")
    tri = mtri.Triangulation(*macro.points.T, triangles=macro.cells)
    for axis, name, title in zip(
        axes,
        ("eta_1", "eta_2", "eta_3", "eta_osc"),
        ("Flux defect η1,K", "Nonconformity η2,K", "Divergence defect η3,K", "Oscillation ηosc,K"),
        strict=True,
    ):
        artist = axis.tripcolor(tri, facecolors=data[name], shading="flat", cmap="viridis")
        draw_macro_mesh(axis, macro)
        axis.set(title=title, aspect="equal", xlabel="x", ylabel="y")
        figure.colorbar(artist, ax=axis, shrink=0.75, format="%.2g")
    save(figure, "local-indicators")


def main() -> None:
    """Solve five mesh levels or redraw archived norms and sampled fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.reuse_results:
        rows = []
        with threadpool_limits(1):
            for ell in (0, 1):
                for n in (1, 2, 4, 8, 16):
                    result = solve(n, ell)
                    row = record(result, n, ell)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if ell == 1 and n == 4:
                        np.savez_compressed(RESULTS / "estimator-fields.npz", **sample(result))
        (RESULTS / "estimator.json").write_text(
            json.dumps(
                {
                    "reference": "10.1137/24M1673073",
                    "interpretation": (
                        "Analytical case with declared meshes; no published-table equality claim"
                    ),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
    records = json.loads((RESULTS / "estimator.json").read_text())
    convergence(records["rows"])
    with np.load(RESULTS / "estimator-fields.npz") as data:
        fields(data)


if __name__ == "__main__":
    main()
