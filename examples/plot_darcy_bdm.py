"""Verify high-order conservative Darcy fluxes with analytical cosine data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from plot_mesh import draw_macro_mesh
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.mixed_bdm import BDMDarcySolution, solve_darcy_bdm
from pymhm.fem.hdiv.bdm import bdm2_evaluate
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/darcy-bdm"


def pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate p=cos(pi*x)cos(pi*y), with nonhomogeneous boundary trace."""
    return np.cos(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])


def flux(points: np.ndarray) -> np.ndarray:
    """Evaluate the physical flux -grad(p)."""
    x, y = np.pi * points.T
    return np.pi * np.column_stack((np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)))


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate div(q)=2*pi²*p."""
    return 2 * np.pi**2 * pressure(points)


def solve(resolution: int, degree: int, refinement: int = 2) -> BDMDarcySolution:
    """Solve the cosine case with independently selected macro trace degree."""
    mesh = TriangleMesh.unit_square(resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree) for _ in mesh.faces))
    return solve_darcy_bdm(
        mesh,
        source=source,
        dirichlet=pressure,
        skeleton=skeleton,
        local_refinement=refinement,
        quadrature_order=6,
    )


def record(solution: BDMDarcySolution) -> dict[str, float]:
    """Measure physical quadrature errors and all equilibrium moments."""
    q_error = solution.flux_l2_error(flux, 10)
    return {
        "pressure_l2": solution.l2_error(pressure, 10),
        "flux_l2": q_error,
        "flux_relative_l2": q_error / (np.pi / np.sqrt(2)),
        "divergence_l2": solution.divergence_l2_error(source, 10),
        "macro_conservation_linf": float(np.max(np.abs(solution.conservation_residuals()))),
        "fine_equilibrium_linf": float(
            max(np.max(np.abs(v)) for v in solution.fine_equilibrium_residuals())
        ),
        "normal_moment_linf": float(
            max(np.max(np.abs(v)) for v in solution.normal_flux_residuals())
        ),
    }


def save(figure: Any, name: str) -> None:
    """Save PNG and rasterized-field SVG versions of a scientific figure."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def field_plot(solution: BDMDarcySolution) -> None:
    """Preserve broken fields and show analytical, numerical and signed errors."""
    reference = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    samples = reference.submesh(0, 3)
    bary = np.column_stack((1 - samples.points.sum(axis=1), samples.points))
    coordinates, cells, fields = [], [], []
    offset = 0
    for mesh, p, q in zip(solution.local_meshes, solution.pressure, solution.flux, strict=True):
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        values, _ = bdm2_evaluate(mesh, q, bary)
        coordinates.append(points.reshape(-1, 2))
        fields.append(np.concatenate(((p @ bary.T)[..., None], values), axis=2).reshape(-1, 3))
        for cell in range(len(mesh.cells)):
            cells.append(samples.cells + offset + cell * len(bary))
        offset += len(mesh.cells) * len(bary)
    points, numerical = np.vstack(coordinates), np.vstack(fields)
    exact = np.column_stack((pressure(points), flux(points)))
    triangulation = mtri.Triangulation(*points.T, triangles=np.vstack(cells))
    figure, axes = plt.subplots(3, 3, figsize=(12.3, 11.2), layout="constrained")
    for row, name in enumerate(("Pressure p", "Flux q_x", "Flux q_y")):
        low, high = (
            min(exact[:, row].min(), numerical[:, row].min()),
            max(exact[:, row].max(), numerical[:, row].max()),
        )
        error = numerical[:, row] - exact[:, row]
        extent = max(float(np.max(np.abs(error))), 1e-15)
        for column, values in enumerate((exact[:, row], numerical[:, row], error)):
            artist = axes[row, column].tripcolor(
                triangulation,
                values,
                shading="gouraud",
                rasterized=True,
                cmap="RdBu_r" if column == 2 else "viridis",
                vmin=-extent if column == 2 else low,
                vmax=extent if column == 2 else high,
            )
            draw_macro_mesh(axes[row, column], solution.skeleton.mesh)
            axes[row, column].set(
                title=f"{name}: {('exact', 'pyMHM BDM2', 'pyMHM − exact')[column]}",
                xlabel="x",
                ylabel="y",
                aspect="equal",
            )
            figure.colorbar(artist, ax=axes[row, column], shrink=0.85, format="%.2g")
    metrics = record(solution)
    figure.suptitle(
        "Cosine pressure: 32 macros, local refinement 4, P2 macro traces\n"
        f"Quadrature L2 errors: p={metrics['pressure_l2']:.3e}, q={metrics['flux_l2']:.3e} "
        f"({100 * metrics['flux_relative_l2']:.3f}% relative), "
        f"div(q)={metrics['divergence_l2']:.3e}"
    )
    save(figure, "cosine-fields")


def study_plot(convergence: list[dict[str, Any]], trace_study: list[dict[str, Any]]) -> None:
    """Separate local polynomial accuracy, macro refinement and trace enrichment."""
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), layout="constrained")
    for degree, marker in ((1, "o"), (2, "s")):
        rows = [row for row in convergence if row["trace_degree"] == degree]
        h = 1 / np.array([row["macro_resolution"] for row in rows])
        axes[0].loglog(
            h, [row["pressure_l2"] for row in rows], marker=marker, label=f"P{degree} trace"
        )
        axes[1].loglog(h, [row["flux_l2"] for row in rows], marker=marker, label=f"P{degree} trace")
    for axis, name in zip(axes[:2], ("Pressure", "H(div) flux"), strict=True):
        axis.set(
            xlabel="Macro grid spacing 1/n",
            ylabel="Absolute L2 error",
            title=f"{name}: five macro mesh levels",
        )
        axis.invert_xaxis()
    axes[2].plot(
        [row["trace_degree"] for row in trace_study],
        [100 * row["flux_relative_l2"] for row in trace_study],
        "o-",
        label="BDM2 local flux",
    )
    axes[2].set(
        xlabel="Unsplit macro trace degree",
        ylabel="Relative physical flux L2 error (%)",
        title="Fixed 32 macros, local refinement 4",
        xticks=[0, 1, 2],
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, "convergence-and-trace")


def main() -> None:
    """Solve analytical studies and archive measured errors without external solvers."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    convergence, trace_study = [], []
    with threadpool_limits(1):
        for degree in (1, 2):
            for n in (1, 2, 4, 8, 16):
                solution = solve(n, degree)
                convergence.append(
                    {"macro_resolution": n, "trace_degree": degree, **record(solution)}
                )
                print("convergence", convergence[-1], flush=True)
        for degree in (0, 1, 2):
            solution = solve(4, degree, 4)
            trace_study.append({"trace_degree": degree, **record(solution)})
            print("trace", trace_study[-1], flush=True)
            if degree == 2:
                field_plot(solution)
    study_plot(convergence, trace_study)
    path = ROOT / "examples/results/darcy-bdm.json"
    path.write_text(
        json.dumps(
            {
                "method": "pyMHM BDM2/P1 Darcy with aligned unsplit macroface polynomial traces",
                "pressure": "cos(pi*x)cos(pi*y)",
                "permeability": "identity",
                "boundary": "analytical nonhomogeneous weak Dirichlet trace on all exterior faces",
                "mesh": "unit-square Cartesian southwest-diagonal triangles",
                "assembly_quadrature_order": 6,
                "error_quadrature_order": 10,
                "convergence_local_refinement": 2,
                "trace_study_local_refinement": 4,
                "convergence": convergence,
                "trace_study": trace_study,
                "scope": "Analytical high-order verification, not a reproduced published table.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
