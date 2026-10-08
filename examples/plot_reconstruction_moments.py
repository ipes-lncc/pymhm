"""Measure the RT moment reconstruction with the smooth data of MMS 2026."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import darcy as solve_darcy
from examples.plot_mesh import draw_macro_mesh
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.postprocessing.solutions import DarcySolution
from pymhm.recovery.moments import MomentFluxSolution, reconstruct_darcy_moments

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/reconstruction-moments"


def pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate the smooth two-period sine pressure from section 6.1."""
    x, y = 2 * np.pi * points.T
    return np.sin(x) * np.sin(y)


def flux(points: np.ndarray) -> np.ndarray:
    """Evaluate the physical analytical flux -grad(p)."""
    x, y = 2 * np.pi * points.T
    return -2 * np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate div(q)=8*pi²*p."""
    return 8 * np.pi**2 * pressure(points)


def solve(resolution: int, trace_degree: int) -> DarcySolution:
    """Use primal k=ell+2, h=H/2 and the unchanged homogeneous pressure boundary."""
    mesh = TriangleMesh.unit_square(resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(trace_degree) for _ in mesh.faces))
    return solve_darcy(
        mesh,
        source=source,
        skeleton=skeleton,
        degree=trace_degree + 2,
        local_refinement=2,
        quadrature_order=8,
    )


def raw_values(
    solution: DarcySolution, macrocell: int, bary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the broken primal flux and its actual polynomial divergence."""
    mesh = solution.local_meshes[macrocell]
    dofs, _, _, gradients, hessians = tabulate(mesh, solution.degree, bary)
    coefficients = solution.pressure[macrocell][dofs]
    values = -np.einsum("ti,tqia->tqa", coefficients, gradients)
    divergence = -np.einsum("ti,tqiaa->tq", coefficients, hessians)
    return values, divergence


def record(solution: DarcySolution, recovered: MomentFluxSolution) -> dict[str, float]:
    """Integrate physical errors separately from every conservation diagnostic."""
    bary, weights = triangle_quadrature(10)
    squared = 0.0
    for cell, mesh in enumerate(solution.local_meshes):
        _, divergence = raw_values(solution, cell, bary)
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        difference = divergence - source(points.reshape(-1, 2)).reshape(divergence.shape)
        squared += float(mesh.areas @ (difference**2 @ weights))
    return {
        "pressure_l2": solution.l2_error(pressure, 10),
        "raw_flux_l2": solution.flux_l2_error(flux, 10),
        "reconstructed_flux_l2": recovered.flux_l2_error(flux, 10),
        "raw_primal_divergence_l2": float(np.sqrt(squared)),
        "reconstructed_divergence_l2": recovered.divergence_l2_error(source, 10),
        "projected_divergence_l2": recovered.projected_divergence_l2_error(source, 10),
        "continuous_moment_linf": float(
            max(np.max(np.abs(v)) for v in recovered.continuous_moment_residuals())
        ),
        "fine_integrated_defect_linf": float(
            max(np.max(np.abs(v)) for v in recovered.fine_conservation_residuals())
        ),
        "macro_conservation_linf": float(np.max(np.abs(recovered.conservation_residuals()))),
        "boundary_moment_linf": float(
            max(np.max(np.abs(v)) for v in recovered.normal_flux_residuals())
        ),
    }


def save(figure: Any, name: str) -> None:
    """Save compact rasterized-field SVG and PNG versions of a figure."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def field_plots(solution: DarcySolution, recovered: MomentFluxSolution) -> None:
    """Display broken one-sided fields with all actual macro boundaries overlaid."""
    reference = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    samples = reference.submesh(0, 4)
    bary = np.column_stack((1 - samples.points.sum(axis=1), samples.points))
    coordinates, cells, raw, reconstruction, divergence_fields = [], [], [], [], []
    projected = recovered.continuous_divergence_projection()
    offset = 0
    for cell, mesh in enumerate(solution.local_meshes):
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        q_raw, div_raw = raw_values(solution, cell, bary)
        q_rt, div_rt = rt_evaluate(mesh, recovered.flux[cell], recovered.degree, bary)
        dofs, _, basis, _, _ = tabulate(mesh, recovered.degree, bary)
        div_projected = projected[cell][dofs] @ basis.T
        coordinates.append(points.reshape(-1, 2))
        raw.append(q_raw.reshape(-1, 2))
        reconstruction.append(q_rt.reshape(-1, 2))
        divergence_fields.append(np.stack((div_raw, div_rt, div_projected), axis=-1).reshape(-1, 3))
        for fine_cell in range(len(mesh.cells)):
            cells.append(samples.cells + offset + fine_cell * len(bary))
        offset += len(mesh.cells) * len(bary)
    points = np.vstack(coordinates)
    raw_q, rt_q, divergence = (
        np.vstack(raw),
        np.vstack(reconstruction),
        np.vstack(divergence_fields),
    )
    exact_q, exact_f = flux(points), source(points)
    triangulation = mtri.Triangulation(*points.T, triangles=np.vstack(cells))
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), layout="constrained")
    for row in range(2):
        data = (exact_q[:, row], raw_q[:, row], rt_q[:, row], rt_q[:, row] - exact_q[:, row])
        low, high = min(v.min() for v in data[:3]), max(v.max() for v in data[:3])
        for column, values in enumerate(data):
            error = column == 3
            limit = max(float(np.max(np.abs(values))), 1e-15)
            artist = axes[row, column].tripcolor(
                triangulation,
                values,
                shading="gouraud",
                rasterized=True,
                cmap="RdBu_r" if error else "viridis",
                vmin=-limit if error else low,
                vmax=limit if error else high,
            )
            draw_macro_mesh(axes[row, column], solution.skeleton.mesh)
            axes[row, column].set(
                title=(
                    f"q_{'xy'[row]}: "
                    + ("exact", "pyMHM raw P3", "pyMHM RT2", "RT2 − exact")[column]
                ),
                aspect="equal",
                xlabel="x",
                ylabel="y",
            )
            figure.colorbar(artist, ax=axes[row, column], shrink=0.8, format="%.2g")
    metrics = record(solution, recovered)
    figure.suptitle(
        "Moment reconstruction: 32 macros, four fine triangles/macro, P1 skeleton\n"
        f"Physical flux L2 error: raw={metrics['raw_flux_l2']:.3e}, "
        f"RT2={metrics['reconstructed_flux_l2']:.3e}"
    )
    save(figure, "flux-fields")
    figure, axes = plt.subplots(3, 3, figsize=(12.5, 11.3), layout="constrained")
    for row, name in enumerate(
        ("raw primal divergence", "RT2 divergence", "continuous P2 projection")
    ):
        numerical = divergence[:, row]
        error = numerical - exact_f
        low, high = min(exact_f.min(), numerical.min()), max(exact_f.max(), numerical.max())
        extent = max(float(np.max(np.abs(error))), 1e-15)
        for column, values in enumerate((exact_f, numerical, error)):
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
                title=("Exact source f", "pyMHM: " + name, "Numerical − f")[column],
                aspect="equal",
                xlabel="x",
                ylabel="y",
            )
            figure.colorbar(artist, ax=axes[row, column], shrink=0.8, format="%.2g")
    figure.suptitle(
        "Different quantities: raw divergence and its continuous macro-local projection\n"
        f"L2 errors: primal={metrics['raw_primal_divergence_l2']:.3e}, "
        f"RT2={metrics['reconstructed_divergence_l2']:.3e}, "
        f"projected={metrics['projected_divergence_l2']:.3e}"
    )
    save(figure, "divergence-fields")


def study_plot(rows: list[dict[str, Any]]) -> None:
    """Keep raw/projected divergence and fine/continuous balance on separate axes."""
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), layout="constrained")
    for ell, color in ((0, "tab:blue"), (1, "tab:orange")):
        selected = [row for row in rows if row["trace_degree"] == ell]
        h = 1 / np.array([row["macro_resolution"] for row in selected])
        label = f"ell={ell}, k={ell + 2}"
        for key, style, method in (
            ("raw_flux_l2", "o--", "raw"),
            ("reconstructed_flux_l2", "s-", "RT2"),
        ):
            axes[0, 0].loglog(
                h, [row[key] for row in selected], style, color=color, label=f"{label}, {method}"
            )
        for key, style, method in (
            ("raw_primal_divergence_l2", "o--", "primal"),
            ("reconstructed_divergence_l2", "s-", "RT2"),
        ):
            axes[0, 1].loglog(
                h, [row[key] for row in selected], style, color=color, label=f"{label}, {method}"
            )
        axes[1, 0].loglog(
            h, [row["projected_divergence_l2"] for row in selected], "o-", color=color, label=label
        )
        axes[1, 1].loglog(
            h,
            [row["fine_integrated_defect_linf"] for row in selected],
            "s-",
            color=color,
            label=f"{label}, fine DG0",
        )
        axes[1, 1].loglog(
            h,
            [max(row["continuous_moment_linf"], 1e-17) for row in selected],
            "o--",
            color=color,
            label=f"{label}, C0 P2",
        )
    titles = (
        "Physical flux L2 error",
        "Raw divergence L2 error",
        "Projected divergence L2 error",
        "Integrated balance defects",
    )
    for axis, title in zip(axes.ravel(), titles, strict=True):
        axis.set(title=title, xlabel="Macro grid spacing 1/n", ylabel="Absolute error / defect")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("pyMHM analytical study: five meshes, RT2 reconstruction, h=H/2")
    save(figure, "convergence-and-conservation")


def main() -> None:
    """Solve original analytical studies; no external solver or digitized curves are used."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(1):
        for ell in (0, 1):
            for resolution in (1, 2, 4, 8, 16):
                solution = solve(resolution, ell)
                recovered = reconstruct_darcy_moments(solution, degree=2)
                rows.append(
                    {
                        "macro_resolution": resolution,
                        "trace_degree": ell,
                        "primal_degree": ell + 2,
                        **record(solution, recovered),
                    }
                )
                print(rows[-1], flush=True)
                if ell == 1 and resolution == 4:
                    field_plots(solution, recovered)
    study_plot(rows)
    (ROOT / "examples/results/reconstruction-moments.json").write_text(
        json.dumps(
            {
                "case": "p=sin(2*pi*x)sin(2*pi*y), K=I, homogeneous Dirichlet",
                "reference": (
                    "Barrenechea et al., MMS 2026, DOI 10.1137/24M1673073, "
                    "section 6.1 analytical data"
                ),
                "scope": (
                    "Original pyMHM analytical study; no published ordinates or FreeFem++ execution"
                ),
                "reconstruction": "RT2 canonical face/volume moments, equations 4.9 and 5.1",
                "local_refinement": 2,
                "assembly_quadrature_order": 8,
                "error_quadrature_order": 10,
                "convergence": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
