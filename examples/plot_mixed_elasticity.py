"""Render original AFW/MHM verification and the oscillatory-modulus analytical case."""

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

from pymhm import MixedElasticitySolution, TriangleMesh, solve_elasticity_mixed
from pymhm.bdm import bdm2_evaluate

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/mixed-elasticity"
RESULTS = ROOT / "examples/results/mixed-elasticity.json"


def smooth_fields(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return solenoidal trigonometric displacement, stress, force and rotation."""
    x, y = np.pi * points.T
    displacement = np.column_stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))
    stress = np.zeros((len(points), 2, 2), dtype=np.result_type(points, float))
    stress[:, 0, 0] = 2 * np.pi * np.cos(x) * np.cos(y)
    stress[:, 1, 1] = -stress[:, 0, 0]
    return displacement, stress, 2 * np.pi**2 * displacement, -np.pi * np.sin(x) * np.sin(y)


def oscillatory_modulus(points: np.ndarray) -> np.ndarray:
    """Young modulus E=100[1+0.3 sin(10pi(x-0.5))cos(10pi y)]."""
    x, y = points.T
    return 100 * (1 + 0.3 * np.sin(10 * np.pi * (x - 0.5)) * np.cos(10 * np.pi * y))


def oscillatory_fields(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate the 2021 analytical displacement, including gradients of E.

    The displacement is (x²y² cos(6pi x)sin(7pi y)/27, exp(y)sin(4pi x)/5).
    Poisson ratio is 0.3. Boundary values are taken from this formula, which
    does not vanish on the entire boundary of the unit square.
    """
    x, y = points.T
    a, b, c, d = np.array([6, 7, 4, 10]) * np.pi
    xx, yy = x * x * np.cos(a * x), y * y * np.sin(b * y)
    dx = 2 * x * np.cos(a * x) - a * x * x * np.sin(a * x)
    dy = 2 * y * np.sin(b * y) + b * y * y * np.cos(b * y)
    dxx = 2 * np.cos(a * x) - 4 * a * x * np.sin(a * x) - a * a * x * x * np.cos(a * x)
    dyy = 2 * np.sin(b * y) + 4 * b * y * np.cos(b * y) - b * b * y * y * np.sin(b * y)
    second = np.exp(y) * np.sin(c * x) / 5
    second_dx = c * np.exp(y) * np.cos(c * x) / 5
    displacement = np.column_stack((xx * yy / 27, second))
    gradient = np.empty((len(points), 2, 2), dtype=np.result_type(points, float))
    gradient[:, 0, 0], gradient[:, 0, 1] = dx * yy / 27, xx * dy / 27
    gradient[:, 1, 0], gradient[:, 1, 1] = second_dx, second
    laplacian = np.column_stack(((dxx * yy + xx * dyy) / 27, (1 - c * c) * second))
    grad_div = np.column_stack((dxx * yy / 27 + second_dx, dx * dy / 27 + second))
    poisson = 0.3
    mu0 = 1 / (2 * (1 + poisson))
    lambda0 = poisson / ((1 + poisson) * (1 - 2 * poisson))
    strain_stress = mu0 * (gradient + gradient.swapaxes(1, 2))
    strain_stress += lambda0 * np.trace(gradient, axis1=1, axis2=2)[:, None, None] * np.eye(2)
    modulus = oscillatory_modulus(points)
    grad_modulus = (
        30
        * d
        * np.column_stack(
            (np.cos(d * (x - 0.5)) * np.cos(d * y), -np.sin(d * (x - 0.5)) * np.sin(d * y))
        )
    )
    stress = modulus[:, None, None] * strain_stress
    force = -np.einsum("nij,nj->ni", strain_stress, grad_modulus)
    force -= modulus[:, None] * (mu0 * laplacian + (lambda0 + mu0) * grad_div)
    rotation = (gradient[:, 0, 1] - gradient[:, 1, 0]) / 2
    return displacement, stress, force, rotation


def check_manufactured_data() -> None:
    """Check every force component independently by complex-step stress differentiation."""
    points = np.random.default_rng(46).uniform(0.02, 0.98, (17, 2))
    for fields in (smooth_fields, oscillatory_fields):
        divergence = np.zeros_like(points)
        for axis in range(2):
            shifted = points.astype(complex)
            shifted[:, axis] += 1e-30j
            divergence += fields(shifted)[1][:, :, axis].imag / 1e-30
        np.testing.assert_allclose(-divergence, fields(points)[2], atol=2e-10, rtol=3e-13)


def metrics(solution: MixedElasticitySolution, fields: Any) -> dict[str, float]:
    """Integrate errors independently of plot sampling and record physical defects."""
    return {
        "displacement_l2": solution.l2_error(lambda x: fields(x)[0], 10),
        "stress_l2": solution.stress_l2_error(lambda x: fields(x)[1], 10),
        "rotation_l2": solution.rotation_l2_error(lambda x: fields(x)[3], 10),
        "stress_divergence_l2": solution.divergence_l2_error(lambda x: -fields(x)[2], 10),
        "force_moment_linf": float(np.max(np.abs(solution.equilibrium_residuals()))),
        "fine_force_linf": float(max(np.max(np.abs(x)) for x in solution.fine_force_residuals())),
        "weak_symmetry_linf": float(
            max(np.max(np.abs(x)) for x in solution.weak_symmetry_residuals())
        ),
        "normal_traction_linf": float(
            max(np.max(np.abs(x)) for x in solution.normal_traction_residuals())
        ),
    }


def spatial_arrays(solution: MixedElasticitySolution) -> tuple[Any, np.ndarray, np.ndarray]:
    """Sample each broken fine triangle separately, preserving its one-sided values."""
    reference = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    samples = reference.submesh(0, 3)
    bary = np.column_stack((1 - samples.points.sum(axis=1), samples.points))
    coordinates, triangles, values = [], [], []
    offset = 0
    for mesh, stress, displacement, rotation in zip(
        solution.local_meshes,
        solution.stress,
        solution.displacement,
        solution.rotation,
        strict=True,
    ):
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        sigma, _ = bdm2_evaluate(mesh, stress, bary)
        u = np.einsum("qi,tia->tqa", bary, displacement)
        q = rotation @ bary.T
        coordinates.append(points.reshape(-1, 2))
        values.append(np.stack((u[..., 0], sigma[..., 0, 0], q), axis=-1).reshape(-1, 3))
        for cell in range(len(mesh.cells)):
            triangles.append(samples.cells + offset + cell * len(bary))
        offset += len(mesh.cells) * len(bary)
    points = np.vstack(coordinates)
    return mtri.Triangulation(*points.T, triangles=np.vstack(triangles)), points, np.vstack(values)


def save(figure: Any, name: str) -> None:
    """Save a vector-labelled SVG and a portable PNG for each scientific figure."""
    for suffix in ("svg", "png"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def field_figure(solution: MixedElasticitySolution, fields: Any, name: str, title: str) -> None:
    """Compare exact, numerical and signed errors on common exact/numerical scales."""
    triangulation, points, numerical = spatial_arrays(solution)
    u, sigma, _, q = fields(points)
    exact = np.column_stack((u[:, 0], sigma[:, 0, 0], q))
    figure, axes = plt.subplots(3, 3, figsize=(12.6, 11.4), layout="constrained")
    names = (r"$u_x$", r"$\sigma_{xx}$", r"$q$ (independent rotation)")
    for row, label in enumerate(names):
        low = min(float(exact[:, row].min()), float(numerical[:, row].min()))
        high = max(float(exact[:, row].max()), float(numerical[:, row].max()))
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
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=f"{label}: {('exact', 'pyMHM', 'pyMHM − exact')[column]}",
            )
            figure.colorbar(artist, ax=axes[row, column], shrink=0.85, format="%.2g")
    record = metrics(solution, fields)
    figure.suptitle(
        f"{title}\nQuadrature L2 errors: displacement={record['displacement_l2']:.3e}, "
        f"stress={record['stress_l2']:.3e}, rotation={record['rotation_l2']:.3e}"
    )
    save(figure, name)


def study_figure(convergence: list[dict[str, Any]], sweep: list[dict[str, Any]]) -> None:
    """Show five mesh levels and the fixed-space Lamé-ratio sweep."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    styles = (
        ("displacement_l2", "Displacement", "o"),
        ("stress_l2", "Cauchy stress", "s"),
        ("rotation_l2", "Rotation", "^"),
    )
    h = 1 / np.array([row["macro_resolution"] for row in convergence])
    positions = np.arange(len(sweep))
    labels = [
        "∞" if row["lame_lambda"] == "infinity" else f"{row['lame_lambda']:.0e}" for row in sweep
    ]
    for key, label, marker in styles:
        values = np.array([row[key] for row in convergence])
        axes[0].loglog(h, values, marker=marker, label=label)
        axes[1].semilogy(
            positions,
            [row[key] for row in sweep],
            marker=marker,
            label=label,
        )
    axes[0].set(
        xlabel="Macro grid spacing 1/n",
        ylabel="Absolute L2 error",
        title="BDM2/P1/P1: five mesh levels",
    )
    axes[0].invert_xaxis()
    axes[1].set(
        xlabel=r"Lamé ratio $\lambda/\mu$ (categorical)",
        ylabel="Absolute L2 error",
        title="Fixed mesh: 32 macros, local refinement 2",
    )
    for key, label in (
        ("force_moment_linf", "Macro force / moment"),
        ("fine_force_linf", "Fine P1 force moments"),
        ("weak_symmetry_linf", "P1 symmetry moments"),
    ):
        axes[2].semilogy(positions, [row[key] for row in sweep], "o-", label=label)
    axes[2].set(
        xlabel=r"Lamé ratio $\lambda/\mu$ (categorical)",
        ylabel="Maximum absolute moment defect",
        title="Physical constraints; no stress symmetrization",
    )
    for axis in axes[1:]:
        axis.set_xticks(positions, labels, rotation=45)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, "convergence-and-incompressibility")


def main() -> None:
    """Run exact-data studies, archive their metrics and render the gallery."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    check_manufactured_data()
    convergence, sweep = [], []
    with threadpool_limits(1):
        for n in (1, 2, 4, 8, 16):
            solution = solve_elasticity_mixed(
                TriangleMesh.unit_square(n),
                dirichlet=lambda x: smooth_fields(x)[0],
                source=lambda x: smooth_fields(x)[2],
                quadrature_order=6,
            )
            convergence.append({"macro_resolution": n, **metrics(solution, smooth_fields)})
            print("smooth", convergence[-1], flush=True)
            if n == 8:
                field_figure(
                    solution,
                    smooth_fields,
                    "smooth-fields",
                    "Mixed elasticity: exact solenoidal field",
                )
        for lam in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e8, np.inf):
            solution = solve_elasticity_mixed(
                TriangleMesh.unit_square(4),
                dirichlet=lambda x: smooth_fields(x)[0],
                source=lambda x: smooth_fields(x)[2],
                lame_lambda=lam,
                quadrature_order=6,
            )
            sweep.append(
                {
                    "lame_lambda": "infinity" if np.isinf(lam) else lam,
                    "poisson_ratio": 0.5 if np.isinf(lam) else lam / (2 * (lam + 1)),
                    **metrics(solution, smooth_fields),
                }
            )
        oscillatory = solve_elasticity_mixed(
            TriangleMesh.unit_square(16),
            dirichlet=lambda x: oscillatory_fields(x)[0],
            source=lambda x: oscillatory_fields(x)[2],
            lame_mu=lambda x: oscillatory_modulus(x) / 2.6,
            lame_lambda=lambda x: oscillatory_modulus(x) * 0.3 / (1.3 * 0.4),
            quadrature_order=8,
        )
        oscillatory_record = {"macro_resolution": 16, **metrics(oscillatory, oscillatory_fields)}
        print("oscillatory", oscillatory_record, flush=True)
        field_figure(
            oscillatory,
            oscillatory_fields,
            "oscillatory-fields",
            "2021 analytical data: oscillatory E, Poisson ratio 0.3",
        )
    study_figure(convergence, sweep)
    RESULTS.write_text(
        json.dumps(
            {
                "method": "pyMHM BDM2 stress rows / discontinuous P1 displacement and rotation",
                "mesh": "unit-square Cartesian southwest-diagonal macro triangulation",
                "local_refinement": 2,
                "interior_trace_degree": 1,
                "exterior_trace": "P2 on each fine boundary face",
                "error_quadrature_order": 10,
                "convergence": convergence,
                "lame_sweep": sweep,
                "oscillatory_2021_analytical_data": oscillatory_record,
                "scope": (
                    "Own mesh study; the oscillatory case uses published analytical data, "
                    "not reproduced table ordinates."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
