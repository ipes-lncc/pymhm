"""Plot exact and reconstructed Stokes, Brinkman, and Oseen verification fields.

Run ``pixi run -e notebooks python examples/plot_flow_cases.py``. The script
writes standalone PNG/SVG figures and unrounded quantitative results. Analytical
fields follow the 2017 Stokes example; Brinkman and Oseen forces are derived from
those fields. These meshes do not reproduce a published numerical table.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_segment_profile
from examples.formulations.application import brinkman as solve_brinkman
from examples.manufactured import stokes_pressure, stokes_source, stokes_velocity
from examples.plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.fem.scalar.operators import p1_geometry
from pymhm.postprocessing.solutions import VectorSolution

FloatArray = NDArray[np.float64]
LABELS = {"taylor-hood": "Taylor–Hood P2/P1", "usfem": "USFEM P1/P1"}
PROFILE_COLORS = {"taylor-hood": "#0072B2", "usfem": "#D55E00"}


@dataclass(frozen=True)
class Samples:
    """Elementwise plot samples that preserve jumps between macroelements."""

    points: FloatArray
    triangles: NDArray[np.int64]
    velocity: FloatArray
    pressure: FloatArray
    divergence: FloatArray

    def triangulation(self) -> Triangulation:
        """Return the explicit disconnected triangulation for sampled fields."""
        return Triangulation(self.points[:, 0], self.points[:, 1], self.triangles)


def basis_values(bary: FloatArray, degree: int) -> FloatArray:
    """Evaluate nodal triangle bases in vertex then edge (01,12,20) order."""
    if degree == 1:
        return bary
    return np.column_stack(
        (
            bary * (2 * bary - 1),
            4 * bary[:, 0] * bary[:, 1],
            4 * bary[:, 1] * bary[:, 2],
            4 * bary[:, 2] * bary[:, 0],
        )
    )


def sample_solution(solution: VectorSolution) -> Samples:
    """Evaluate true P1/P2 polynomials on three subdivisions per microtriangle."""
    reference = TriangleMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]]).submesh(0, 3)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    basis = basis_values(bary, solution.degree)
    points, triangles, velocity, pressure, divergence = [], [], [], [], []
    offset = 0
    for mesh, values, local_pressure in zip(
        solution.local_meshes, solution.values, solution.pressure, strict=True
    ):
        dofs = mesh.cells
        gradients, _ = p1_geometry(mesh)
        derivative = np.broadcast_to(gradients[:, None], (len(mesh.cells), len(bary), 3, 2))
        if solution.degree == 2:
            dofs = np.column_stack((mesh.cells, len(mesh.points) + mesh.cell_faces))
            derivative = np.empty((len(mesh.cells), len(bary), 6, 2))
            derivative[:, :, :3] = (4 * bary[None, :, :, None] - 1) * gradients[:, None]
            for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
                derivative[:, :, 3 + k] = 4 * (
                    bary[None, :, i, None] * gradients[:, None, j]
                    + bary[None, :, j, None] * gradients[:, None, i]
                )
        xyz = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        points.append(xyz.reshape(-1, 2))
        local_triangles = reference.cells[None] + (
            np.arange(len(mesh.cells))[:, None, None] * len(bary) + offset
        )
        triangles.append(local_triangles.reshape(-1, 3))
        velocity.append(np.einsum("qi,tia->tqa", basis, values[dofs]).reshape(-1, 2))
        pressure.append((local_pressure[mesh.cells] @ bary.T).ravel())
        divergence.append(np.einsum("tqia,tia->tq", derivative, values[dofs]).ravel())
        offset += len(mesh.cells) * len(bary)
    return Samples(
        *(np.concatenate(values) for values in (points, triangles, velocity, pressure, divergence))
    )


def save_figure(figure: Figure, directory: Path, name: str) -> None:
    """Save editable vector artwork and a matching raster preview."""
    figure.savefig(directory / f"{name}.svg")
    figure.savefig(directory / f"{name}.png", dpi=170)
    plt.close(figure)


def comparison(
    sampled: dict[str, Samples],
    directory: Path,
    name: str,
    title: str,
    field: str,
    macro_mesh: TriangleMesh,
) -> None:
    """Plot exact and numerical velocity magnitude or pressure with one common color range."""
    first = next(iter(sampled.values()))
    exact = (
        np.linalg.norm(stokes_velocity(first.points), axis=1)
        if field == "velocity_magnitude"
        else stokes_pressure(first.points)
    )
    fields = {"Exact": (first, exact)}
    for method, samples in sampled.items():
        values = (
            np.linalg.norm(samples.velocity, axis=1)
            if field == "velocity_magnitude"
            else samples.pressure
        )
        fields[LABELS[method]] = samples, values
    minimum = min(float(values.min()) for _, values in fields.values())
    maximum = max(float(values.max()) for _, values in fields.values())
    if field == "pressure":
        maximum = max(abs(minimum), abs(maximum))
        minimum = -maximum
    levels = np.linspace(minimum, maximum, 29)
    figure, axes = plt.subplots(
        1, len(fields), figsize=(4 * len(fields), 3.9), constrained_layout=True, squeeze=False
    )
    for axis, (label, (samples, values)) in zip(axes[0], fields.items(), strict=True):
        artist = axis.tricontourf(
            samples.triangulation(),
            values,
            levels=levels,
            cmap="viridis" if field == "velocity_magnitude" else "RdBu_r",
        )
        if field == "velocity_magnitude":
            grid_x, grid_y = np.meshgrid(np.linspace(0.08, 0.92, 7), np.linspace(0.08, 0.92, 7))
            targets = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            nearest = np.argmin(
                np.sum((samples.points[:, None] - targets[None]) ** 2, axis=2), axis=0
            )
            arrow_points = samples.points[nearest]
            vectors = (
                stokes_velocity(arrow_points) if label == "Exact" else samples.velocity[nearest]
            )
            axis.quiver(
                *arrow_points.T,
                *vectors.T,
                angles="xy",
                scale_units="xy",
                scale=20,
                color="white",
                edgecolor="#222222",
                linewidth=0.3,
                width=0.004,
            )
        axis.set(title=label, xlabel="x", ylabel="y", aspect="equal", xlim=(0, 1), ylim=(0, 1))
        macro_artist = draw_macro_mesh(axis, macro_mesh)
    figure.legend(
        [macro_artist], ["Macro mesh"], loc="outside lower center", frameon=False, fontsize=8
    )
    figure.colorbar(artist, ax=axes[0], label=r"$|u|$" if field == "velocity_magnitude" else "$p$")
    figure.suptitle(title)
    save_figure(figure, directory, name)


def diagnostics(
    sampled: dict[str, Samples],
    directory: Path,
    name: str,
    title: str,
    metrics: dict[str, dict[str, float]],
    macro_mesh: TriangleMesh,
) -> None:
    """Show field errors and signed divergence on method-independent row scales."""
    errors = {
        method: np.linalg.norm(samples.velocity - stokes_velocity(samples.points), axis=1)
        for method, samples in sampled.items()
    }
    error_max = max(float(values.max()) for values in errors.values())
    pressure_errors = {
        method: np.abs(samples.pressure - stokes_pressure(samples.points))
        for method, samples in sampled.items()
    }
    pressure_max = max(float(values.max()) for values in pressure_errors.values())
    div_max = max(float(np.max(np.abs(samples.divergence))) for samples in sampled.values())
    figure, axes = plt.subplots(
        3, len(sampled), figsize=(4.5 * len(sampled), 10.0), constrained_layout=True, squeeze=False
    )
    for column, (method, samples) in enumerate(sampled.items()):
        error_artist = axes[0, column].tricontourf(
            samples.triangulation(),
            errors[method],
            levels=np.linspace(0, error_max, 25),
            cmap="magma",
        )
        pressure_artist = axes[1, column].tricontourf(
            samples.triangulation(),
            pressure_errors[method],
            levels=np.linspace(0, pressure_max, 25),
            cmap="magma",
        )
        div_artist = axes[2, column].tricontourf(
            samples.triangulation(),
            samples.divergence,
            levels=np.linspace(-div_max, div_max, 25),
            cmap="RdBu_r",
        )
        axes[0, column].set_title(
            f"{LABELS[method]}\n"
            + r"$\|u_h-u\|_{L^2}$"
            + f" = {metrics[method]['velocity_l2']:.3e}"
        )
        axes[1, column].set_title(r"$\|p_h-p\|_{L^2}$" + f" = {metrics[method]['pressure_l2']:.3e}")
        axes[2, column].set_title(
            r"$\|\nabla\!\cdot u_h\|_{L^2}$" + f" = {metrics[method]['divergence_l2']:.3e}"
        )
        for axis in axes[:, column]:
            axis.set(xlabel="x", ylabel="y", aspect="equal", xlim=(0, 1), ylim=(0, 1))
            macro_artist = draw_macro_mesh(axis, macro_mesh)
    figure.legend(
        [macro_artist], ["Macro mesh"], loc="outside lower center", frameon=False, fontsize=8
    )
    figure.colorbar(error_artist, ax=axes[0], label=r"$|u_h-u|$")
    figure.colorbar(pressure_artist, ax=axes[1], label=r"$|p_h-p|$")
    figure.colorbar(div_artist, ax=axes[2], label=r"$\nabla\!\cdot u_h$ (exact: 0)")
    figure.suptitle(
        title.replace(", β", "\nβ").replace(" — ", "\n") if len(sampled) == 1 else title
    )
    save_figure(figure, directory, name)


def profiles(solutions: dict[str, VectorSolution], directory: Path, name: str, title: str) -> None:
    """Compare cross sections away from lines aligned with the macro grid."""
    coordinate = np.linspace(0, 1, 301)
    vertical = np.column_stack((np.full_like(coordinate, 0.43), coordinate))
    horizontal = np.column_stack((coordinate, np.full_like(coordinate, 0.37)))
    exact = (
        stokes_velocity(vertical)[:, 0],
        stokes_velocity(horizontal)[:, 1],
        stokes_pressure(horizontal),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    macro_mesh = next(iter(solutions.values())).skeleton.mesh
    for axis, values in zip(axes, exact, strict=True):
        axis.plot(coordinate, values, color="black", linewidth=2, label="Exact")
    for method, solution in solutions.items():
        for axis, fields, degree, line, component, coordinate_axis in zip(
            axes,
            (solution.values, solution.values, solution.pressure),
            (solution.degree, solution.degree, solution.pressure_degree),
            (vertical, horizontal, horizontal),
            (0, 1, None),
            (1, 0, 0),
            strict=True,
        ):
            sampled = sample_segment_profile(solution, fields, degree, line[0], line[-1])
            for index, (points, values) in enumerate(
                zip(sampled["profile_points"], sampled["profile_values"], strict=True)
            ):
                axis.plot(
                    points[:, coordinate_axis],
                    values if component is None else values[:, component],
                    "--",
                    color=PROFILE_COLORS[method],
                    linewidth=1.5,
                    label=LABELS[method] if index == 0 else "_nolegend_",
                )
    for axis, title_, xlabel, ylabel, segment in zip(
        axes,
        ("x = 0.43", "y = 0.37", "y = 0.37"),
        ("y", "x", "x"),
        (r"$u_x$", r"$u_y$", "$p$"),
        (
            (vertical[0], vertical[-1]),
            (horizontal[0], horizontal[-1]),
            (horizontal[0], horizontal[-1]),
        ),
        strict=True,
    ):
        mark_macro_interfaces(axis, macro_profile_breaks(macro_mesh, *segment)[1:-1], label=True)
        axis.set(title=title_, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncols=len(labels), frameon=False, fontsize=8
    )
    figure.suptitle(title)
    save_figure(figure, directory, name)


def polynomial_transport(points: FloatArray, beta: tuple[float, float]) -> FloatArray:
    """Differentiate the manufactured velocity to evaluate (beta dot grad) u."""
    a = Polynomial([0, 0, 1, -2, 1])
    x, y = points.T
    dx = 128 * np.column_stack((-a.deriv()(x) * a.deriv()(y), a.deriv(2)(x) * a(y)))
    dy = 128 * np.column_stack((-a(x) * a.deriv(2)(y), a.deriv()(x) * a.deriv()(y)))
    return beta[0] * dx + beta[1] * dy


def rotation(points: FloatArray) -> FloatArray:
    """Return the exactly representable divergence-free affine Oseen patch."""
    return np.column_stack((points[:, 1], -points[:, 0]))


def main() -> None:
    """Solve the gallery, check its analytical expectations, and write all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-resolution", type=int, default=4)
    parser.add_argument("--local-refinement", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("docs/figures/flow"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    mesh = TriangleMesh.unit_square(args.macro_resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2)
    records: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "macro_resolution": args.macro_resolution,
        "macro_triangles": len(mesh.cells),
        "trace_dofs": skeleton.size,
        "local_refinement": args.local_refinement,
        "trace_degree": 1,
        "assembly_quadrature": 6,
        "error_quadrature": 8,
        "evidence": "Exact-field verification; no published mesh or table reproduction claimed",
    }
    for case, viscosity, drag, beta, methods in (
        ("stokes", 1.0, 0.0, (0.0, 0.0), ("taylor-hood", "usfem")),
        ("brinkman", 1.0, 100.0, (0.0, 0.0), ("taylor-hood", "usfem")),
        ("oseen", 0.5, 1.0, (2.0, 3.0), ("taylor-hood",)),
    ):

        def source(
            points: FloatArray,
            viscosity: float = viscosity,
            drag: float = drag,
            beta: tuple[float, float] = beta,
        ) -> FloatArray:
            """Derive the complete momentum forcing for the current coefficients."""
            return stokes_source(points, viscosity, drag) + polynomial_transport(points, beta)

        solutions, sampled, metrics = {}, {}, {}
        for method in methods:
            solution = solve_brinkman(
                mesh,
                viscosity=viscosity,
                drag=drag,
                advection=beta,
                source=source,
                dirichlet=stokes_velocity,
                skeleton=skeleton,
                local_refinement=args.local_refinement,
                quadrature_order=6,
                formulation=method,
            )
            solutions[method] = solution
            sampled[method] = sample_solution(solution)
            metrics[method] = {
                "velocity_l2": solution.l2_error(stokes_velocity, order=8),
                "pressure_l2": solution.pressure_l2_error(stokes_pressure, order=8),
                "divergence_l2": solution.divergence_l2(),
                "algebraic_residual": solution.hybrid.residual,
            }
            if not all(np.isfinite(value) for value in metrics[method].values()):
                raise AssertionError("A verification diagnostic is not finite")
            if metrics[method]["algebraic_residual"] > 1e-10:
                raise AssertionError("The assembled saddle system did not converge")
        records[case] = {
            "viscosity": viscosity,
            "drag": drag,
            "advection": beta,
            "methods": metrics,
        }
        label = f"{case.capitalize()}: ν = {viscosity:g}, drag = {drag:g}"
        if any(beta):
            label += f", β = {beta}"
        comparison(
            sampled,
            args.output,
            f"{case}-velocity",
            label + " — velocity magnitude",
            "velocity_magnitude",
            mesh,
        )
        comparison(
            sampled,
            args.output,
            f"{case}-pressure",
            label + " — zero-mean pressure",
            "pressure",
            mesh,
        )
        diagnostics(
            sampled, args.output, f"{case}-diagnostics", label + " — measured errors", metrics, mesh
        )
        profiles(
            solutions, args.output, f"{case}-profiles", label + " — exact and numerical profiles"
        )
    patch = solve_brinkman(
        mesh,
        drag=1,
        advection=(2, 3),
        source=lambda points: rotation(points) + [3, -2],
        dirichlet=rotation,
        skeleton=skeleton,
        local_refinement=args.local_refinement,
    )
    patch_metrics = {
        "velocity_l2": patch.l2_error(rotation),
        "pressure_l2": patch.pressure_l2_error(0),
        "divergence_l2": patch.divergence_l2(),
    }
    if max(patch_metrics.values()) > 1e-9:
        raise AssertionError("The affine Oseen consistency patch failed")
    patch_samples = sample_solution(patch)
    np.testing.assert_allclose(
        patch_samples.velocity, rotation(patch_samples.points), atol=1e-11, rtol=0
    )
    np.testing.assert_allclose(patch_samples.divergence, 0, atol=1e-10, rtol=0)
    profile_points = np.array([[0.23, 0.47], [0.69, 0.32]])
    profile_samples = sample_segment_profile(patch, patch.values, patch.degree, *profile_points)
    np.testing.assert_allclose(
        profile_samples["profile_values"].reshape(-1, 2),
        rotation(profile_samples["profile_points"].reshape(-1, 2)),
        atol=1e-11,
        rtol=0,
    )
    records["oseen_affine_patch"] = patch_metrics
    (args.output / "metrics.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
