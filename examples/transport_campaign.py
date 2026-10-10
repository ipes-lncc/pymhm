"""Separate multilevel campaigns for conservative and Darcy-coupled transport."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.formulations.application import darcy as solve_darcy
from examples.formulations.application import transport as solve_transport
from examples.formulations.darcy_transport import (
    solve_darcy_trajectory as solve_darcy_transport,
)
from examples.plot_mesh import draw_macro_mesh
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.adaptivity.transport import TransportBounds, solve_adaptive_transport
from pymhm.core.validation import positive_int
from pymhm.io.workspace import case_workspace

ROOT = case_workspace()
DATA = ROOT / "examples/results/transport"
FIGURES = ROOT / "docs/figures/transport"


def layer(points: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """Evaluate the stable a=1 solution of L11 section 5.1."""
    x = points[:, 0]
    return x - (np.exp((x - 1) / epsilon) - np.exp(-1 / epsilon)) / (1 - np.exp(-1 / epsilon))


def natural_horizontal(mesh: TriangleMesh) -> dict[int, float]:
    """Select horizontal exterior sides for homogeneous diffusive flux."""
    return {int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) < 1e-14}


def archive(solution: Any, name: str, exact: Any) -> None:
    """Save separate fine-element samples and the actual macrogeometry."""
    samples: dict[str, Any] = sample_field(
        solution.local_meshes, solution.values, solution.degree, 3
    )
    np.savez_compressed(
        DATA / f"{name}.npz",
        **samples,
        exact=exact(samples["points"]),
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
    )


def published_stationary() -> list[dict[str, Any]]:
    """Refine the macro mesh for the published analytical mixed-boundary PDE."""
    rows = []
    for resolution in (1, 2, 4, 8, 16):
        mesh = TriangleMesh.unit_square(resolution)
        solution = solve_transport(
            mesh,
            diffusion=0.02,
            velocity=(1, 0),
            source=1,
            degree=3,
            local_refinement=4,
            stabilization="supg",
            diffusive_flux=natural_horizontal(mesh),
            dirichlet_enforcement="strong",
            quadrature_order=8,
        )
        row = {
            "macro_resolution": resolution,
            "macro_triangles": len(mesh.cells),
            "trace_dofs": solution.skeleton.size,
            "local_refinement": 4,
            "degree": 3,
            "l2_error_order10": solution.l2_error(layer, 10),
            "l2_error_order12": solution.l2_error(layer, 12),
        }
        rows.append(row)
        print("stationary", row, flush=True)
        if resolution == 8:
            archive(solution, "published-layer", layer)
    return rows


def adaptive() -> list[dict[str, Any]]:
    """Refine faces for the same exact field with full analytical Dirichlet data."""
    mesh = TriangleMesh.unit_square(4)
    result = solve_adaptive_transport(
        SkeletonSpace(mesh),
        TransportBounds(0.02, 1, 0),
        iterations=5,
        theta=0.75,
        solve_step=solve_transport,
        diffusion=0.02,
        velocity=(1, 0),
        source=1,
        dirichlet=layer,
        degree=3,
        local_refinement=8,
        stabilization="supg",
        quadrature_order=8,
    )
    rows = []
    for index, (solution, indicator) in enumerate(
        zip(result.solutions, result.indicators, strict=True)
    ):
        row = {
            "iteration": index,
            "trace_dofs": solution.skeleton.size,
            "indicator": indicator.total,
            "l2_error_order10": solution.l2_error(layer, 10),
            "l2_error_order12": solution.l2_error(layer, 12),
            "maximum_face_segments": max(len(face.degrees) for face in solution.skeleton.faces),
        }
        rows.append(row)
        print("adaptive", row, flush=True)
        if index in (0, len(result.solutions) - 1):
            archive(solution, f"adaptive-{index}", layer)
    return rows


def conductivity(points: np.ndarray) -> np.ndarray:
    """Use two horizontal isotropic layers with permeability one and ten."""
    return np.where(points[:, 1] < 0.5, 1.0, 10.0)


def concentration(points: np.ndarray) -> np.ndarray:
    """Return the quadratic spatial factor of a manufactured concentration."""
    x = points[:, 0]
    return 1 + x * (3 - x) / 9


def transient_force(points: np.ndarray, time: float) -> np.ndarray:
    """Evaluate u_t-div(D grad(u))+q.grad(u) for the layered Darcy velocity."""
    flux = conductivity(points)
    diffusivity = 1e-6 + 1e-2 * flux
    x = points[:, 0]
    return np.exp(-time) * (-concentration(points) + 2 * diffusivity / 9 + flux * (3 - 2 * x) / 9)


def layered_darcy(macro_divisions: int = 2) -> tuple[Any, SkeletonSpace]:
    """Prepare the documented layered RT0/P0 Darcy field and P2 transport trace.

    The domain is [0,3] by [0,1], with K=1 below y=1/2 and K=10 above.
    Pressure 3-x is imposed on the exterior. An even number of Cartesian
    divisions fits the material interface to both macro and fine triangles;
    every macro retains four edge subdivisions, independent of this input.
    The default preserves the original eight-macrotriangular campaign.
    """
    count = positive_int(macro_divisions, "macro divisions")
    if count % 2:
        raise ValueError("layered Darcy macro divisions must be even to fit y=1/2")
    unit = TriangleMesh.unit_square(count)
    mesh = TriangleMesh(unit.points * (3, 1), unit.cells)
    darcy = solve_darcy(
        mesh,
        formulation="mixed",
        permeability=conductivity,
        dirichlet=lambda points: 3 - points[:, 0],
        local_refinement=4,
        quadrature_order=8,
    )
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    return darcy, skeleton


def layered_trajectory(
    darcy: Any, skeleton: SkeletonSpace, steps: int, *, check_original: bool = False
) -> Any:
    """Advance the existing analytical concentration to t=1 on the supplied field.

    Darcy's actual fine partition is retained. Transport uses local P3/SUPG,
    P2 macroface traces, unit capacity, zero reaction, the independently derived
    transient_force, strong concentration exp(-t) at x=0,3 and zero diffusive
    flux at y=0,1. The shared time integrator owns every numerical operation.
    """
    count = positive_int(steps, "time steps")
    return solve_darcy_transport(
        darcy,
        np.linspace(0, 1, count + 1),
        skeleton=skeleton,
        initial=concentration,
        source=transient_force,
        dirichlet=lambda points, time: np.full(len(points), np.exp(-time)),
        diffusive_flux=natural_horizontal(skeleton.mesh),
        degree=3,
        stabilization="supg",
        quadrature_order=8,
        check_original=check_original,
    )


def transient() -> list[dict[str, Any]]:
    """Check five time increments using the computed conservative Darcy RT0 flux."""
    darcy, skeleton = layered_darcy()
    flux_error = darcy.flux_l2_error(
        lambda points: np.column_stack((conductivity(points), np.zeros(len(points)))), 10
    )
    rows = []
    for steps in (4, 8, 16, 32, 64):
        result = layered_trajectory(darcy, skeleton, steps)
        solution = result.solutions[-1]
        row = {
            "steps": steps,
            "dt": 1 / steps,
            "l2_error_order10": solution.l2_error(lambda x: np.exp(-1) * concentration(x), 10),
            "l2_error_order12": solution.l2_error(lambda x: np.exp(-1) * concentration(x), 12),
            "operator_builds": result.operator_builds,
            "darcy_flux_l2_error": flux_error,
            "maximum_discrete_balance_residual": float(np.max(np.abs(result.balance_residuals))),
            "final_mass": float(result.total_mass()[-1]),
        }
        rows.append(row)
        print("transient", row, flush=True)
        if steps == 64:
            archive(solution, "darcy-transient", lambda x: np.exp(-1) * concentration(x))
    return rows


def save(figure: Any, name: str) -> None:
    """Write a PNG and a vector-labelled SVG without altering measured data."""
    import matplotlib.pyplot as plt

    for extension in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def plot_fields(name: str) -> None:
    """Compare exact, numerical and signed error using separate broken triangles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    data = np.load(DATA / f"{name}.npz")
    mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
    triangulation = mtri.Triangulation(
        data["points"][:, 0], data["points"][:, 1], triangles=data["cells"]
    )
    exact, numerical = data["exact"], data["values"]
    figure, axes = plt.subplots(
        1, 3, figsize=(12, 2.9 if name == "darcy-transient" else 4.4), layout="constrained"
    )
    low, high = min(exact.min(), numerical.min()), max(exact.max(), numerical.max())
    for index, (values, title) in enumerate(
        zip(
            (exact, numerical, numerical - exact),
            ("Analytical", "MHM", "MHM − analytical"),
            strict=True,
        )
    ):
        limit = max(float(np.max(np.abs(values))), 1e-15)
        artist = axes[index].tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap="RdBu_r" if index == 2 else "viridis",
            vmin=-limit if index == 2 else low,
            vmax=limit if index == 2 else high,
            rasterized=True,
        )
        draw_macro_mesh(axes[index], mesh)
        axes[index].set(title=title, xlabel="x", ylabel="y", aspect="equal")
        colorbar = figure.colorbar(
            artist,
            ax=axes[index],
            orientation="horizontal",
            pad=0.2 if name == "darcy-transient" else 0.14,
            label="Signed error" if index == 2 else "Scalar field",
        )
        colorbar.locator = MaxNLocator(5)
        colorbar.formatter = ScalarFormatter(useMathText=True)
        colorbar.formatter.set_powerlimits((-2, 3))
        colorbar.update_ticks()
    save(figure, name)


def plot() -> None:
    """Render all recorded refinement curves and physical-field comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter

    FIGURES.mkdir(parents=True, exist_ok=True)
    record = json.loads((DATA / "campaign.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for index, (section, abscissa, label) in enumerate(
        (
            ("stationary", "macro_resolution", "Macro subdivisions"),
            ("adaptive", "trace_dofs", "Scalar trace DOFs"),
            ("transient", "dt", "Time increment"),
        )
    ):
        rows = record[section]
        x = np.array([row[abscissa] for row in rows])
        error = np.array([row["l2_error_order12"] for row in rows])
        axes[index].loglog(x, error, "o-", label="Physical L² error")
        if section == "adaptive":
            axes[index].loglog(
                x, [row["indicator"] for row in rows], "s--", label="Face indicator η"
            )
        if section == "transient":
            axes[index].loglog(x, error[-1] * x / x[-1], ":", label="Slope one")
        axes[index].set(xlabel=label, ylabel="Error / indicator", title=section.capitalize())
        axes[index].xaxis.set_minor_formatter(NullFormatter())
        if section == "stationary":
            axes[index].set_xticks(x, [str(int(value)) for value in x])
        elif section == "transient":
            axes[index].set_xticks(x, [f"1/{int(round(1 / value))}" for value in x])
        else:
            axes[index].xaxis.set_major_locator(MaxNLocator(4))
            axes[index].xaxis.set_major_formatter(ScalarFormatter())
        axes[index].grid(True, which="both", alpha=0.2)
        axes[index].legend()
    save(figure, "refinement")
    for name in ("published-layer", "adaptive-0", "adaptive-4", "darcy-transient"):
        plot_fields(name)


def main() -> None:
    """Collect independent campaigns on request, or only render saved records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    options = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if options.collect:
        with threadpool_limits(1):
            record = {
                "stationary": published_stationary(),
                "adaptive": adaptive(),
                "transient": transient(),
                "conventions": {
                    "stationary": (
                        "L11 section 5.1 PDE; independently specified "
                        "triangular P3/SUPG mesh series"
                    ),
                    "adaptive": (
                        "Analytical layer with full Dirichlet boundary; "
                        "L11 face bisection; fixed 32 macros, r8, P3"
                    ),
                    "transient": (
                        "Original layered manufactured test using L11 Darcy-dispersion coupling; "
                        "backward Euler"
                    ),
                },
            }
        (DATA / "campaign.json").write_text(json.dumps(record, indent=2) + "\n")
    plot()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.transport_campaign").main()
