"""Render exact-reference RAD, heat and elasticity cases for the documentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from field_sampling import sample_profile
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.ticker import NullFormatter
from plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.transport.solver import solve_heat, solve_transport
from pymhm._legacy.models.vector import solve_elasticity

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/scalar-elasticity"


def sine(points: np.ndarray) -> np.ndarray:
    """Evaluate the spatial eigenfunction sin(pi*x) sin(pi*y)."""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def displacement(points: np.ndarray) -> np.ndarray:
    """Return a smooth non-affine displacement with zero boundary values."""
    value = sine(points)
    return np.column_stack((value, 0.5 * value))


def elastic_force(points: np.ndarray) -> np.ndarray:
    """Evaluate -mu*Delta(u)-(lambda+mu)*grad(div(u)), with lambda=mu=1."""
    value = sine(points)
    cross = np.cos(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])
    return np.pi**2 * np.column_stack((4 * value - cross, 2 * value - 2 * cross))


def transport_force(points: np.ndarray) -> np.ndarray:
    """Evaluate -0.2*Delta(u)+(1,0.5).grad(u)+2*u for the sine field."""
    x, y = points.T
    return (
        (0.4 * np.pi**2 + 2) * sine(points)
        + np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        + 0.5 * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    )


def skeleton(mesh: TriangleMesh, components: int = 1) -> SkeletonSpace:
    """Use degree-one face traces on each unpartitioned macroface."""
    return SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(degree=1) for _ in mesh.faces), components=components
    )


def field_arrays(solution: Any) -> tuple[mtri.Triangulation, np.ndarray, np.ndarray]:
    """Concatenate broken P1 fields without merging opposite macroface vertices."""
    points, cells, fields = [], [], []
    offset = 0
    for mesh, values in zip(solution.local_meshes, solution.values, strict=True):
        points.append(mesh.points)
        cells.append(mesh.cells + offset)
        fields.append(values)
        offset += len(mesh.points)
    coordinates = np.concatenate(points)
    triangulation = mtri.Triangulation(*coordinates.T, triangles=np.concatenate(cells))
    return triangulation, coordinates, np.concatenate(fields)


def field_panel(
    axis: Any,
    triangulation: mtri.Triangulation,
    values: np.ndarray,
    title: str,
    *,
    macro_mesh: TriangleMesh,
    vmin: float = 0.0,
    vmax: float | None = None,
    error: bool = False,
) -> None:
    """Plot a P1 field with its real macrotriangulation and an explicit color scale."""
    artist = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap="magma" if error else "viridis",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    draw_macro_mesh(axis, macro_mesh)
    axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    axis.figure.colorbar(artist, ax=axis, shrink=0.82)


def save(figure: Any, name: str) -> None:
    """Save a standalone vector container and a high-resolution raster figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def rad_case() -> dict[str, Any]:
    """Compare a resolved reaction-advection-diffusion field with its exact reference."""
    mesh = TriangleMesh.unit_square(4)
    solution = solve_transport(
        mesh,
        skeleton=skeleton(mesh),
        diffusion=0.2,
        velocity=(1.0, 0.5),
        reaction=2.0,
        source=transport_force,
        local_refinement=4,
    )
    triangulation, points, values = field_arrays(solution)
    exact = sine(points)
    minimum = float(min(values.min(), exact.min()))
    maximum = float(max(values.max(), exact.max()))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    field_panel(
        axes[0],
        triangulation,
        exact,
        "Exact scalar field",
        macro_mesh=mesh,
        vmin=minimum,
        vmax=maximum,
    )
    field_panel(
        axes[1],
        triangulation,
        values,
        "MHM P1 field",
        macro_mesh=mesh,
        vmin=minimum,
        vmax=maximum,
    )
    field_panel(
        axes[2],
        triangulation,
        abs(values - exact),
        "Nodal absolute error",
        macro_mesh=mesh,
        error=True,
    )
    save(figure, "rad-fields")
    x = np.linspace(0, 1, 501)
    profile = np.column_stack((x, np.full(len(x), 0.37)))
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    axis.plot(x, sine(profile), "k--", label="Exact sine profile")
    sampled = sample_profile(solution, solution.values, 1)
    for index, (points, values) in enumerate(
        zip(sampled["profile_points"], sampled["profile_values"], strict=True)
    ):
        axis.plot(
            points[:, 0], values, color="#0072B2", label="MHM P1" if index == 0 else "_nolegend_"
        )
    crossings = macro_profile_breaks(mesh, profile[0], profile[-1])
    mark_macro_interfaces(axis, crossings[1:-1], label=True)
    axis.set(xlabel="x", ylabel="u(x, 0.37)", title="RAD: a profile crossing macrofaces")
    axis.legend()
    axis.grid(alpha=0.25)
    save(figure, "rad-profile")
    error = solution.l2_error(sine, order=8)
    assert error < 0.04
    return {"l2_error": error, "macro_resolution": 4, "local_refinement": 4, "trace_degree": 1}


def heat_case() -> dict[str, Any]:
    """Compare backward Euler with the exact heat eigenmode and time-only BE decay."""
    mesh = TriangleMesh.unit_square(4)
    histories = {}
    rows = []
    for steps in (4, 8, 16, 32, 64):
        times = np.linspace(0, 0.1, steps + 1)
        solutions = solve_heat(
            mesh, times, initial=sine, skeleton=skeleton(mesh), local_refinement=4
        )
        histories[steps] = (times, solutions)
        energy = [field.l2_error(0.0, order=8) ** 2 for field in solutions]
        assert np.all(np.diff(energy) < 0)
        rows.append(
            {
                "steps": steps,
                "dt": float(times[1]),
                "l2_error": solutions[-1].l2_error(
                    lambda x: np.exp(-2 * np.pi**2 * 0.1) * sine(x), order=8
                ),
                "energies": energy,
            }
        )
    final = histories[max(histories)][1][-1]
    triangulation, points, values = field_arrays(final)
    exact = np.exp(-2 * np.pi**2 * 0.1) * sine(points)
    _, _, coarse = field_arrays(histories[4][1][-1])
    minimum = float(min(coarse.min(), values.min(), exact.min()))
    maximum = float(max(coarse.max(), values.max(), exact.max()))
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for axis, field, title in zip(
        axes[:3],
        (exact, coarse, values),
        (
            "Exact\nat t=0.1",
            "Backward Euler\ndt=0.025",
            f"Backward Euler\ndt={rows[-1]['dt']:g}",
        ),
        strict=True,
    ):
        field_panel(axis, triangulation, field, title, macro_mesh=mesh, vmin=minimum, vmax=maximum)
    field_panel(
        axes[3],
        triangulation,
        abs(values - exact),
        "Fine-step nodal error",
        macro_mesh=mesh,
        error=True,
    )
    save(figure, "heat-fields")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    time = np.linspace(0, 0.1, 200)
    axes[0].semilogy(time, 0.25 * np.exp(-4 * np.pi**2 * time), "k--", label="Exact continuum")
    for row in rows:
        times = histories[row["steps"]][0][1:]
        axes[0].semilogy(times, row["energies"], "o-", ms=3, label=f"MHM dt={row['dt']:g}")
    axes[0].set(xlabel="Time", ylabel="Squared L2 norm", title="Heat: energy decays at every step")
    dt = np.array([row["dt"] for row in rows])
    time_only = 0.5 * abs((1 + 2 * np.pi**2 * dt) ** (-0.1 / dt) - np.exp(-0.2 * np.pi**2))
    axes[1].loglog(dt, [row["l2_error"] for row in rows], "o-", label="MHM spatial + time error")
    axes[1].loglog(dt, time_only, "k--", label="Time-only BE error (exact space)")
    axes[1].set(
        xlabel="Time step", ylabel="L2 error at t=0.1", title="Expected first-order time trend"
    )
    axes[1].set_xticks(dt, [f"1/{round(1 / value)}" for value in dt])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, "heat-decay")
    assert rows[-1]["l2_error"] < rows[0]["l2_error"]
    return {"time_steps": rows, "macro_resolution": 4, "local_refinement": 4, "trace_degree": 1}


def elasticity_case() -> dict[str, Any]:
    """Check a non-affine plane-strain problem and show a scaled deformation."""
    rows = []
    solution = None
    for resolution in (2, 4, 8, 16, 32):
        mesh = TriangleMesh.unit_square(resolution)
        result = solve_elasticity(
            mesh,
            formulation="primal",
            skeleton=skeleton(mesh, 2),
            source=elastic_force,
            local_refinement=4,
        )
        rows.append({"macro_resolution": resolution, "l2_error": result.l2_error(displacement, 8)})
        if resolution == 4:
            solution = result
    assert solution is not None
    macro_mesh = solution.skeleton.mesh
    triangulation, points, values = field_arrays(solution)
    exact = displacement(points)
    magnitude, exact_magnitude = np.linalg.norm(values, axis=1), np.linalg.norm(exact, axis=1)
    error = np.linalg.norm(values - exact, axis=1)
    maximum = float(max(magnitude.max(), exact_magnitude.max()))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    field_panel(
        axes[0],
        triangulation,
        exact_magnitude,
        "Exact displacement\nmagnitude",
        macro_mesh=macro_mesh,
        vmax=maximum,
    )
    field_panel(
        axes[1],
        triangulation,
        magnitude,
        "MHM P1 displacement\nmagnitude",
        macro_mesh=macro_mesh,
        vmax=maximum,
    )
    field_panel(
        axes[2],
        triangulation,
        error,
        "Nodal vector error\nmagnitude",
        macro_mesh=macro_mesh,
        error=True,
    )
    save(figure, "elasticity-fields")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    scale = 0.2
    deformed = points + scale * values
    axes[0].triplot(triangulation, color="0.75", linewidth=0.35)
    draw_macro_mesh(axes[0], macro_mesh, label=True)
    axes[0].set(xlabel="x", ylabel="y", aspect="equal", title="Reference macro and local meshes")
    axes[0].legend(fontsize=8)
    axes[1].triplot(*deformed.T, triangulation.triangles, color="#087e8b", linewidth=0.45)
    macro_boundaries = np.concatenate(
        [
            (fine.points + scale * field)[fine.faces[fine.boundary_faces]]
            for fine, field in zip(solution.local_meshes, solution.values, strict=True)
        ]
    )
    boundary_artist = LineCollection(
        macro_boundaries,
        colors="#202020",
        linewidths=1.0,
        zorder=3,
        label="Deformed macro boundaries",
    )
    boundary_artist.set_path_effects(
        [patheffects.Stroke(linewidth=2.0, foreground="white"), patheffects.Normal()]
    )
    axes[1].add_collection(boundary_artist)
    axes[1].set(xlabel="x", ylabel="y", aspect="equal", title="Deformed local meshes: x + 0.2 u_h")
    axes[1].legend(fontsize=8)
    spacing = 1 / np.array([row["macro_resolution"] for row in rows])
    errors = np.array([row["l2_error"] for row in rows])
    axes[2].loglog(spacing, errors, "o-", label="Measured displacement L2 error")
    axes[2].loglog(spacing, errors[0] * (spacing / spacing[0]) ** 2, "k--", label="Slope 2 guide")
    axes[2].set(xlabel="Macro grid spacing", ylabel="L2 error", title="Simultaneous H/h refinement")
    axes[2].set_xticks(spacing, [f"1/{row['macro_resolution']}" for row in rows])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(fontsize=8)
    save(figure, "elasticity-deformation")
    assert np.all(np.diff(errors) < 0)
    return {"convergence": rows, "lame_lambda": 1.0, "lame_mu": 1.0, "trace_degree": 1}


def main() -> None:
    """Generate checked scalar/vector figures and machine-readable error records."""
    with threadpool_limits(limits=1):
        report = {"rad": rad_case(), "heat": heat_case(), "elasticity": elasticity_case()}
    destination = ROOT / "examples/results/scalar-elasticity.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
