"""Material-fitted face partitions for the two-layer problem of CAMWA 2026."""

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
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.layered_poisson import LayeredPoissonSeries
from examples.plot_mesh import draw_macro_mesh
from examples.unfitted_geometry import macro_mesh
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.fem.quadrature.material import (
    fit_material_faces,
    fit_material_mesh,
    material_triangle_quadrature,
)
from pymhm.fem.scalar.quadrilateral import qk_space, quadrilateral_operators
from pymhm.fem.scalar.triangle import element_tabulate
from pymhm.linalg.linear import solve_linear
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/unfitted"
FIGURES = ROOT / "docs/figures/unfitted"


def errors(solution: Any, modes: int = 511, order: int = 8) -> dict[str, float]:
    """Integrate pressure and genuine broken gradient errors across material cuts."""
    reference = LayeredPoissonSeries(10, modes)
    pressure_error, gradient_error, energy_error = 0.0, 0.0, 0.0
    for mesh, coefficients in zip(solution.local_meshes, solution.pressure, strict=True):
        bary, weights, material = material_triangle_quadrature(mesh, solution.permeability, order)
        points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
        exact, exact_gradient = reference.evaluate(points.reshape(-1, 2))
        dofs, _, basis, gradients, _ = element_tabulate(mesh, solution.degree, bary)
        values = np.einsum("ti,tqi->tq", coefficients[dofs], basis)
        computed_gradient = np.einsum("ti,tqia->tqa", coefficients[dofs], gradients)
        dp = values - exact.reshape(weights.shape)
        dg = computed_gradient - exact_gradient.reshape(*weights.shape, 2)
        squared = np.sum(dg**2, axis=-1)
        pressure_error += float(mesh.areas @ np.sum(weights * dp**2, axis=1))
        gradient_error += float(mesh.areas @ np.sum(weights * squared, axis=1))
        energy_error += float(
            mesh.areas
            @ np.sum(weights * squared * np.asarray(material).reshape(weights.shape), axis=1)
        )
    return {
        "pressure_l2": float(np.sqrt(pressure_error)),
        "gradient_l2": float(np.sqrt(gradient_error)),
        "energy_error": float(np.sqrt(energy_error)),
    }


def classical() -> list[dict[str, float | int]]:
    """Compute five conforming Q1 levels and their energy error by Galerkin orthogonality."""
    field = CartesianCellField(np.array([[10.0, 1.0]]), (1.0, 0.5))
    reference = LayeredPoissonSeries(10, 4095)
    exact_energy = reference.energy_squared()
    rows = []
    previous_energy = None
    for resolution in (32, 64, 128, 256, 512):
        mesh = CartesianMacroMesh(resolution, resolution)
        _, points = qk_space(mesh, 1)
        matrix, _, force = quadrilateral_operators(mesh, 1, permeability=field, source=1, order=2)
        free = np.flatnonzero(np.all((points > 0) & (points < 1), axis=1))
        values = np.zeros(len(points))
        values[free] = solve_linear(matrix[free][:, free], force[free])
        energy = float(force @ values)
        row = {
            "resolution": resolution,
            "quadrilateral_cells": resolution**2,
            "dofs": len(points),
            "energy_squared": energy,
            "analytical_energy_squared_4095_modes": exact_energy,
            "energy_error": float(np.sqrt(exact_energy - energy)),
            "successive_energy_difference": (
                0.0 if previous_energy is None else float(np.sqrt(energy - previous_energy))
            ),
        }
        rows.append(row)
        previous_energy = energy
        print("classical", row, flush=True)
    return rows


def solve_case(
    setting: str, delta: float, refinement: int, *, archive: bool = False, fitted: bool = False
) -> dict[str, Any]:
    """Solve a fixed-degree primal problem with only the stated skeletal modification."""
    mesh = macro_mesh(delta)
    field = CartesianCellField(np.array([[10.0, 1.0]]), (1.0, 0.5))
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    if setting == "S2":
        skeleton = fit_material_faces(skeleton, field)
    local_meshes = (
        tuple(
            fit_material_mesh(mesh.submesh(cell, refinement), field)
            for cell in range(len(mesh.cells))
        )
        if fitted
        else None
    )
    tag = f"{'fitted-' if fitted else ''}r{refinement}"
    solution = solve_darcy(
        mesh,
        permeability=field,
        source=1,
        skeleton=skeleton,
        degree=4,
        local_refinement=refinement,
        local_meshes=local_meshes,
        quadrature_order=7,
    )
    metrics = errors(solution)
    row = {
        "setting": setting,
        "delta": delta,
        "local_refinement": refinement,
        "local_degree": 4,
        "material_fitted_local_meshes": fitted,
        "fine_triangles": sum(len(local.cells) for local in solution.local_meshes),
        "trace_degree": 2,
        "trace_dofs": skeleton.size,
        "macro_triangles": 16,
        "macro_balance_linf": float(np.max(np.abs(solution.conservation_residuals()))),
        **metrics,
    }
    if archive:
        samples = sample_field(solution.local_meshes, solution.pressure, 4, 2)
        exact, gradient = LayeredPoissonSeries(10, 1023).evaluate(samples["points"])
        np.savez_compressed(
            DATA / f"{setting}-{tag}.npz",
            **samples,
            exact=exact,
            exact_gradient=gradient,
            macro_points=mesh.points,
            macro_cells=mesh.cells,
        )
        independent = errors(solution, modes=1023, order=10)
        row["independent_1023_modes_order10"] = independent
    print("MHM", row, flush=True)
    return row


def collect(refinement: int, fitted: bool = False) -> None:
    """Acquire five perturbations and independently refined classical fields."""
    tag = f"{'fitted-' if fitted else ''}r{refinement}"
    rows = [solve_case("S0", 0, refinement, archive=True, fitted=fitted)]
    for delta in (1 / 126, 1 / 62, 1 / 30, 1 / 14, 1 / 6):
        for setting in ("S1", "S2"):
            rows.append(
                solve_case(setting, delta, refinement, archive=delta == 1 / 6, fitted=fitted)
            )
        (DATA / f"mhm-{tag}.json").write_text(json.dumps(rows, indent=2) + "\n")
    path = DATA / "classical.json"
    if not path.exists():
        path.write_text(json.dumps(classical(), indent=2) + "\n")


def save(figure: Any, name: str) -> None:
    """Save rasterized-field vector and PNG figures."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def plot(refinement: int, fitted: bool = False) -> None:
    """Render the actual macro meshes and same-scale scalar fields."""
    tag = f"{'fitted-' if fitted else ''}r{refinement}"
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = json.loads((DATA / f"mhm-{tag}.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for setting, marker in (("S1", "x"), ("S2", "o")):
        selected = [row for row in rows if row["setting"] == setting]
        for axis, key in zip(axes, ("gradient_l2", "pressure_l2"), strict=True):
            axis.semilogx(
                [row["delta"] for row in selected],
                [row[key] for row in selected],
                marker + "-",
                label=setting,
            )
    for axis, key, label in zip(
        axes,
        ("gradient_l2", "pressure_l2"),
        ("Broken gradient L² error", "Pressure L² error"),
        strict=True,
    ):
        axis.axhline(rows[0][key], color="black", linestyle="--", label="S0, fitted macros")
        axis.set(xlabel="Macro perturbation δ", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend()
    save(figure, f"perturbation-{tag}")
    fields = [np.load(DATA / f"{setting}-{tag}.npz") for setting in ("S0", "S1", "S2")]
    low = min(float(field["values"].min()) for field in fields)
    high = max(float(field["values"].max()) for field in fields)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for axis, field, setting in zip(axes, fields, ("S0", "S1", "S2"), strict=True):
        mesh = TriangleMesh(field["macro_points"], field["macro_cells"])
        triangulation = mtri.Triangulation(*field["points"].T, triangles=field["cells"])
        artist = axis.tripcolor(
            triangulation, field["values"], shading="gouraud", vmin=low, vmax=high, rasterized=True
        )
        draw_macro_mesh(axis, mesh)
        axis.axhline(0.5, color="red", linewidth=1.2)
        if setting == "S2":
            skeleton = fit_material_faces(
                SkeletonSpace(mesh), CartesianCellField(np.ones((1, 2)), (1, 0.5))
            )
            for face, space in enumerate(skeleton.faces):
                start, end = mesh.points[mesh.faces[face]]
                points = start + np.asarray(space.breaks[1:-1])[:, None] * (end - start)
                axis.scatter(*points.T, color="black", s=16, zorder=7)
        axis.set(title=setting, xlabel="x", ylabel="y", aspect="equal")
        figure.colorbar(artist, ax=axis, orientation="horizontal", pad=0.14, label="Pressure")
    save(figure, f"fields-{tag}")


def main() -> None:
    """Run a separate numerical campaign or plot its existing record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--refinement", type=int, default=16)
    parser.add_argument("--fit-locals", action="store_true")
    options = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if options.collect:
        with threadpool_limits(1):
            collect(options.refinement, options.fit_locals)
    plot(options.refinement, options.fit_locals)


if __name__ == "__main__":
    main()
