"""Build a conforming Q3 Darcy reference for the layer-36 SPE10 case.

This is a separately assembled continuous finite-element calculation with
strong pressure on the horizontal boundaries and natural no-flow sidewalls.
The article's 768-by-1408 quadrilateral grid has 9,738,625 Q3 coefficients.
Pixel-aligned 240-by-440 and 480-by-880 grids provide intermediate resolution
checks. The article grid cuts material pixels, so its Gauss order must be
resolved by integrating on exact material-pixel intersections.

Run from the checkout with ``pixi run -e intel python -m
examples.solve_spe10_reference --shape 768 1408 --order 5``. The default
240-by-440 grid is the smaller, pixel-aligned reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.solve_spe10 import OUTPUT, ROOT, load_layer
from pymhm.fem.scalar.quadrilateral import qk_basis, qk_space, quadrilateral_operators
from pymhm.linalg.linear import solve_linear
from pymhm.meshes.cartesian import CartesianMacroMesh


def evaluate(
    mesh: CartesianMacroMesh,
    degree: int,
    coefficients: np.ndarray,
    permeability: object,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the continuous reference without interpolating material pixels."""
    coordinates = points / mesh.spacing
    counts = np.array([mesh.nx, mesh.ny])
    indices = np.minimum(np.maximum(np.floor(coordinates).astype(int), 0), counts - 1)
    reference = np.clip(coordinates - indices, 0, 1)
    width = mesh.nx * degree + 1
    offsets = np.array([j * width + i for j in range(degree + 1) for i in range(degree + 1)])
    origins = (indices[:, 1] * width + indices[:, 0]) * degree
    nodal = coefficients[origins[:, None] + offsets]
    basis, gradients = qk_basis(degree, reference)
    pressure = np.einsum("qi,qi->q", nodal, basis)
    gradient = np.einsum("qi,qia->qa", nodal, gradients / mesh.spacing)
    centers = (indices + 0.5) * mesh.spacing
    boundary = np.isclose(reference, 0, rtol=0, atol=32 * np.finfo(float).eps) | np.isclose(
        reference, 1, rtol=0, atol=32 * np.finfo(float).eps
    )
    inside = np.where(boundary, np.nextafter(points, centers), points)
    flux = -np.einsum("qab,qb->qa", permeability(inside), gradient)
    return pressure, flux


def run(nx: int, ny: int, *, order: int, solver: str, threads: int) -> dict:
    """Assemble, eliminate boundary values, solve, and save reference samples."""
    degree = 3
    mesh = CartesianMacroMesh(nx, ny, (0, 1200, 0, 2200))
    permeability = load_layer()
    beginning = perf_counter()
    with threadpool_limits(limits=threads):
        matrix, mass, _ = quadrilateral_operators(
            mesh, degree, permeability=permeability, source=0.0, order=order
        )
        del mass
        _, nodes = qk_space(mesh, degree)
        bottom = nodes[:, 1] == 0
        top = nodes[:, 1] == 2200
        free = np.flatnonzero(~(bottom | top))
        coefficients = bottom.astype(float)
        rhs = -(matrix @ coefficients)[free]
        assembled = perf_counter()
        coefficients[free] = solve_linear(matrix[free][:, free], rhs, solver=solver)
        solved = perf_counter()
        # Reactions represent boundary flux in the discrete weak equation;
        # they are not the same as raw-gradient pointwise normal flux.
        residual = matrix @ coefficients
        relative = float(np.linalg.norm(residual[free]) / np.linalg.norm(rhs))
        outward_bottom = -float(residual[bottom].sum())
        outward_top = -float(residual[top].sum())
        del matrix, residual, rhs
        points = np.array(
            [(x, y) for x in np.arange(10, 1200, 20) for y in np.arange(5, 2200, 10)], dtype=float
        )
        pressure, flux = evaluate(mesh, degree, coefficients, permeability, points)
        profile_points = np.array(
            [
                np.column_stack((np.full(201, 199.0), np.linspace(row * 200, (row + 1) * 200, 201)))
                for row in range(11)
            ]
        )
        pprofile, qprofile = [], []
        for row, profile in enumerate(profile_points):
            inside = profile.copy()
            inside[:, 1] = np.nextafter(inside[:, 1], (row + 0.5) * 200)
            p, q = evaluate(mesh, degree, coefficients, permeability, inside)
            pprofile.append(p)
            qprofile.append(q)
        sampled = perf_counter()
    stem = f"reference-q3-{nx}x{ny}-order{order}"
    path = OUTPUT / f"{stem}.npz"
    np.savez_compressed(
        path,
        points=points.reshape(60, 220, 2),
        pressure=pressure.reshape(60, 220),
        flux=flux.reshape(60, 220, 2),
        profile_points=profile_points,
        profile_pressure=np.asarray(pprofile),
        profile_flux=np.asarray(qprofile),
    )
    coefficient_path = ROOT / "build/results/spe10" / f"{stem}-coefficients.npz"
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        coefficient_path,
        coefficients=coefficients,
        shape=np.array([nx, ny]),
        degree=degree,
        order=order,
    )
    report = dict(
        archive=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layer=36,
        method="Conforming continuous Q3 finite elements",
        shape=[nx, ny],
        cells=nx * ny,
        degrees_of_freedom=len(nodes),
        free_degrees_of_freedom=len(free),
        coefficient_pixels_aligned=nx % 60 == 0 and ny % 220 == 0,
        article_grid=(nx == 768 and ny == 1408),
        material_integration="exact Cartesian pixel intersections; aligned-grid fast path",
        quadrature_order=order,
        pressure_nodal_min=float(coefficients.min()),
        pressure_nodal_max=float(coefficients.max()),
        free_residual_relative=float(relative),
        outward_flux_bottom=outward_bottom,
        outward_flux_top=outward_top,
        exterior_balance=outward_bottom + outward_top,
        flux_diagnostic="boundary reactions in the assembled weak equation",
        assembly_seconds=assembled - beginning,
        solve_seconds=solved - assembled,
        sampling_seconds=sampled - solved,
        solver=solver,
        native_threads=threads,
        timing_scope="observed wall time, no exclusive performance benchmark",
        permeability_unit="mD",
        coordinate_unit="ft",
        source=0,
        dirichlet="p=1 at y=0 and p=0 at y=2200, imposed strongly",
        neumann="q dot n=0 on x=0 and x=1200",
        reference="Paredes et al., DOI 10.1016/j.cam.2023.115415, Section 5.2",
    )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Execute one explicitly sized reference with controllable native threads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=[240, 440])
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--solver", default="pypardiso")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            run(*args.shape, order=args.order, solver=args.solver, threads=args.threads), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
