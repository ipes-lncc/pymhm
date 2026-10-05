"""Solve the declared SPE10 layer-1 USFEM-MHM configuration with side slip.

The default crisscross mesh, P3/P3 local elements and broken P1 skeleton follow
Araya, Harder, Poza and Valentin (2017), Figures 18–20. Material values retain the source
mD numbers and coordinate values in feet: this is the numerical coefficient
convention of the benchmark, not an implicit SI conversion. Coefficient jumps
cut triangular fine elements; the volume quadrature order is recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy
from field_sampling import local_values
from plot_mesh import macro_profile_breaks
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.vector import VectorSolution, solve_brinkman
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space, tabulate
from pymhm.io.provenance import current_source_manifest
from pymhm.materials.cartesian import CartesianCellField

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/spe10"


def crisscross_mesh(nx: int = 6, ny: int = 11) -> TriangleMesh:
    """Split each rectangular grid cell into four counterclockwise triangles."""
    points = [[1200 * i / nx, 2200 * j / ny] for j in range(ny + 1) for i in range(nx + 1)]
    cells = []
    for j in range(ny):
        for i in range(nx):
            corners = [j * (nx + 1) + i, j * (nx + 1) + i + 1]
            corners.extend([corners[1] + nx + 1, corners[0] + nx + 1])
            center = len(points)
            points.append(np.mean(np.asarray(points)[corners], axis=0).tolist())
            cells.extend([corners[k], corners[(k + 1) % 4], center] for k in range(4))
    return TriangleMesh(points, cells)


def inlet_velocity(points: np.ndarray) -> np.ndarray:
    """Prescribe upward unit inflow on the bottom; side normal velocity is zero."""
    return np.column_stack((np.zeros(len(points)), (points[:, 1] == 0.0).astype(float)))


def layer_resistance() -> CartesianCellField:
    """Return scalar gamma=0.3/Kx from the unchanged layer-1 material pixels."""
    with np.load(OUTPUT / "layer-1.npz") as data:
        return CartesianCellField(0.3 / data["permeability"][..., 0], (20.0, 10.0))


def owners(mesh: TriangleMesh, points: np.ndarray) -> np.ndarray:
    """Select a containing macrotriangle deterministically for display samples."""
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    result = []
    for batch in np.array_split(points, max(1, len(points) // 2000)):
        xi = np.einsum("tij,tqj->tqi", inverse, batch[None] - vertices[:, None, 0])
        bary = np.concatenate((1 - xi.sum(axis=2, keepdims=True), xi), axis=2)
        owner = np.argmax(bary.min(axis=2), axis=0)
        if np.min(bary[owner, np.arange(len(batch))]) < -1e-10:
            raise ValueError("display point lies outside the macro mesh")
        result.append(owner)
    return np.concatenate(result)


def evaluate_samples(solution: VectorSolution, points: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate scalar pressure and vector velocity from each selected macrocell."""
    owner = owners(solution.skeleton.mesh, points)
    velocity, pressure = np.empty((len(points), 2)), np.empty(len(points))
    for cell in np.unique(owner):
        mask = owner == cell
        fine = solution.local_meshes[cell]
        velocity[mask] = local_values(fine, solution.values[cell], solution.degree, points[mask])
        pressure[mask] = local_values(
            fine, solution.pressure[cell], solution.pressure_degree, points[mask]
        )
    return dict(points=points, owner=owner, velocity=velocity, pressure=pressure)


def archive_fields(solution: VectorSolution) -> dict[str, np.ndarray]:
    """Keep complete broken coefficients, pixel samples and one-sided profile pieces."""
    mesh = solution.skeleton.mesh
    points = np.array([(x, y) for x in np.arange(10, 1200, 20) for y in np.arange(5, 2200, 10)])
    arrays = evaluate_samples(solution, points)
    arrays = {key: value.reshape((60, 220) + value.shape[1:]) for key, value in arrays.items()}
    start, end = np.array([199.0, 0.0]), np.array([199.0, 2200.0])
    breaks = macro_profile_breaks(mesh, start, end)
    profile_points, profile_velocity, profile_pressure, profile_owner = [], [], [], []
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        middle = (start + (left + right) / 2 * (end - start))[None]
        cell = int(owners(mesh, middle)[0])
        samples = start + np.linspace(left, right, 121)[:, None] * (end - start)
        fine = solution.local_meshes[cell]
        profile_points.append(samples)
        profile_velocity.append(local_values(fine, solution.values[cell], solution.degree, samples))
        profile_pressure.append(
            local_values(fine, solution.pressure[cell], solution.pressure_degree, samples)
        )
        profile_owner.append(cell)
    arrays.update(
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.asarray([fine.points for fine in solution.local_meshes]),
        local_cells=np.asarray([fine.cells for fine in solution.local_meshes]),
        velocity_nodes=np.asarray(
            [nodal_space(fine, solution.degree)[1] for fine in solution.local_meshes]
        ),
        pressure_nodes=np.asarray(
            [nodal_space(fine, solution.pressure_degree)[1] for fine in solution.local_meshes]
        ),
        local_velocity=np.asarray(solution.values),
        local_pressure=np.asarray(solution.pressure),
        trace=solution.hybrid.trace,
        coarse=np.asarray(solution.hybrid.coarse),
        profile_points=np.asarray(profile_points),
        profile_velocity=np.asarray(profile_velocity),
        profile_pressure=np.asarray(profile_pressure),
        profile_owner=np.asarray(profile_owner),
        profile_breaks=breaks,
    )
    return arrays


def physical_diagnostics(solution: VectorSolution, refinement: int) -> dict[str, Any]:
    """Integrate local divergence and external velocity flux with resolved face pieces."""
    mesh = solution.skeleton.mesh
    bary, weights = triangle_quadrature(max(5, solution.degree + 2))
    balances = []
    for fine, field in zip(solution.local_meshes, solution.values, strict=True):
        dofs, _, _, gradient, _ = tabulate(fine, solution.degree, bary)
        divergence = np.einsum("tqia,tia->tq", gradient, field[dofs])
        balances.append(float(fine.areas @ (divergence @ weights)))
    flux = np.zeros(4)
    gauss, w = np.polynomial.legendre.leggauss(solution.degree + 2)
    t = ((np.arange(refinement)[:, None] + (gauss[None] + 1) / 2) / refinement).ravel()
    w = np.tile(w / (2 * refinement), refinement)
    for face in mesh.boundary_faces:
        cell = int(np.flatnonzero(np.any(mesh.cell_faces == face, axis=1))[0])
        a, b = mesh.points[mesh.faces[face]]
        normal = mesh.normals[face]
        velocity = local_values(
            solution.local_meshes[cell],
            solution.values[cell],
            solution.degree,
            a + t[:, None] * (b - a),
        )
        side = 0 if normal[1] < -0.5 else 1 if normal[0] > 0.5 else 2 if normal[1] > 0.5 else 3
        flux[side] += mesh.lengths[face] * (w @ (velocity @ normal))
    return dict(
        divergence_l2=solution.divergence_l2(),
        macro_divergence_integrals=balances,
        macro_divergence_integral_max=float(np.max(np.abs(balances))),
        exterior_velocity_flux_bottom_right_top_left=flux.tolist(),
        total_exterior_velocity_flux=float(flux.sum()),
    )


def source_hashes() -> dict[str, str]:
    """Record all package sources and this driver at the time of acquisition."""
    paths = [*sorted((ROOT / "src/pymhm").rglob("*.py")), Path(__file__).resolve()]
    return current_source_manifest(
        {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    """Solve one explicitly declared configuration and write complete field provenance."""
    before = source_hashes()
    mesh = crisscross_mesh(args.nx, args.ny)
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, args.segments) for _ in mesh.faces), 2
    )
    top = {int(f): (0.0, 0.0) for f in mesh.boundary_faces if mesh.normals[f, 1] > 0.5}
    side = {int(f): {1: 0.0} for f in mesh.boundary_faces if abs(mesh.normals[f, 0]) > 0.5}
    resistance = layer_resistance()
    minimum = (
        (float(np.min(resistance.values)) if args.gamma_min is None else args.gamma_min)
        if args.stabilization == "minimum-2017"
        else None
    )
    start = perf_counter()
    with threadpool_limits(limits=1):
        solution = solve_brinkman(
            mesh,
            viscosity=0.3,
            drag=resistance,
            source=0.0,
            dirichlet=inlet_velocity,
            traction=top,
            traction_components=side,
            skeleton=skeleton,
            local_refinement=args.refinement,
            degree=args.degree,
            formulation="usfem",
            stabilization=args.stabilization,
            gamma_min=args.gamma_min,
            quadrature_order=args.order,
            backend=args.backend,
            workers=args.workers,
            solver=args.solver,
            local_solver=args.local_solver,
        )
        solved = perf_counter()
        arrays = archive_fields(solution)
        diagnostics = physical_diagnostics(solution, args.refinement)
    after = source_hashes()
    stem = (
        f"flow-layer1-n{args.nx}x{args.ny}-p{args.degree}"
        f"-r{args.refinement}-s{args.segments}-q{args.order}-{args.stabilization}"
    )
    path = args.output / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    prescribed = sum(skeleton.faces[f].size * 2 for f in top) + sum(
        skeleton.faces[f].size for f in side
    )
    report = dict(
        archive=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layer_archive="layer-1.npz",
        layer_sha256=hashlib.sha256((OUTPUT / "layer-1.npz").read_bytes()).hexdigest(),
        macro_shape=[args.nx, args.ny],
        macro_cells=len(mesh.cells),
        macro_geometry="crisscross",
        domain=[0, 1200, 0, 2200],
        coefficient="scalar gamma=0.3/Kx; viscosity=0.3",
        coefficient_units="source mD numbers and ft coordinates; no implicit SI conversion",
        velocity_degree=args.degree,
        pressure_degree=args.degree,
        local_refinement=args.refinement,
        local_unknowns=3 * len(solution.values[0]),
        skeleton_degree=1,
        skeleton_segments=args.segments,
        skeleton_continuous=False,
        total_skeleton_unknowns=skeleton.size,
        prescribed_traction_unknowns=prescribed,
        retained_unknowns=sum(len(field) for field in solution.hybrid.coarse),
        free_trace_plus_retained=skeleton.size
        - prescribed
        + sum(len(field) for field in solution.hybrid.coarse),
        quadrature_order=max(args.order, args.degree + 2),
        quadrature_requested=args.order,
        integration=(
            "geometric pixel-triangle intersections; positive Gaussian rules per subtriangle"
        ),
        stabilization=args.stabilization,
        gamma_min=minimum,
        stabilization_resistance_bound=(
            "global minimum eigenvalue over the archived Cartesian field"
            if args.stabilization == "minimum-2017" and args.gamma_min is None
            else "explicit supplied gamma_min"
            if args.stabilization == "minimum-2017"
            else "pointwise smallest eigenvalue on each pixel intersection"
            if args.stabilization == "pointwise-2017"
            else "maximum eigenvalue over positive-area pixels in each fine triangle"
        ),
        boundary="bottom u=(0,1); sides ux=0 and ty=0; top full outward pseudotraction=0",
        pseudotraction="(viscosity*grad(u)-p*I)n; hybrid multiplier has opposite sign",
        profile=(
            "x=199; separate actual macro intervals with both interface limits; "
            "independently chosen diagnostic"
        ),
        residual=solution.hybrid.residual,
        **diagnostics,
        velocity_nodal_min=np.min(arrays["local_velocity"], axis=(0, 1)).tolist(),
        velocity_nodal_max=np.max(arrays["local_velocity"], axis=(0, 1)).tolist(),
        velocity_magnitude_nodal_max=float(np.linalg.norm(arrays["local_velocity"], axis=-1).max()),
        pressure_nodal_min=float(arrays["local_pressure"].min()),
        pressure_nodal_max=float(arrays["local_pressure"].max()),
        solve_seconds=solved - start,
        postprocessing_seconds=perf_counter() - solved,
        timing_scope="observed wall time; not an exclusive performance benchmark",
        backend=args.backend,
        workers=args.workers,
        global_solver=args.solver,
        local_solver=args.local_solver,
        local_native_threads=1,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_hashes=before,
        source_hashes_after=after,
        source_changed=[name for name in before if before[name] != after[name]],
        reproduction_limit={
            "pointwise-2017": (
                "The 2017 formula is evaluated pointwise on each material pixel; "
                "its historical spatial evaluation convention was not reported. "
                "Material intersections are integrated geometrically without changing the spaces."
            ),
            "minimum-2017": (
                "The 2017 formula uses an explicitly declared global minimum eigenvalue. "
                "The historical spatial convention was not reported; this choice can lose "
                "coercivity in a high-contrast material despite a small algebraic residual."
            ),
            "tensor-2025": (
                "This uses the 2025 elementwise maximum-eigenvalue stabilization, "
                "a separate convention from the 2017 formula; material cuts are resolved."
            ),
        }[args.stabilization],
        figure_reference=(
            "Araya, Harder, Poza and Valentin (2017), DOI 10.1016/j.cma.2017.05.027, Figures 18–20"
        ),
    )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Parse acquisition settings without silently overwriting an existing result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=11)
    parser.add_argument("--refinement", type=int, default=10)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument(
        "--stabilization",
        choices=["minimum-2017", "pointwise-2017", "tensor-2025"],
        default="pointwise-2017",
    )
    parser.add_argument("--gamma-min", type=float)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=["serial", "thread", "process"], default="process")
    parser.add_argument("--solver", default="scipy")
    parser.add_argument("--local-solver", default="scipy")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (
        min(args.nx, args.ny, args.refinement, args.segments, args.degree, args.order, args.workers)
        < 1
    ):
        parser.error("mesh sizes, polynomial degree, quadrature order and workers must be positive")
    stem = (
        f"flow-layer1-n{args.nx}x{args.ny}-p{args.degree}"
        f"-r{args.refinement}-s{args.segments}-q{args.order}-{args.stabilization}"
    )
    if (args.output / f"{stem}.json").exists() or (args.output / f"{stem}.npz").exists():
        parser.error("result exists; choose a different output directory to preserve provenance")
    print(json.dumps(run_case(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
