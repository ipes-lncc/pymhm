"""Run declared quadrilateral MHM resolutions for SPE10 Model 2, layer 36.

The 6-by-11 macrogrid and continuous-P1 face spaces follow Paredes et al.
(JCAM 436, 115415, Section 5.2). The historical local refinement was not
reported. This program therefore records and varies it independently.
Run from the checkout with ``pixi run python -m examples.solve_spe10`` after
creating the compact layer archive with ``examples/plot_spe10_data.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.hybrid import HybridSolution
from pymhm.mesh import FaceSpace, SkeletonSpace
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    QuadrilateralDarcySolution,
    solve_darcy_quadrilateral,
)
from pymhm.reservoir import CartesianCellField

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/spe10"


def load_layer(number: int = 36) -> CartesianCellField:
    """Load unchanged horizontal tensor values from the compact pinned archive."""
    with np.load(OUTPUT / f"layer-{number}.npz") as data:
        horizontal = data["permeability"][..., :2]
        tensor = np.zeros(horizontal.shape[:2] + (2, 2))
        tensor[..., 0, 0] = horizontal[..., 0]
        tensor[..., 1, 1] = horizontal[..., 1]
        spacing = tuple(data["spacing"])
    return CartesianCellField(tensor, spacing)


def pressure_boundary(points: np.ndarray) -> np.ndarray:
    """Return one on the bottom and zero on the top of the physical rectangle."""
    return 1 - points[:, 1] / 2200


def sample_solution(solution: QuadrilateralDarcySolution) -> dict[str, np.ndarray]:
    """Preserve pixel-center fields and distinct profile values on each macrocell.

    The profile x=199 is stored in eleven separate y intervals, including
    duplicated interface positions. Plotters must not average these one-sided
    pressure or flux values. Full Qk nodal coefficients are archived separately.
    """
    mesh = solution.skeleton.mesh
    centers = np.array(
        [(x, y) for x in np.arange(10, 1200, 20) for y in np.arange(5, 2200, 10)], dtype=float
    )
    owners = (centers[:, 1] / 200).astype(int) * 6 + (centers[:, 0] / 200).astype(int)
    pressure = np.empty(len(centers))
    flux = np.empty((len(centers), 2))
    for cell in range(len(mesh.cells)):
        mask = owners == cell
        pressure[mask], flux[mask] = solution.evaluate(cell, centers[mask])
    profile_points = np.array(
        [
            np.column_stack((np.full(201, 199.0), np.linspace(row * 200, (row + 1) * 200, 201)))
            for row in range(11)
        ]
    )
    profile_pressure, profile_flux = [], []
    for row, points in enumerate(profile_points):
        p, q = solution.evaluate(row * 6, points)
        profile_pressure.append(p)
        profile_flux.append(q)
    return dict(
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        points=centers.reshape(60, 220, 2),
        pressure=pressure.reshape(60, 220),
        flux=flux.reshape(60, 220, 2),
        owner=owners.reshape(60, 220),
        local_pressure=np.asarray(solution.pressure),
        profile_points=profile_points,
        profile_pressure=np.asarray(profile_pressure),
        profile_flux=np.asarray(profile_flux),
        profile_owner=np.arange(11) * 6,
        trace=solution.hybrid.trace,
        coarse=np.asarray(solution.hybrid.coarse),
        macro_balance=solution.conservation_residuals(),
    )


def load_solution(record: dict) -> QuadrilateralDarcySolution:
    """Restore archived nodal coefficients for sampling without another PDE solve."""
    with np.load(OUTPUT / record["archive"]) as arrays:
        mesh = CartesianMacroMesh(6, 11, tuple(record["domain"]))
        skeleton = SkeletonSpace(
            mesh,
            tuple(
                FaceSpace.uniform(1, record["skeleton_segments"], continuous=True)
                for _ in mesh.faces
            ),
        )
        local_meshes = tuple(
            mesh.submesh(cell, tuple(record["local_refinement"])) for cell in range(len(mesh.cells))
        )
        pressure = tuple(arrays["local_pressure"])
        hybrid = HybridSolution(
            arrays["trace"], tuple(arrays["coarse"]), pressure, record["residual"], np.empty(0)
        )
    return QuadrilateralDarcySolution(
        skeleton,
        local_meshes,
        pressure,
        (),
        hybrid,
        load_layer(),
        0.0,
        record["degree"],
        record["quadrature_order"],
    )


def run_case(
    refinement: int,
    segments: int,
    *,
    workers: int,
    backend: str,
    local_solver: str,
    solver: str,
    degree: int = 1,
    order: int = 3,
) -> dict:
    """Solve one declared local/trace configuration and archive its physical fields."""
    mesh = CartesianMacroMesh(6, 11, (0, 1200, 0, 2200))
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, segments, continuous=True) for _ in mesh.faces)
    )
    neumann = {int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) > 0.5}
    start = perf_counter()
    with threadpool_limits(limits=1):
        solution = solve_darcy_quadrilateral(
            mesh,
            permeability=load_layer(),
            source=0.0,
            dirichlet=pressure_boundary,
            neumann=neumann,
            skeleton=skeleton,
            local_refinement=refinement,
            degree=degree,
            quadrature_order=order,
            backend=backend,
            workers=workers,
            solver=solver,
            local_solver=local_solver,
        )
        solved = perf_counter()
        arrays = sample_solution(solution)
    sampled = perf_counter()
    stem = f"darcy-q{degree}-r{refinement}-s{segments}"
    path = OUTPUT / f"{stem}.npz"
    np.savez_compressed(path, **arrays)
    raw_min = float(min(values.min() for values in solution.pressure))
    raw_max = float(max(values.max() for values in solution.pressure))
    exterior_flux = np.zeros(4)
    for face in mesh.boundary_faces:
        t, w = skeleton.faces[face].quadrature(3)
        value = skeleton.faces[face].evaluate(t) @ solution.hybrid.trace[skeleton.dofs(int(face))]
        normal = mesh.normals[face]
        side = 0 if normal[1] < -0.5 else 1 if normal[0] > 0.5 else 2 if normal[1] > 0.5 else 3
        exterior_flux[side] += mesh.lengths[face] * (w @ value)
    report = dict(
        archive=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layer=36,
        macro_shape=[6, 11],
        macro_cells=66,
        domain=[0, 1200, 0, 2200],
        permeability_unit="mD",
        coordinate_unit="ft",
        degree=degree,
        local_refinement=[refinement, refinement],
        local_unknowns=(refinement * degree + 1) ** 2,
        skeleton_degree=1,
        skeleton_segments=segments,
        skeleton_continuous=True,
        total_skeleton_unknowns=skeleton.size,
        prescribed_flux_unknowns=sum(skeleton.faces[f].size for f in neumann),
        free_trace_plus_retained=skeleton.size - sum(skeleton.faces[f].size for f in neumann) + 66,
        quadrature_order=solution.quadrature_order,
        material_integration="exact Cartesian pixel intersections; aligned-grid fast path",
        coefficient_pixels_aligned=refinement % 20 == 0,
        residual=solution.hybrid.residual,
        macro_balance_max=float(np.max(np.abs(arrays["macro_balance"]))),
        exterior_flux_bottom_right_top_left=exterior_flux.tolist(),
        pressure_nodal_min=raw_min,
        pressure_nodal_max=raw_max,
        solve_seconds=solved - start,
        sample_seconds=sampled - solved,
        timing_scope="observed wall time, no exclusive performance benchmark",
        backend=backend,
        workers=workers,
        local_solver=local_solver,
        global_solver=solver,
        local_native_threads=1,
        figure_reference=(
            "Paredes, Valentin and Versieux, DOI 10.1016/j.cam.2023.115415, Figures 6–8"
        ),
        reproduction_limit=(
            "Published MHM local refinement is unspecified; this run states its own refinement."
        ),
        profile_convention="x=199; 11 separate macro intervals with both interface limits retained",
        pressure_rendering="unaveraged local Qk coefficients and pixel-center samples",
    )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run requested cases, preserving existing archives unless explicitly forced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements", type=int, nargs="+", default=[40, 60, 80, 100, 120])
    parser.add_argument("--segments", type=int, nargs="+", default=[32])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=["serial", "thread", "process"], default="process")
    parser.add_argument("--solver", default="scipy")
    parser.add_argument("--local-solver", default="scipy")
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {
        path.name: json.loads(path.read_text()) for path in OUTPUT.glob("darcy-q*-r*-s*.json")
    }
    for refinement in args.refinements:
        for segments in args.segments:
            path = OUTPUT / f"darcy-q{args.degree}-r{refinement}-s{segments}.json"
            if path.exists() and not args.force:
                record = json.loads(path.read_text())
                archive = OUTPUT / record["archive"]
                if hashlib.sha256(archive.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Archive checksum differs from {path.name}")
            else:
                print(
                    f"Start Q{args.degree} refinement={refinement}, face segments={segments}",
                    flush=True,
                )
                record = run_case(
                    refinement,
                    segments,
                    workers=args.workers,
                    backend=args.backend,
                    local_solver=args.local_solver,
                    solver=args.solver,
                    degree=args.degree,
                    order=args.order,
                )
            records[path.name] = record
            print(json.dumps(record), flush=True)
            (OUTPUT / "darcy-campaign.json").write_text(
                json.dumps(dict(rows=[records[name] for name in sorted(records)]), indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
