"""Acquire affine tetrahedral/prismatic mixed MHM on the same octagonal well domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from solve_mapped_well import WellData
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.hdiv_3d import solve_darcy_hdiv3d
from pymhm.fem.hdiv.family_3d import cell_quadrature
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.mixed import AffineMixedMesh

ROOT = Path(__file__).resolve().parents[1]


def acquire(
    kind: str, pressure_degree: int, fine_factor: int, macro_factor: int, workers: int = 1
) -> dict:
    """Solve one declared mixed space and archive physical fields and exact-volume norms."""
    if fine_factor % macro_factor:
        raise ValueError("macro factor must divide the fine factor")
    data = WellData()
    hexa = HexMesh.annular_prism(
        np.geomspace(data.inner_radius, data.outer_radius, 5), data.height, 8
    )
    base = AffineMixedMesh.from_extruded_hexahedra(hexa.points, hexa.cells, kind)
    mesh = base.refined(macro_factor)
    neumann = {
        int(f): 0.0 for f in mesh.boundary_faces if np.ptp(mesh.points[mesh.faces[f], 2]) < 1e-12
    }
    watched = [
        ROOT / f"src/pymhm/{name}"
        for name in (
            "fem/hdiv/family_3d.py",
            "meshes/mixed.py",
            "_legacy/models/darcy/hdiv_3d.py",
            "core/contracts.py",
            "linalg/linear.py",
        )
    ] + [Path(__file__), ROOT / "examples/solve_mapped_well.py"]
    hashes = current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in watched
        }
    )
    started = time.perf_counter()
    with threadpool_limits(1):
        solution = solve_darcy_hdiv3d(
            mesh,
            pressure_degree=pressure_degree,
            local_refinement=fine_factor // macro_factor,
            permeability=data.permeability / data.viscosity,
            dirichlet=data.pressure,
            neumann=neumann,
            quadrature_order=5,
            boundary_quadrature_order=12,
            backend="process" if workers > 1 else "serial",
            workers=workers,
            global_rtol=1e-12,
            global_refinement_precision="extended",
        )
        elapsed = time.perf_counter() - started
        errors = {
            str(order): solution.errors(data.pressure, data.flux, order=order) for order in (7, 10)
        }
        x, w = cell_quadrature(kind, 10)
        pnorm = qnorm = 0.0
        for fine in solution.local_meshes:
            physical = fine.geometry(x)
            p = data.pressure(physical.reshape(-1, 3)).reshape(physical.shape[:2])
            q = data.flux(physical.reshape(-1, 3)).reshape(physical.shape)
            pnorm += np.sum(fine.determinants[:, None] * w * p * p)
            qnorm += np.sum(fine.determinants[:, None] * w * np.sum(q * q, axis=2))
        balance = max(float(np.max(abs(v))) for v in solution.equilibrium_residuals())
    current = current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in watched
        }
    )
    if current != hashes:
        raise RuntimeError("acquisition source changed during the solve")
    directory = ROOT / "examples/results/mixed-well-geometries"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{kind}-p{pressure_degree}-fine{fine_factor}-macro{macro_factor}"
    archive = directory / f"{name}.npz"
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.array([fine.points for fine in solution.local_meshes]),
        local_cells=np.array([fine.cells for fine in solution.local_meshes]),
        pressure=np.array(solution.pressure),
        flux=np.array(solution.flux),
        trace=solution.hybrid.trace,
        flux_basis_coefficients=solution.family.coefficients,
    )
    result = dict(
        kind=kind,
        archive_schema=2,
        flux_basis_convention=(
            "Canonical face moments; fixed ordered interior monomial moments, "
            "followed by positive-diagonal Cholesky orthonormalization. "
            "The executed component-monomial matrix is stored in the archive."
        ),
        flux_basis_sha256=hashlib.sha256(solution.family.coefficients.tobytes()).hexdigest(),
        interior_moment_seeds=solution.family.interior_moment_seeds,
        pressure_degree=pressure_degree,
        normal_degree=1,
        fine_factor=fine_factor,
        macro_factor=macro_factor,
        macrocells=len(mesh.cells),
        fine_cells=sum(len(fine.cells) for fine in solution.local_meshes),
        skeleton_dofs=solution.skeleton.size,
        retained_dofs=len(mesh.cells),
        errors=errors,
        relative_errors=dict(
            pressure_l2=errors["10"]["pressure_l2"] / np.sqrt(pnorm),
            flux_l2=errors["10"]["flux_l2"] / np.sqrt(qnorm),
        ),
        exact_norms=dict(pressure_l2=np.sqrt(pnorm), flux_l2=np.sqrt(qnorm)),
        physical_block_residual=float(solution.physical_residuals.max()),
        fine_pressure_moment_defect=balance,
        volume=float(sum(mesh.volumes)),
        permeability_over_viscosity=data.permeability / data.viscosity,
        volume_quadrature_order=5,
        boundary_quadrature_order=12,
        error_quadrature_orders=[7, 10],
        geometry="Common planar octagonal annulus; its boundary is preserved exactly.",
        boundary=(
            "Exact radial pressure on both polygonal walls; zero outward flux on horizontal caps."
        ),
        formulation="Mixed H(div) local solvers with coarse normal-flux moment restriction.",
        comparison_scope=(
            "Original geometry hierarchy for Problem 4; "
            "historical meshes and error ratios are not reproduced."
        ),
        elapsed_s=elapsed,
        timing_scope="Concurrent scientific acquisition, not a controlled speed measurement.",
        source_hashes=hashes,
        archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    (directory / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run one reproducible case, keeping memory proportional to its declared local meshes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["tetrahedron", "prism"], required=True)
    parser.add_argument("--pressure-degree", type=int, default=1)
    parser.add_argument("--fine-factor", type=int, default=2)
    parser.add_argument("--macro-factor", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = acquire(
        args.kind, args.pressure_degree, args.fine_factor, args.macro_factor, args.workers
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
