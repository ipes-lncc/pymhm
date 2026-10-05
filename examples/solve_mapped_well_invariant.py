"""Solve the exact vertically invariant RT1 subspace for the oscillatory producing well.

The three-dimensional geometry and physical integrals are retained. Only modes
with zero coefficients by the stated vertical symmetry are removed. This is a
classical conforming mixed method, without a macro-skeleton restriction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from examples.solve_mapped_oscillatory_well import OUTPUT, ROOT, OscillatoryWellData
from pymhm.fem.hdiv.mapped import mapped_rt_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.linalg.linear import factorize
from pymhm.meshes.hexahedron import HexMesh, _geometry, cube_quadrature

FLUX_MODES = np.array([0, 2, 4, 6, 8, 10, 12, 14, 24, 26, 28, 30])
PRESSURE_MODES = np.array([0, 2, 4, 6])


def assemble(mesh: HexMesh, data: OscillatoryWellData, order: int) -> tuple:
    """Assemble horizontal Piola RT1 and z-constant Q1 on the extruded annular mesh."""
    transforms = mesh.face_transforms[:, :4]
    expected = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    if not np.allclose(transforms, expected, rtol=0, atol=2e-14):
        raise ValueError("the invariant annular assembly requires the declared face coordinates")
    vertices = mesh.points[mesh.cells]
    if not np.allclose(vertices[:, 1::2, :2], vertices[:, ::2, :2], rtol=0, atol=1e-13):
        raise ValueError("the invariant geometry must be a vertical extrusion")
    if np.any(np.ptp(vertices[:, ::2, 2], axis=1) > 1e-13) or np.any(
        np.ptp(vertices[:, 1::2, 2], axis=1) > 1e-13
    ):
        raise ValueError("the invariant geometry requires horizontal planar caps")
    count = len(mesh.cells)
    old = np.column_stack(
        (
            (4 * mesh.cell_faces[:, :4, None] + np.array([0, 2])).reshape(count, 8),
            4 * len(mesh.faces) + 12 * np.arange(count)[:, None] + np.array([0, 2, 4, 6]),
        )
    )
    retained, inverse_ids = np.unique(old, return_inverse=True)
    ids = inverse_ids.reshape(count, 12)
    nq = len(retained)
    signs = np.column_stack((np.repeat(mesh.signs[:, :4], 2, axis=1), np.ones((count, 4))))
    xy, weights = cube_quadrature(order, 2)
    points = np.column_stack((xy, np.full(len(xy), 0.5)))
    unit, div, pressure = mapped_rt_basis(HexMesh.unit_cube(), 1, points)
    reference = unit[0][:, FLUX_MODES]
    tests = pressure[:, PRESSURE_MODES]
    divergence = np.einsum("q,qi,qj->ij", weights, tests, div[0][:, FLUX_MODES])
    masses = np.empty((count, 12, 12))
    moments = np.empty((count, 4))
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        physical, jacobian, determinant = _geometry(vertices[start:stop], points)
        basis = np.einsum("tqab,qib->tqia", jacobian, reference) / determinant[..., None, None]
        tensor = data.tensor(physical.reshape(-1, 3)).reshape(*determinant.shape, 3, 3)
        # This experiment has the explicitly diagonal OscillatoryWellData tensor.
        # Reciprocal diagonal entries implement its exact inverse without batched LU.
        diagonal = np.diagonal(tensor, axis1=-2, axis2=-1)
        if not np.array_equal(tensor, diagonal[..., :, None] * np.eye(3)):
            raise ValueError("the invariant well experiment requires its diagonal tensor")
        weighted = (determinant * weights)[..., None, None] * basis / diagonal[..., None, :]
        masses[start:stop] = np.einsum("tqia,tqja->tij", basis, weighted, optimize=True)
        moments[start:stop] = np.einsum("tq,qi->ti", determinant * weights, tests)
    masses *= signs[:, :, None] * signs[:, None, :]
    rows = np.repeat(ids, 12, axis=1).ravel()
    columns = np.tile(ids, (1, 12)).ravel()
    mass = sparse.coo_matrix((masses.ravel(), (rows, columns)), shape=(nq, nq)).tocsr()
    pids = np.arange(4 * count).reshape(count, 4)
    div_data = divergence[None] * signs[:, None]
    derivative = sparse.coo_matrix(
        (div_data.ravel(), (np.repeat(pids, 12, axis=1).ravel(), np.tile(ids, (1, 4)).ravel())),
        shape=(4 * count, nq),
    ).tocsr()
    load = np.zeros(nq)
    uv, face_weights = cube_quadrature(order, 2)
    face_basis = np.column_stack((np.ones(len(uv)), 3 * (2 * uv[:, 0] - 1)))
    mapping = {int(old_id): new_id for new_id, old_id in enumerate(retained)}
    for face in mesh.boundary_faces:
        cell, side = mesh.incidence[face][0]
        if side >= 4:
            continue
        axis, end = divmod(side, 2)
        ref = np.empty((len(uv), 3))
        ref[:, axis], ref[:, np.arange(3) != axis] = end, uv
        physical = _geometry(vertices[cell : cell + 1], ref)[0][0]
        boundary = -face_basis.T @ (face_weights * data.pressure(physical))
        load[[mapping[int(4 * face + mode)] for mode in (0, 2)]] += boundary
    operator = sparse.bmat([[mass, -derivative.T], [-derivative, None]], format="csr")
    flux_scale = 1 / np.sqrt(mass.diagonal())
    pressure_scale = 1 / np.sqrt(np.asarray(derivative.power(2) @ flux_scale**2))
    scaling = np.r_[flux_scale, pressure_scale]
    return operator, np.r_[load, np.zeros(4 * count)], scaling, ids, signs, divergence, moments


def solve(
    mesh: HexMesh,
    data: OscillatoryWellData,
    order: int,
    *,
    solver: str = "scipy",
    rtol: float = 1e-17,
    refinement_steps: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolve the congruently scaled saddle and check the original physical equations."""
    started = time.perf_counter()
    matrix, rhs, scaling, ids, signs, divergence, moments = assemble(mesh, data, order)
    assembled = time.perf_counter()
    transform = sparse.diags(scaling)
    balanced = (transform @ matrix @ transform).tocsr()
    with factorize(balanced, solver=solver, rtol=rtol) as decomposition:
        solution = scaling.astype(np.longdouble) * decomposition.solve(
            scaling * rhs,
            refinement_precision="extended",
            refinement_steps=refinement_steps,
        )
    nq = int(ids.max()) + 1
    pressure = np.zeros((len(mesh.cells), 8), dtype=np.longdouble)
    pressure[:, PRESSURE_MODES] = solution[nq:].reshape(-1, 4)
    flux = np.zeros((len(mesh.cells), 36), dtype=np.longdouble)
    flux[:, FLUX_MODES] = solution[:nq][ids] * signs
    defect = matrix.astype(np.longdouble) @ solution - rhs
    action = abs(matrix).astype(np.longdouble) @ abs(solution) + abs(rhs)
    block = [
        float(np.linalg.norm(defect[part]) / np.linalg.norm(action[part]))
        for part in (slice(0, nq), slice(nq, None))
    ]
    balances = flux[:, FLUX_MODES] @ divergence.astype(np.longdouble).T
    local_action = abs(flux[:, FLUX_MODES]) @ abs(divergence.astype(np.longdouble)).T
    relative = np.linalg.norm(balances, axis=1) / np.linalg.norm(local_action, axis=1)
    if max(*block, float(relative.max())) > 1e-10:
        raise ValueError("the original mixed/divergence equations exceed their physical criterion")
    # Archival float64 coefficients must independently retain the physical balance.
    stored_flux, stored_pressure = np.asarray(flux, float), np.asarray(pressure, float)
    archived = np.asarray(solution, float).astype(np.longdouble)
    archived_defect = matrix.astype(np.longdouble) @ archived - rhs
    archived_action = abs(matrix).astype(np.longdouble) @ abs(archived) + abs(rhs)
    archived_block = [
        float(np.linalg.norm(archived_defect[part]) / np.linalg.norm(archived_action[part]))
        for part in (slice(0, nq), slice(nq, None))
    ]
    if max(archived_block) > 1e-10:
        raise ValueError("archival coefficients do not retain the physical mixed criterion")
    stored_balances = stored_flux[:, FLUX_MODES].astype(np.longdouble) @ divergence.T
    stored_relative = np.linalg.norm(stored_balances, axis=1) / np.linalg.norm(local_action, axis=1)
    if float(stored_relative.max()) > 1e-10:
        raise ValueError("archival coefficients do not retain the physical divergence criterion")
    rates = dict(inner=0.0, outer=0.0, top_bottom=0.0)
    for face in mesh.boundary_faces:
        cell, side = mesh.incidence[face][0]
        if side < 4:
            radius = np.linalg.norm(mesh.points[mesh.faces[face], :2], axis=1)
            label = "inner" if radius.max() < 1 else "outer"
            rates[label] += float(flux[cell, side * 4])
    diagnostics = dict(
        unknowns=len(rhs),
        flux_dofs=nq,
        pressure_dofs=len(rhs) - nq,
        physical_block_backward_residual_max=block,
        archived_physical_block_backward_residual_max=archived_block,
        divergence_cell_backward_residual_max=float(relative.max()),
        archived_divergence_cell_backward_residual_max=float(stored_relative.max()),
        divergence_moment_linf=float(abs(stored_balances).max()),
        pressure_volume_integral=float(np.sum(pressure[:, PRESSURE_MODES] * moments)),
        boundary_rates=rates,
        assembly_seconds=assembled - started,
        solve_and_diagnostics_seconds=time.perf_counter() - assembled,
        solver=solver,
        solve_rtol=rtol,
        refinement_precision="extended",
        refinement_steps_limit=refinement_steps,
        diagonal_congruence="unit flux diagonal and unit pressure/divergence row norm",
    )
    return stored_pressure, stored_flux, diagnostics


def main() -> None:
    """Acquire one native, vertically invariant classical mixed reference with source digests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-factor", type=int, default=16)
    parser.add_argument("--quadrature-xy", type=int, default=12)
    parser.add_argument("--solver", default="petsc")
    parser.add_argument("--rtol", type=float, default=1e-17)
    parser.add_argument("--refinement-steps", type=int, default=2)
    args = parser.parse_args()
    if args.fine_factor < 1 or args.quadrature_xy < 3:
        parser.error("positive refinement and at least three Gauss points are required")
    if args.refinement_steps < 0:
        parser.error("refinement steps must be nonnegative")
    data = OscillatoryWellData()
    radii = np.geomspace(data.inner_radius, data.outer_radius, 5)
    mesh = HexMesh.annular_prism(radii, data.height, 8).refined(
        (args.fine_factor, args.fine_factor, 1)
    )
    paths = [
        Path(__file__),
        ROOT / "examples/solve_mapped_oscillatory_well.py",
        ROOT / "examples/solve_mapped_well.py",
        ROOT / "src/pymhm/_legacy/models/darcy/mapped.py",
        ROOT / "src/pymhm/linalg/linear.py",
    ]
    hashes = current_source_manifest(
        {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )
    snapshots = ROOT / "build/source-snapshots/mapped-well-oscillatory"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in paths:
        (snapshots / f"{hashes[path.relative_to(ROOT).as_posix()]}-{path.name}").write_bytes(
            path.read_bytes()
        )
    with threadpool_limits(1):
        pressure, flux, report = solve(
            mesh,
            data,
            args.quadrature_xy,
            solver=args.solver,
            rtol=args.rtol,
            refinement_steps=args.refinement_steps,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"classical-invariant-xy{args.fine_factor}-z1-q{args.quadrature_xy}"
    archive = OUTPUT / (name + ".npz")
    np.savez_compressed(
        archive,
        vertices=mesh.points[mesh.cells],
        pressure=pressure,
        flux=flux,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
    )
    report.update(
        case="Classical conforming RT1/Q1 restricted to the exact vertically invariant subspace",
        archive_layout="canonical-reference-cell",
        fine_factor=args.fine_factor,
        vertical_factor=1,
        fine_cells=len(mesh.cells),
        degree=1,
        quadrature=[args.quadrature_xy, args.quadrature_xy, 1],
        symmetry=(
            "Extruded geometry; block-diagonal K with Kxy(x,y), Kz(z); lateral p(x,y), "
            "impermeable caps and zero source."
        ),
        vertical_modes=(
            "Removed z-dependent pressure, horizontal-flux modes and vertical flux; "
            "full 3D Piola geometry and integrals retained."
        ),
        physical_parameters=data.__dict__,
        radii=radii.tolist(),
        units="m, Pa, s; physical flux m/s",
        archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_hashes=hashes,
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.relative_to(ROOT).as_posix()]
        for path in paths
    ):
        raise RuntimeError("Acquisition sources changed; inspect the preserved source snapshots")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
