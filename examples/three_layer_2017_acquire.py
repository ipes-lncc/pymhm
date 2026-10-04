"""Acquire portable selected three-layer inputs and original local MHM operators.

Default preparation performs no FEM assembly. Explicit local acquisition preserves
the P3/r8, vector P2/s8 spaces and original PyMHM mass/elasticity/load owners. A
completed local archive does not establish a global trajectory, reference accuracy
or reproduction of historical figures. Full acquisition and trajectory execution
require separately measured resource budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag, cholesky, solve_triangular
from threadpoolctl import threadpool_limits

from examples.campaign_provenance import file_digest, require_equal, verify_archive
from examples.three_layer_2017 import DATA, ThreeLayerCase, assembly_order, load_case
from examples.three_layer_field_archive import (
    actual_local_tables,
    capture_field_basis,
    product_recipe,
)
from pymhm import ElastodynamicSolution, ElastodynamicStepper, TriangleMesh
from pymhm.elasticity import _rigid
from pymhm.elastodynamics import ElastodynamicLocal, _make_local
from pymhm.mesh import FaceSpace, SkeletonSpace
from pymhm.solvers import factorize

ROOT = Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    """Identify actual imported numerical owners and the acquisition/input/lock bytes."""
    import pymhm.elastodynamics as owner

    package = Path(owner.__file__).parent
    files = {f"pymhm/{path.relative_to(package)}": path for path in sorted(package.rglob("*.py"))}
    files.update(
        {
            f"examples/{name}.py": ROOT / f"examples/{name}.py"
            for name in (
                "three_layer_2017",
                "three_layer_2017_acquire",
                "three_layer_field_archive",
                "campaign_provenance",
            )
        }
    )
    files["pixi.lock"] = ROOT / "pixi.lock"
    return {name: file_digest(path) for name, path in files.items()}


def _write(path: Path, values: dict[str, Any]) -> None:
    """Expose complete metadata atomically after its payload is durable."""
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(values, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save(path: Path, **arrays: Any) -> None:
    """Persist only non-pickle numeric arrays before their checkpoint metadata."""
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save_field_basis(path: Path, local: ElastodynamicLocal) -> dict[str, Any]:
    """Bind a numeric field archive to the producer's literal basis and executed tables."""
    record, arrays = capture_field_basis(local)
    _save(path, **arrays)
    metadata = path.with_suffix(".json")
    record.update(archive=path.name, archive_sha256=file_digest(path))
    _write(metadata, record)
    return {
        "schema": record["schema"],
        "archive": path.name,
        "archive_sha256": record["archive_sha256"],
        "metadata": metadata.name,
        "metadata_sha256": file_digest(metadata),
    }


def _verify_field_archives(directory: Path, fields: dict[str, dict[str, Any]]) -> None:
    """Require every immutable producer table and recipe to keep its executed digest."""
    for field in fields.values():
        verify_archive(directory / field["archive"], field["archive_sha256"])
        verify_archive(directory / field["metadata"], field["metadata_sha256"])


def _bind(directory: Path, identity: dict[str, Any]) -> None:
    """Reject changed executed sources, data or discretization before a completed skip."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.exists():
        require_equal(
            json.loads(path.read_text()), identity, label="three-layer acquisition identity"
        )
    else:
        _write(path, identity)


def _unchanged(case: ThreeLayerCase, before: dict[str, str]) -> None:
    """Reject input/source mutations without rewriting earlier acquired provenance."""
    case.verify_inputs()
    require_equal(source_hashes(), before, label="executed sources during acquisition")


def preflight(case: ThreeLayerCase) -> dict[str, Any]:
    """Report selected geometry and honest storage estimates without assembling any operator."""
    skeleton = case.skeleton()
    length = case.scales[0]
    diameters = (
        np.linalg.norm(
            case.mesh.points[case.mesh.faces[:, 1]] - case.mesh.points[case.mesh.faces[:, 0]],
            axis=1,
        )
        * length
    )
    return {
        **case.metadata(),
        "status": "physical_inputs_validated; no FEM assembly or trajectory executed",
        "macro_cells": len(case.mesh.cells),
        "macro_nodes": len(case.mesh.points),
        "macro_faces": len(case.mesh.faces),
        "global_trace_dofs": skeleton.size,
        "local_degree": 3,
        "local_refinement": 8,
        "local_scalar_P3_nodes": 325,
        "local_vector_P3_dofs": 650,
        "trace_degree": 2,
        "trace_segments": 8,
        "trace_pairing_boundary_argument": boundary_propagation_diagnostic(
            case.mesh, tuple(int(face) for face in case.mesh.boundary_faces)
        ),
        "physical_macro_edge_min_max_m": [float(diameters.min()), float(diameters.max())],
        "estimated_two_resident_lift_arrays_bytes": len(case.mesh.cells) * 650 * 144 * 8 * 2,
        "estimated_u_v_301_states_bytes": len(case.mesh.cells) * 650 * 8 * 2 * 301,
        "resource_limits": (
            "Storage arithmetic only; cut quadrature, sparse factors, global fill and setup/time "
            "remain unmeasured. Local q5/q7 pilots and full setup/three-step measurements precede "
            "full 341-macro and 301-state campaigns."
        ),
        "classical_reference_pending": (
            "Same physical operator/source, conforming P3 h8/4/2 m and own "
            "spatial/time/load refinement"
        ),
    }


def prepare(case: ThreeLayerCase, directory: Path) -> dict[str, Any]:
    """Archive physical/material/trace input arrays without a numerical solve."""
    before = source_hashes()
    identity = {
        "schema": 1,
        "input_sha256": case.input_sha256,
        "source_sha256": before,
        "stage": "inputs",
    }
    _bind(directory, identity)
    record_path = directory / "inputs.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        verify_archive(directory / record["archive"], record["archive_sha256"])
        _unchanged(case, before)
        return record
    times = np.linspace(0, case.source_duration_s, 40001)
    density, stiffness = case.materials()
    skeleton = case.skeleton()
    archive = directory / "inputs.npz"
    _save(
        archive,
        macro_points=case.mesh.points,
        macro_cells=case.mesh.cells,
        macro_faces=case.mesh.faces,
        macro_signs=case.mesh.signs,
        trace_offsets=skeleton.offsets,
        trace_breaks=np.linspace(0, 1, 9),
        trace_degree=np.asarray(2),
        abscissae=case.abscissae,
        horizons=case.heights,
        density_kg_m3=case.density,
        lambda_Pa=case.lame_lambda,
        mu_Pa=case.lame_mu,
        nondimensional_density=density.values,
        nondimensional_kelvin_stiffness=stiffness.values,
        physical_time_s=times,
        wavelet=case.wavelet(times),
    )
    _unchanged(case, before)
    record = {
        **preflight(case),
        "source_sha256": before,
        "archive": archive.name,
        "archive_sha256": file_digest(archive),
        "schema": 1,
    }
    _write(record_path, record)
    return record


def _sparse_arrays(name: str, matrix: Any) -> dict[str, np.ndarray]:
    """Archive executed CSC storage without narrowing its coefficient dtype."""
    matrix = sparse.csc_matrix(matrix)
    return {
        f"{name}_{key}": value
        for key, value in {
            "data": matrix.data,
            "indices": matrix.indices,
            "indptr": matrix.indptr,
            "shape": np.asarray(matrix.shape, dtype=np.int64),
        }.items()
    }


def local_arrays(local: ElastodynamicLocal) -> dict[str, np.ndarray]:
    """Retain actual M/K/B, nodal basis geometry and signed trace injection maps.

    These arrays describe the executed interpolation coordinates used by a
    displacement/velocity vector, including every one-sided fine element.
    They permit native source/field evaluation without importing a pickle or
    reconstructing a basis from dimensions and hashes alone.
    """
    return {
        "local_points": local.mesh.points,
        "local_cells": local.mesh.cells,
        "local_nodes": local.nodes,
        "cell_dofs": local.dofs,
        "trace_dofs": local.trace_dofs,
        "coupling": local.coupling,
        "basis_degree": np.asarray(local.degree),
        **actual_local_tables(local),
        **product_recipe()[1],
        **_sparse_arrays("mass", local.mass),
        **_sparse_arrays("stiffness", local.stiffness),
    }


def operator_contract_digest(arrays: dict[str, np.ndarray]) -> str:
    """Bind coefficients to executed geometry/maps, operator and spatial-force bytes.

    Each sorted array contributes its name, exact dtype, shape and C-order bytes;
    no numerical conversion, tolerance or basis regeneration enters the digest.
    This local contract is checked again before every persisted physical state.
    """
    digest = hashlib.sha256()
    for name, raw in sorted(arrays.items()):
        values = np.asarray(raw)
        if values.dtype.hasobject or not np.isfinite(values).all():
            raise ValueError("operator contracts require finite non-object numeric arrays")
        header = json.dumps(
            {"name": name, "dtype": values.dtype.str, "shape": list(values.shape)}, sort_keys=True
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
    return digest.hexdigest()


def boundary_propagation_diagnostic(
    mesh: TriangleMesh, prescribed_faces: tuple[int, ...]
) -> dict[str, Any]:
    """Check the geometry needed to eliminate the two local P3/r8-P2/s8 trace modes.

    This is a conditional exact-space argument, not a numerical global rank or
    inf-sup estimate. Each local kernel restricts injectively to a full face.
    Fixing its exterior face coefficients therefore kills that macro's kernel
    amplitude; interior face continuity propagates zero along the macro graph.
    A connected component with no prescribed exterior face is excluded.
    """
    if (
        not prescribed_faces
        or len(set(prescribed_faces)) != len(prescribed_faces)
        or any(type(face) is not int for face in prescribed_faces)
        or not set(prescribed_faces).issubset(set(mesh.boundary_faces))
    ):
        raise ValueError("prescribed traction faces must be distinct exterior faces")
    neighbours: list[set[int]] = [set() for _ in mesh.cells]
    for first, second in mesh.face_cells:
        if second >= 0:
            neighbours[first].add(int(second))
            neighbours[second].add(int(first))
    seeds = {int(mesh.face_cells[face, 0]) for face in prescribed_faces}
    visited, pending = set(seeds), list(seeds)
    while pending:
        current = pending.pop()
        new = neighbours[current] - visited
        visited.update(new)
        pending.extend(new)
    if len(visited) != len(mesh.cells):
        raise ValueError("every macro component requires a prescribed exterior traction face")
    return {
        "macro_count": len(mesh.cells),
        "prescribed_exterior_faces": list(prescribed_faces),
        "prescribed_exterior_macros": len(seeds),
        "macros_reachable_from_prescribed_exterior": len(visited),
        "positive_affine_macro_areas": bool(np.all(mesh.areas > 0)),
        "local_vector_kernel_dimension": 2,
        "exact_space_argument": (
            "For continuous vector P3 on uniform r8 and independent vector P2/s8, "
            "each local closed 24-segment contour has two alternating endpoint modes. "
            "A zero prescribed full exterior face removes both amplitudes; face "
            "sharing propagates zero through every reachable macro. Positive mass "
            "makes the one-substep Newmark local response coercive."
        ),
        "global_numerical_rank_verified": False,
        "inf_sup_estimate_verified": False,
    }


def trace_pairing_diagnostic(
    local: ElastodynamicLocal, mesh: TriangleMesh, skeleton: SkeletonSpace, cell: int
) -> dict[str, Any]:
    """Measure the two expected local modes in physical mass/trace L2 coordinates.

    The executed mass and coupling remain unchanged. This selected-space check
    uses C=L_M^-1 B L_Gamma^-T, where M=L_M L_M^T and Gamma is the physical trace
    L2 Gram matrix evaluated by the declared face basis. SVD vectors are only
    diagnostic; no persisted field coefficients use them. A local Gram's tiny
    signed eigenvalues do not establish injectivity.
    """
    selected = FaceSpace.uniform(2, 8)
    if (
        type(cell) is not int
        or not 0 <= cell < len(mesh.cells)
        or skeleton.mesh is not mesh
        or skeleton.components != 2
        or local.degree != 3
        or len(local.mesh.cells) != 64
        or local.coupling.shape != (650, 144)
        or not np.array_equal(local.trace_dofs, skeleton.cell_dofs(cell))
        or any(skeleton.faces[face] != selected for face in mesh.cell_faces[cell])
    ):
        raise ValueError("trace diagnostic requires the executed vector P3/r8-P2/s8 pairing")
    blocks = []
    for face in mesh.cell_faces[cell]:
        space = skeleton.faces[face]
        points, weights = space.quadrature(4)
        basis = space.evaluate(points)
        scalar_gram = basis.T @ (weights[:, None] * basis) * mesh.lengths[face]
        blocks.append(np.kron(scalar_gram, np.eye(2)))
    trace_cholesky = cholesky(block_diag(*blocks), lower=True)
    mass_cholesky = cholesky(local.mass.toarray(), lower=True)
    scaled = solve_triangular(mass_cholesky, local.coupling, lower=True)
    scaled = solve_triangular(trace_cholesky, scaled.T, lower=True).T
    _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    relative_bound = float(128 * np.finfo(scaled.dtype).eps * max(scaled.shape))
    rank = int(np.count_nonzero(singular > relative_bound * singular[0]))
    kernel = right[-2:].T
    kernel_residual = float(
        np.linalg.norm(scaled @ kernel) / (np.linalg.norm(scaled) * np.linalg.norm(kernel))
    )
    restrictions = [
        float(np.linalg.svd(kernel[48 * side : 48 * (side + 1)], compute_uv=False)[-1])
        for side in range(3)
    ]
    if rank != 142 or kernel_residual > relative_bound or min(restrictions) <= relative_bound:
        raise RuntimeError("expected local pairing rank/kernel/face restriction check failed")
    return {
        "expected_rank": 142,
        "measured_mass_trace_L2_scaled_rank": rank,
        "local_vector_kernel_dimension": 2,
        "relative_roundoff_rank_threshold": relative_bound,
        "smallest_positive_to_largest_singular_value": float(singular[-3] / singular[0]),
        "two_null_to_largest_singular_values": (singular[-2:] / singular[0]).tolist(),
        "mass_trace_L2_scaled_kernel_relative_residual": kernel_residual,
        "kernel_full_face_restriction_minimum_singular_values": restrictions,
        "normalization": "C=L_M^-1 B L_Gamma^-T; Gamma uses actual Cartesian P2/s8 basis",
        "scope": "Local pairing only; global numerical rank and inf-sup estimates pending",
    }


def acquire(
    case: ThreeLayerCase, directory: Path, macros: tuple[int, ...], *, order: int = 5
) -> list[dict[str, Any]]:
    """Acquire original local M/K/B/f and actual nodal coordinates one macro at a time.

    No alternate local quadrature, Newmark formula, gauge or source vector is
    implemented here. ``_make_local`` is the shared spatial owner used by the
    public stepper. Archives are numeric; they are not serialized local objects
    or a substitute for independent native comparisons and global acceptance.
    """
    order = assembly_order(order)
    if (
        not macros
        or len(set(macros)) != len(macros)
        or any(type(i) is not int or not 0 <= i < len(case.mesh.cells) for i in macros)
    ):
        raise ValueError("select distinct macro indices in [0,341)")
    before = source_hashes()
    _bind(
        directory,
        {
            "schema": 1,
            "input_sha256": case.input_sha256,
            "source_sha256": before,
            "stage": "local_operators",
            "degree": 3,
            "refinement": 8,
            "trace_degree": 2,
            "segments": 8,
            "requested_assembly_order": order,
            "executed_assembly_order": order,
        },
    )
    density, stiffness = case.materials()
    skeleton = case.skeleton()
    records = []
    for cell in macros:
        path = directory / f"macro-{cell}-q{order}.json"
        if path.exists():
            record = json.loads(path.read_text())
            require_equal(record["macro"], cell, label="archived macro")
            verify_archive(directory / record["archive"], record["archive_sha256"])
            _verify_field_archives(directory, {str(cell): record["executed_field_basis"]})
            _unchanged(case, before)
            records.append(record)
            continue
        started = perf_counter()
        local = _make_local(cell, case.mesh, skeleton, 3, 8, order, density, stiffness, None, None)
        mass_error = float(
            sparse.linalg.norm(local.mass - local.mass.T) / sparse.linalg.norm(local.mass)
        )
        stiffness_error = float(
            sparse.linalg.norm(local.stiffness - local.stiffness.T)
            / sparse.linalg.norm(local.stiffness)
        )
        rigid = _rigid(local.nodes, local.nodes.mean(axis=0)).reshape(len(local.nodes) * 2, 3)
        kernel_error = float(
            np.linalg.norm(local.stiffness @ rigid)
            / (sparse.linalg.norm(local.stiffness) * np.linalg.norm(rigid))
        )
        force = local.load_at_time(case.source(), case.source_center_time_s / case.scales[3])
        with factorize(local.mass) as inverse:
            gram = local.coupling.T @ inverse.solve(local.coupling)
            eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2)
            mass_dual_force = float(np.sqrt(force @ inverse.solve(force)))
        pairing = trace_pairing_diagnostic(local, case.mesh, skeleton, cell)
        if max(mass_error, stiffness_error, kernel_error) > 1e-10:
            raise RuntimeError("original local symmetry/kernel check failed")
        archive = path.with_suffix(".npz")
        _save(
            archive,
            source_load=force,
            **local_arrays(local),
        )
        field_basis = _save_field_basis(directory / f"field-basis-{cell}-q{order}.npz", local)
        _unchanged(case, before)
        record = {
            "schema": 1,
            "macro": cell,
            "requested_assembly_order": order,
            "executed_assembly_order": order,
            "local_vector_P3_dofs": local.mass.shape[0],
            "trace_columns": local.coupling.shape[1],
            "elapsed_seconds": perf_counter() - started,
            "mass_symmetry_relative_error": mass_error,
            "stiffness_symmetry_relative_error": stiffness_error,
            "rigid_kernel_relative_error": kernel_error,
            "trace_pairing": pairing,
            "mass_dual_trace_gram_eigenvalues_min_max": [
                float(eigenvalues[0]),
                float(eigenvalues[-1]),
            ],
            "source_mass_dual_norm": mass_dual_force,
            "source_resultant_nondimensional": force.reshape(-1, 2).sum(axis=0).tolist(),
            "source_first_moment_nondimensional": (
                (local.nodes - case.source().center).T @ force.reshape(-1, 2)
            ).tolist(),
            "archive": archive.name,
            "archive_sha256": file_digest(archive),
            "source_sha256": before,
            "input_sha256": case.input_sha256,
            "basis_contract": (
                "Executed analytic Polynomial factors/derivatives, actual held local mass "
                "tables, original mesh/DOFs and literal one-sided field q5/q7 tables; "
                "interleaved Cartesian components"
            ),
            "executed_field_basis": field_basis,
            "scope": (
                "Original local operator acquisition; no global trajectory or "
                "independent reference acceptance"
            ),
            "scientific_acceptance": False,
        }
        _write(path, record)
        records.append(record)
        del local
    return records


def original_step_checks(
    stepper: ElastodynamicStepper,
    before: ElastodynamicSolution,
    after: ElastodynamicSolution,
    source: Any,
) -> dict[str, Any]:
    """Verify original momentum and work/energy for one zero-traction Newmark slab.

    Operators and forces come from the shared local owners. This diagnostic
    evaluates their physical equations; it does not implement a time update.
    The interval multiplier is negative outward traction, giving force -B*lambda.
    Linear/angular momentum uses the shared rigid test functions. Exterior
    traction must be zero, and both endpoint fields satisfy displacement moments.
    Native assembly/trajectory and classical refinement remain separate controls.
    """
    if stepper.dimension != 2:
        raise ValueError("selected-case physical checks require two dimensions")
    boundary_dofs = np.concatenate(
        [stepper.skeleton.dofs(int(face)) for face in stepper.mesh.boundary_faces]
    )
    if (
        any(count != 1 for count in stepper.substeps)
        or np.any(stepper.fixed)
        or np.intersect1d(stepper.free, boundary_dofs).size
    ):
        raise ValueError("original selected-case checks require one substep and zero traction")
    dt = stepper.time_step
    if not np.isclose(after.time - before.time, dt, rtol=0, atol=8 * np.finfo(float).eps):
        raise ValueError("physical checks require consecutive endpoint states")
    momentum_error, work = 0.0, np.longdouble(0)
    old_momentum, new_momentum, source_impulse = (
        np.zeros(3, dtype=np.longdouble) for _ in range(3)
    )
    rigid_scale = 0.0
    center = (
        stepper.mesh.areas
        @ stepper.mesh.points[stepper.mesh.cells].mean(axis=1)
        / stepper.mesh.areas.sum()
    )
    for local, old_u, old_v, new_u, new_v in zip(
        stepper.locals,
        before.displacement,
        before.velocity,
        after.displacement,
        after.velocity,
        strict=True,
    ):
        load = local.load_at_time(source, before.time) + local.load_at_time(source, after.time)
        traction = 2 * local.coupling @ after.trace[local.trace_dofs]
        internal = local.stiffness @ (old_u + new_u)
        inertia = local.mass @ (new_v - old_v)
        defect = inertia - dt / 2 * (load - traction - internal)
        scale = np.linalg.norm(inertia) + dt / 2 * (
            np.linalg.norm(load) + np.linalg.norm(traction) + np.linalg.norm(internal)
        )
        momentum_error = max(
            momentum_error, float(np.linalg.norm(defect) / max(scale, np.finfo(float).tiny))
        )
        work += np.sum((new_u - old_u).astype(np.longdouble) * load.astype(np.longdouble)) / 2
        rigid = _rigid(local.nodes, center).reshape(2 * len(local.nodes), 3)
        old_momentum += rigid.astype(np.longdouble).T @ (local.mass @ old_v).astype(np.longdouble)
        new_momentum += rigid.astype(np.longdouble).T @ (local.mass @ new_v).astype(np.longdouble)
        source_impulse += dt / 2 * (rigid.astype(np.longdouble).T @ load.astype(np.longdouble))
        rigid_scale += np.linalg.norm(rigid) * (
            np.linalg.norm(inertia) + dt / 2 * np.linalg.norm(load)
        )
    energy_scale = abs(after.energy) + abs(before.energy) + abs(work)
    energy_defect = float(
        abs(after.energy - before.energy - work)
        / cast(np.floating[Any], max(energy_scale, np.finfo(float).tiny))
    )
    rigid_defect = float(
        np.linalg.norm(new_momentum - old_momentum - source_impulse)
        / max(rigid_scale, np.finfo(float).tiny)
    )
    if (
        not np.isfinite(
            [momentum_error, energy_defect, rigid_defect, before.energy, after.energy]
        ).all()
        or min(before.energy, after.energy) < 0
        or max(momentum_error, energy_defect, rigid_defect) > 1e-10
    ):
        raise RuntimeError("original physical momentum, energy/work or rigid-momentum gate failed")
    return {
        "original_local_momentum_relative_residual": momentum_error,
        "physical_work_energy_relative_residual": energy_defect,
        "linear_angular_momentum_relative_residual": rigid_defect,
        "source_work_increment_nondimensional": float(work),
        "linear_angular_momentum_nondimensional": [float(value) for value in new_momentum],
        "original_physical_threshold": 1e-10,
    }


def trajectory(
    case: ThreeLayerCase, directory: Path, *, steps: int = 300, order: int = 5
) -> dict[str, Any]:
    """Run the public fresh stepper, retaining every physical state and signed slab multiplier.

    Setup assembles all 341 macros and must first receive a measured campaign
    budget. Restarting serialized private local objects is unsupported; a fresh
    directory preserves a distinct source generation. No numerical operator or
    Newmark update is duplicated here.
    """
    order = assembly_order(order)
    if type(steps) is not int or not 0 <= steps <= 300:
        raise ValueError("select a pilot or complete trajectory with steps in [0,300]")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("a fresh trajectory directory is required")
    before = source_hashes()
    directory.mkdir(parents=True, exist_ok=True)
    density, stiffness = case.materials()
    _write(
        directory / "identity.json",
        {
            "schema": 1,
            "stage": "fresh_trajectory",
            "source_sha256": before,
            "input_sha256": case.input_sha256,
            "degree": 3,
            "refinement": 8,
            "trace_degree": 2,
            "segments": 8,
            "physical_time_step_s": 0.001,
            "physical_final_time_s": 0.3,
            "requested_assembly_order": order,
            "executed_assembly_order": order,
            "scientific_acceptance": False,
        },
    )
    started = perf_counter()
    with ElastodynamicStepper(
        case.mesh,
        time_step=0.001 / case.scales[3],
        degree=3,
        local_refinement=8,
        skeleton=case.skeleton(),
        density=density,
        constitutive=stiffness,
        traction={int(face): np.zeros(2) for face in case.mesh.boundary_faces},
        quadrature_order=order,
    ) as stepper:
        setup_seconds = perf_counter() - started
        source_started = perf_counter()
        source = stepper.prepare_source(case.source())
        source_preparation_seconds = perf_counter() - source_started
        archive_started = perf_counter()
        operators, contracts, fields = {}, {}, {}
        for cell, local in enumerate(stepper.locals):
            path = directory / f"operator-{cell}.npz"
            arrays = {**local_arrays(local), "prepared_spatial_source_load": source.loads[cell]}
            contracts[path.name] = operator_contract_digest(arrays)
            _save(path, **arrays)
            operators[path.name] = file_digest(path)
            fields[str(cell)] = _save_field_basis(directory / f"field-basis-{cell}.npz", local)
        macro_archive = directory / "macro-geometry.npz"
        _save(
            macro_archive,
            points=case.mesh.points,
            cells=case.mesh.cells,
            faces=case.mesh.faces,
            signs=case.mesh.signs,
            trace_offsets=stepper.skeleton.offsets,
            trace_breaks=np.linspace(0, 1, 9),
            trace_degree=np.asarray(2),
        )
        operator_archival_seconds = perf_counter() - archive_started
        _unchanged(case, before)
        _write(
            directory / "setup.json",
            {
                "setup_seconds": setup_seconds,
                "source_preparation_seconds": source_preparation_seconds,
                "operator_archival_seconds": operator_archival_seconds,
                "executed_operator_contract_sha256": contracts,
                "source_contract": (
                    "Explicit shared separable source snapshot; original unweighted nodal "
                    "force vectors and original endpoint/substep temporal factors"
                ),
                "macros": len(stepper.locals),
                "free_trace_dofs": len(stepper.free),
                "global_trace_dofs": stepper.size,
                "global_matrix_nnz": stepper.matrix.nnz,
                "executed_operator_archives_sha256": operators,
                "executed_field_basis": fields,
                "executed_macro_geometry_sha256": file_digest(macro_archive),
                "scientific_acceptance": False,
            },
        )
        previous = None
        step_measurements = []
        for index in range(steps + 1):
            step_started = perf_counter()
            solution = stepper.initialize() if index == 0 else stepper.advance(source)
            numerical_seconds = perf_counter() - step_started
            gate_started = perf_counter()
            checks: dict[str, Any] = (
                {}
                if previous is None
                else original_step_checks(stepper, previous, solution, source)
            )
            physical_gate_seconds = perf_counter() - gate_started
            basis_started = perf_counter()
            for cell, local in enumerate(stepper.locals):
                arrays = {**local_arrays(local), "prepared_spatial_source_load": source.loads[cell]}
                require_equal(
                    operator_contract_digest(arrays),
                    contracts[f"operator-{cell}.npz"],
                    label="executed local basis/operator/source contract",
                )
            basis_gate_seconds = perf_counter() - basis_started
            archival_started = perf_counter()
            archive = directory / f"state-{index:03}.npz"
            _save(
                archive,
                displacement=np.asarray(solution.displacement),
                velocity=np.asarray(solution.velocity),
                trace=solution.trace,
            )
            _unchanged(case, before)
            measurement = {
                "step": index,
                "numerical_seconds": numerical_seconds,
                "original_physical_gate_seconds": physical_gate_seconds,
                "executed_basis_gate_seconds": basis_gate_seconds,
                "state_archive_seconds": perf_counter() - archival_started,
            }
            step_measurements.append(measurement)
            _write(
                archive.with_suffix(".json"),
                {
                    **checks,
                    "measurement": measurement,
                    "executed_operator_contract_sha256": contracts,
                    "executed_field_basis": fields,
                    "step": index,
                    "physical_time_s": solution.time * case.scales[3],
                    "energy_nondimensional": solution.energy,
                    "displacement_constraint_residual": solution.constraint_residual,
                    "trace_convention": case.metadata()["trace_convention"],
                    "archive": archive.name,
                    "archive_sha256": file_digest(archive),
                    "scientific_acceptance": False,
                },
            )
            previous = solution
        _verify_field_archives(directory, fields)
        _unchanged(case, before)
    record = {
        "status": "trajectory_acquired" if steps == 300 else "partial_trajectory_acquired",
        "persisted_time_states": steps + 1,
        "setup_seconds": setup_seconds,
        "source_preparation_seconds": source_preparation_seconds,
        "operator_archival_seconds": operator_archival_seconds,
        "step_measurements": step_measurements,
        "elapsed_seconds": perf_counter() - started,
        "scientific_acceptance": False,
        "executed_field_basis": fields,
        "independent_native_trajectory_and_classical_refinement_pending": True,
    }
    _write(directory / "completion.json", record)
    return record


def main() -> None:
    """Prepare inputs by default; acquire operators or a fresh trajectory only explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", "inputs", "local", "trajectory"),
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--inputs", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--macros", type=int, nargs="+", default=[7, 10, 151, 153])
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    case = load_case(args.inputs)
    if args.stage == "preflight":
        record = preflight(case)
    else:
        if args.output is None:
            parser.error("acquisition requires an explicit output directory")
        with threadpool_limits(1):
            if args.stage == "inputs":
                record = prepare(case, args.output)
            elif args.stage == "local":
                record = {"rows": acquire(case, args.output, tuple(args.macros), order=args.order)}
            else:
                record = trajectory(case, args.output, steps=args.steps, order=args.order)
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
