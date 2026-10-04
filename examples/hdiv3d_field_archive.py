"""Persist executed affine mixed Darcy fields and their complete basis contract.

Flux is a contravariantly mapped H(div) field; pressure uses scalar composition.
The local original matrices are captured from production, with their declared
diagonal coordinate scaling, rather than independently reassembled for replay.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy import sparse

from examples.archive_precision import precision_fields, restore_precision
from examples.campaign_provenance import file_digest
from examples.local_response_cache import array_identity
from examples.transport_checkpoints import checkpoint_field, write_progress
from pymhm.darcy_hdiv3d import Mixed3DDarcySolution
from pymhm.hdiv3d_family import cell_quadrature
from pymhm.hdiv3d_mesh import hdiv3d_dofs, hdiv3d_transform
from pymhm.hybrid import HybridSystem
from pymhm.solvers import _accurate_residual

SCHEMA = "pymhm-affine-hdiv3d-executed-field-v1"


@dataclass
class ProductionObservation:
    """Actual unchanged hybrid solve and applied physical boundary moments."""

    system: HybridSystem | None = None
    boundary_load: np.ndarray | None = None
    fixed: dict[int, float] = dataclass_field(default_factory=dict)


@contextmanager
def observe_system() -> Iterator[ProductionObservation]:
    """Observe one serial production solve in an isolated process, without reassembly.

    The original method and its arguments are delegated unchanged. This observer
    is process scoped and must not overlap another solve in the same process.
    """
    import pymhm.darcy_hdiv3d as owner

    observed = ProductionObservation()
    original = HybridSystem.solve
    boundary_owner = owner._boundary

    def boundary(*args: Any, **kwargs: Any) -> Any:
        """Capture the actual applied boundary vector without evaluating it again."""
        result = boundary_owner(*args, **kwargs)
        if observed.boundary_load is not None:
            raise ValueError("One production boundary call is required")
        observed.boundary_load, observed.fixed = result[0].copy(), dict(result[1])
        return result

    def solve(system: HybridSystem, *args: Any, **kwargs: Any) -> Any:
        """Retain the actual production system only after its solve returns."""
        result = original(system, *args, **kwargs)
        if observed.system is not None:
            raise ValueError("One production system is required")
        observed.system = system
        return result

    with patch.object(HybridSystem, "solve", solve), patch.object(owner, "_boundary", boundary):
        yield observed


def restore(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Restore archived high, correction and tail components without narrowing."""
    return restore_precision(arrays[name], arrays[name + "_correction"], arrays[name + "_tail"])


def _csr(arrays: Mapping[str, np.ndarray], name: str) -> sparse.csr_matrix:
    """Restore literal sparse entries and row orientation from the archive."""
    return sparse.csr_matrix(
        (arrays[name + "_data"], arrays[name + "_indices"], arrays[name + "_indptr"]),
        shape=tuple(arrays[name + "_shape"]),
    )


def _put_csr(arrays: dict[str, np.ndarray], name: str, matrix: Any) -> None:
    """Persist CSR without symmetrizing, scaling or modifying represented entries."""
    matrix = sparse.csr_matrix(matrix)
    arrays.update(
        {
            name + "_data": matrix.data,
            name + "_indices": matrix.indices,
            name + "_indptr": matrix.indptr,
            name + "_shape": np.asarray(matrix.shape, dtype=np.int64),
        }
    )


def field_arrays(
    solution: Mixed3DDarcySolution,
    observed: ProductionObservation,
    *,
    assembly_order: int,
    norm_orders: tuple[int, ...] = (12, 13),
) -> dict[str, np.ndarray]:
    """Capture actual bases, maps, scaled original rows and reconstructed fields.

    Supported publication data use identity permeability, homogeneous weak
    Dirichlet pressure on the complete exterior, and one affine fine cell per
    macro. The retained constant-pressure amplitude is related to the physical
    integral through the actual moment pairing with the executed retained basis.
    """
    system = observed.system
    if system is None or observed.boundary_load is None:
        raise ValueError("The actual production solve and boundary call are required")
    if (
        len(system.responses) != len(solution.local_meshes)
        or not norm_orders
        or len(set(norm_orders)) != len(norm_orders)
        or any(type(order) is not int or order < 4 for order in norm_orders)
        or type(assembly_order) is not int
        or assembly_order < solution.family.pressure_degree + 2
        or any(len(mesh.cells) != 1 for mesh in solution.local_meshes)
    ):
        raise ValueError("Actual single-cell affine systems and sufficient quadratures required")
    family, macro = solution.family, solution.skeleton.mesh
    arrays = {
        "pressure_degree": np.asarray(family.pressure_degree),
        "normal_degree": np.asarray(family.normal_degree),
        "trace_degree": np.asarray(solution.skeleton.degree),
        "cell_kind_code": np.asarray(0 if family.kind == "tetrahedron" else 1),
        "local_count": np.asarray(len(solution.local_meshes)),
        "assembly_order": np.asarray(assembly_order),
        "norm_orders": np.asarray(norm_orders),
        "basis_coefficients": family.coefficients.copy(),
        "macro_points": macro.points,
        "macro_cells": macro.cells,
        "macro_cell_faces": macro.cell_faces,
        "macro_signs": macro.signs,
        "macro_normals": macro.normals,
        "macro_boundary_faces": macro.boundary_faces,
        "macro_face_offsets": np.r_[0, np.cumsum([len(face) for face in macro.faces])],
        "macro_face_vertices": np.concatenate(macro.faces),
        "kernel_offsets": system.kernel_offsets,
        "trace_size": np.asarray(system.trace_size),
        "physical_block_residuals": solution.physical_residuals,
    }
    arrays.update(precision_fields("global_trace", solution.hybrid.trace))
    arrays.update(precision_fields("global_rhs", system.rhs))
    arrays.update(precision_fields("boundary_load", observed.boundary_load))
    arrays["fixed_trace_indices"] = np.asarray(sorted(observed.fixed), dtype=np.int64)
    arrays["fixed_trace_values"] = np.asarray(
        [observed.fixed[key] for key in sorted(observed.fixed)]
    )
    _put_csr(arrays, "global_matrix", system.matrix)
    for order in norm_orders:
        points, weights = cell_quadrature(family.kind, order)
        values, divergence, pressure = family.tabulate(points)
        arrays.update(
            {
                f"q{order}_points": points,
                f"q{order}_weights": weights,
                f"q{order}_reference_flux": values,
                f"q{order}_reference_divergence": divergence,
                f"q{order}_pressure": pressure,
            }
        )
    for cell, (fine, response, metadata, field) in enumerate(
        zip(
            solution.local_meshes,
            system.responses,
            system.local_metadata,
            solution.hybrid.fields,
            strict=True,
        )
    ):
        actual_fine, nq, npres, scaling, moment = metadata
        if actual_fine is not fine:
            raise ValueError("Observed local mesh is not the production mesh")
        problem = response.problem
        owned = {
            "points": fine.points,
            "cells": fine.cells,
            "jacobian": fine.jacobian,
            "inverse": fine.inverse,
            "determinants": fine.determinants,
            "cell_faces": fine.cell_faces,
            "signs": fine.signs,
            "flux_dofs": hdiv3d_dofs(fine, family),
            "moment_transform": hdiv3d_transform(fine, family),
            "scaling": scaling,
            "physical_pressure_moment_scaled": moment,
            "flux_size": np.asarray(nq),
            "pressure_size": np.asarray(npres),
            "trace_dofs": problem.trace_dofs,
            "internal_field_nmant": np.asarray(np.finfo(field.dtype).nmant),
        }
        arrays.update({f"{name}_{cell}": np.asarray(value) for name, value in owned.items()})
        _put_csr(arrays, f"original_a_{cell}", problem.matrix)
        for name, value in (
            ("original_b", problem.coupling),
            ("original_f", problem.load),
            ("kernel", problem.kernel),
            ("constraints", problem.constraints),
            ("retained_basis", response.retained_basis),
            ("source_lift", response.source),
            ("trace_lifts", response.lifts),
            ("internal_field", field),
            ("flux", solution.flux[cell]),
            ("pressure", solution.pressure[cell]),
            ("coarse", solution.hybrid.coarse[cell]),
            ("retained_pressure_moment", moment @ response.retained_basis),
        ):
            arrays.update(precision_fields(f"{name}_{cell}", value))
    return arrays


def reference_tables(
    arrays: Mapping[str, np.ndarray], points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate declared raw polynomial candidates with the executed coefficient matrix.

    No numerical nullspace or new flux basis is constructed. The monomial or
    Bernstein candidate convention follows the recorded family orders.
    """
    from pymhm.hdiv3d_family import _powers, _scalar, _vectors

    kind = ("tetrahedron", "prism")[int(arrays["cell_kind_code"])]
    degree = int(arrays["pressure_degree"])
    normal = int(arrays["normal_degree"])
    if normal == 1 and degree in ((1, 2) if kind == "tetrahedron" else (1,)):
        raw, divergence = _vectors(points, _powers(kind, degree + 1))
    else:
        from pymhm.hdiv3d_general import candidates

        raw, divergence = candidates(kind, degree, points)
    coefficients = arrays["basis_coefficients"]
    return (
        np.einsum("qia,ij->qja", raw, coefficients),
        divergence @ coefficients,
        _scalar(points, _powers(kind, degree)),
    )


def replay(
    arrays: Mapping[str, np.ndarray], cell: int, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pressure, physical H(div) flux and physical divergence from literal tables."""
    if cell not in range(int(arrays["local_count"])) or order not in arrays["norm_orders"]:
        raise ValueError("An executed macro and quadrature are required")
    coordinates = restore(arrays, f"flux_{cell}")[arrays[f"flux_dofs_{cell}"]]
    transform = arrays[f"moment_transform_{cell}"]
    jacobian = arrays[f"jacobian_{cell}"]
    determinant = arrays[f"determinants_{cell}"]
    reference = np.einsum("tij,tj->ti", transform, coordinates)
    vector = np.einsum("ti,qib->tqb", reference, arrays[f"q{order}_reference_flux"])
    flux = np.einsum("tab,tqb->tqa", jacobian, vector) / determinant[:, None, None]
    divergence = (reference @ arrays[f"q{order}_reference_divergence"].T) / determinant[:, None]
    pressure = restore(arrays, f"pressure_{cell}") @ arrays[f"q{order}_pressure"].T
    return pressure, flux, divergence


def original_checks(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Verify captured original physical rows with their unchanged 1e-10 criterion.

    Dividing by the archived scaling restores physical equation units. Block
    backward errors use absolute operator actions; the full local-volume norm
    uses the original physical load. The latter excludes global skeletal rows
    and is reported separately, rather than silently substituting a Schur norm.
    """
    maximum = np.longdouble(0)
    defect_squared = load_squared = np.longdouble(0)
    rows = []
    trace = restore(arrays, "global_trace")
    global_values = np.r_[
        trace,
        np.concatenate(
            [restore(arrays, f"coarse_{cell}") for cell in range(int(arrays["local_count"]))]
        ),
    ]
    global_matrix = _csr(arrays, "global_matrix")
    global_rhs = restore(arrays, "global_rhs")
    compact = _accurate_residual(global_matrix, global_rhs, global_values)
    compact_free = np.ones(len(global_rhs), dtype=bool)
    compact_free[arrays["fixed_trace_indices"]] = False
    compact_relative = float(
        np.linalg.norm(compact[compact_free])
        / max(np.linalg.norm(global_rhs[compact_free]), np.finfo(float).tiny)
    )
    boundary = restore(arrays, "boundary_load")
    weak = -boundary.copy()
    for cell in range(int(arrays["local_count"])):
        matrix = _csr(arrays, f"original_a_{cell}")
        coupling = restore(arrays, f"original_b_{cell}")
        load = restore(arrays, f"original_f_{cell}")
        field = restore(arrays, f"internal_field_{cell}")
        local_trace = trace[arrays[f"trace_dofs_{cell}"]]
        scaling = arrays[f"scaling_{cell}"]
        combined = sparse.hstack((matrix, sparse.csr_matrix(coupling)), format="csr")
        values = np.r_[field, local_trace]
        defect = _accurate_residual(combined, load, values) / scaling
        action = (abs(combined) @ abs(values) + abs(load)) / scaling
        nq, npres = int(arrays[f"flux_size_{cell}"]), int(arrays[f"pressure_size_{cell}"])
        blocks = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        maximum = max(maximum, *blocks)
        defect_squared += np.sum(defect**2, dtype=np.longdouble)
        load_squared += np.sum((load / scaling) ** 2, dtype=np.longdouble)
        np.add.at(weak, arrays[f"trace_dofs_{cell}"], coupling.T @ field)
        declared_bits = int(arrays[f"internal_field_nmant_{cell}"])
        arithmetic = np.float64 if declared_bits == np.finfo(np.float64).nmant else np.longdouble
        if np.finfo(arithmetic).nmant < declared_bits:
            raise ValueError("Original producer arithmetic exceeds this replay platform")
        physical = scaling * field.astype(arithmetic)
        if not np.array_equal(physical[:nq], restore(arrays, f"flux_{cell}")) or not np.array_equal(
            physical[nq : nq + npres], restore(arrays, f"pressure_{cell}").ravel()
        ):
            raise ValueError("Physical fields differ from actual scaled production coordinates")
        moment = arrays[f"physical_pressure_moment_scaled_{cell}"] @ field
        coarse = restore(arrays, f"coarse_{cell}")
        target = restore(arrays, f"retained_pressure_moment_{cell}") @ coarse
        moment_defect = float(moment - target)
        if abs(moment_defect) > 1e-10 * max(1.0, abs(float(moment)), abs(float(target))):
            raise ValueError("Executed retained pressure moment pairing is inconsistent")
        rows.append(
            {
                "macro": cell,
                "physical_block_backward_errors": blocks,
                "physical_pressure_integral": float(moment),
                "coarse_constant_pressure_amplitude": float(coarse[0]),
                "physical_pressure_integral_minus_retained_pairing": moment_defect,
            }
        )
    if maximum > 1e-10 or compact_relative > 1e-10:
        raise ValueError("Captured original equation checks failed the unchanged criteria")
    free_trace = np.ones(len(trace), dtype=bool)
    free_trace[arrays["fixed_trace_indices"]] = False
    rhs_squared = load_squared + np.sum(boundary[free_trace] ** 2, dtype=np.longdouble)
    full_relative = (
        float(np.sqrt((defect_squared + np.sum(weak[free_trace] ** 2)) / rhs_squared))
        if rhs_squared > 0
        else None
    )
    if full_relative is not None and full_relative > 1e-10:
        raise ValueError("Original full uncondensed saddle exceeds 1e-10")
    return {
        "accepted": True,
        "criterion": 1e-10,
        "maximum_physical_block_backward_error": float(maximum),
        "all_local_physical_rows_relative_to_source": (
            float(np.sqrt(defect_squared / load_squared)) if load_squared > 0 else None
        ),
        "compact_original_relative_residual": compact_relative,
        "full_uncondensed_global_saddle_executed": True,
        "original_full_saddle_relative_to_physical_rhs": full_relative,
        "native_whole_field_agreement_verified": False,
        "rows": rows,
    }


def write_field(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    acquisition_uuid: str,
    source_sha256: Mapping[str, str],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Write actual field/basis bytes and all-array identities after original checks pass."""
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("A fresh executed field archive is required")
    checks = original_checks(arrays)
    orientation = verify_orientation(arrays)
    arrays = dict(arrays)
    arrays["schema"] = np.frombuffer(SCHEMA.encode(), dtype=np.uint8)
    arrays["acquisition_uuid"] = np.frombuffer(acquisition_uuid.encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_field(path, arrays, {"schema": SCHEMA, "acquisition_uuid": acquisition_uuid})
    metadata = {
        "schema": SCHEMA,
        "acquisition_uuid": acquisition_uuid,
        "archive_sha256": file_digest(path),
        "arrays_sha256": {key: array_identity(value) for key, value in arrays.items()},
        "source_sha256": dict(source_sha256),
        "configuration": dict(configuration),
        "original_checks": checks,
        "orientation_checks": orientation,
        "executed_basis": (
            "Actual reference matrix and cached moment transforms, "
            "with component monomial/Bernstein candidates"
        ),
        "operator_provenance": (
            "Literal executed scaled original A/B/f and physical diagonal scaling; "
            "no diagnostic reassembly"
        ),
    }
    write_progress(path.with_suffix(".json"), metadata)
    return metadata


def read_field(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify actual archive identities, basis tables and original rows without a solve."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("schema") != SCHEMA or metadata["archive_sha256"] != file_digest(path):
        raise ValueError("Executed field archive schema or digest differs")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if (
        arrays["schema"].tobytes().decode() != SCHEMA
        or arrays["acquisition_uuid"].tobytes().decode() != metadata["acquisition_uuid"]
        or {key: array_identity(value) for key, value in arrays.items()}
        != metadata["arrays_sha256"]
    ):
        raise ValueError("Executed field array identities differ")
    for order in arrays["norm_orders"]:
        order = int(order)
        tables = reference_tables(arrays, arrays[f"q{order}_points"])
        for name, value in zip(
            ("reference_flux", "reference_divergence", "pressure"), tables, strict=True
        ):
            if not np.allclose(value, arrays[f"q{order}_{name}"], rtol=1e-12, atol=1e-12):
                raise ValueError("Archived basis table is inconsistent with its executed matrix")
    verify_orientation(arrays)
    original_checks(arrays)
    return arrays, metadata


def verify_orientation(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Check affine maps and canonical normal-moment duality using the executed C/T.

    Geometry construction and raw polynomial evaluation require no new numerical
    basis. Every face moment tests all flux columns, including interior bubbles.
    This is a finite basis invariant, rather than a uniform inf-sup estimate.
    """
    from pymhm.hdiv3d_family import (
        face_polynomials,
        face_quadrature,
        face_shape,
        face_size,
        reference_faces,
        reference_vertices,
    )
    from pymhm.hdiv3d_mesh import AffineMixedMesh

    kind = ("tetrahedron", "prism")[int(arrays["cell_kind_code"])]
    normal_degree = int(arrays["normal_degree"])
    vertices = reference_vertices(kind)
    local_size = arrays["basis_coefficients"].shape[1]
    maximum = 0.0
    for macro in range(int(arrays["local_count"])):
        mesh = AffineMixedMesh(arrays[f"points_{macro}"], arrays[f"cells_{macro}"], kind)
        for name in ("cells", "jacobian", "inverse", "determinants", "cell_faces", "signs"):
            if not np.allclose(
                getattr(mesh, name), arrays[f"{name}_{macro}"], rtol=1e-14, atol=1e-14
            ):
                raise ValueError("Archived affine geometry is inconsistent with its vertices")
        if not np.array_equal(arrays[f"flux_dofs_{macro}"], np.arange(local_size)[None]):
            raise ValueError("The single-cell actual flux numbering is inconsistent")
        offset = 0
        transform = arrays[f"moment_transform_{macro}"][0]
        for side, face in enumerate(reference_faces(kind)):
            count = face_size(len(face), normal_degree)
            uv, weights = face_quadrature(len(face), max(4, normal_degree + 2))
            nodes = vertices[list(face)]
            reference = face_shape(uv, len(face)) @ nodes
            values = reference_tables(arrays, reference)[0]
            normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
            if normal @ (nodes.mean(axis=0) - vertices.mean(axis=0)) < 0:
                normal *= -1
            physical = mesh.points[mesh.cells[0, 0]] + reference @ mesh.jacobian[0].T
            canonical = mesh.face_coordinates(int(mesh.cell_faces[0, side]), physical)
            tests = face_polynomials(canonical, len(face), normal_degree)
            moments = (
                mesh.signs[0, side] * tests.T @ (weights[:, None] * (values @ normal)) @ transform
            )
            expected = np.zeros((count, local_size))
            expected[:, offset : offset + count] = np.eye(count)
            difference = float(np.max(abs(moments - expected)))
            maximum = max(maximum, difference)
            if difference > 1e-10:
                raise ValueError("Archived moment orientation violates canonical normal duality")
            offset += count
    return {"maximum_canonical_normal_moment_defect": maximum}
