"""Archive finite P2/P0 MsHHO3D fields, physical maps and executed moment lifts.

Original stiffness matrices are diagnostic reassemblies by the shared numerical
owner, explicitly distinguished from the returned executed R/C/E matrices.
Replay uses only archived cardinal tables and physical maps. The supported
archive is the projected constant-source-moment method with physical Dirichlet
pressure on every exterior macroface, on tetrahedral or convex polyhedral macros.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from examples.archive_precision import precision_fields, restore_precision
from examples.local_response_cache import array_identity
from examples.transport_checkpoints import checkpoint_field, write_progress
from pymhm.fem.scalar.tetrahedron import (
    tetra_basis,
    tetra_nodal_space,
    tetra_operators,
    tetrahedron_quadrature,
)
from pymhm.fem.traces.moments_3d import face_moment_rule_3d as _face_rule
from pymhm.io.provenance import file_digest
from pymhm.io.workspace import case_workspace, local_resource, read_resource_text, source_file
from pymhm.linalg.linear import LinearSolveError, accurate_residual
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.postprocessing.solutions import MsHHO3DSolution

SCHEMA = "pymhm-mshho3d-p2-p0-field-v1"
ROOT = case_workspace()
_EDGES = tuple(combinations(range(4), 2))


def restore(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Restore portable high/correction/tail coordinates without narrowing digits."""
    return restore_precision(arrays[name], arrays[name + "_correction"], arrays[name + "_tail"])


def field_arrays(
    solution: MsHHO3DSolution,
    *,
    assembly_order: int,
    source: Any,
    dirichlet: Any = 0.0,
    norm_orders: tuple[int, ...] = (10, 12),
    display_refinement: int = 0,
) -> dict[str, np.ndarray]:
    """Capture actual P2 coefficients/R/C/E and independent physical norm tables.

    The physical source is the executed P0 projection, not the unprojected
    sampled source. The independently reassembled A and source integral audit
    the returned local moment lift; no additional field solve is performed.
    Each original polygonal macroface retains its original single P0 moment.
    """
    if (
        solution.degree != 2
        or solution.source_variant != "projected"
        or any(local.cell_count != 1 for local in solution.local)
        or type(assembly_order) is not int
        or assembly_order < 3
        or not norm_orders
        or len(set(norm_orders)) != len(norm_orders)
        or any(type(q) is not int or q < 3 for q in norm_orders)
        or type(display_refinement) is not int
        or display_refinement < 0
    ):
        raise ValueError(
            "P2/P0 projected fields and positive sufficient quadrature orders required"
        )
    macro = solution.skeleton.mesh
    arrays = {
        "degree": np.asarray(2, dtype=np.int64),
        "cell_degree": np.asarray(0, dtype=np.int64),
        "local_count": np.asarray(len(solution.local), dtype=np.int64),
        "assembly_order": np.asarray(assembly_order, dtype=np.int64),
        "norm_orders": np.asarray(norm_orders, dtype=np.int64),
        "coefficient_nmant": np.asarray(np.finfo(solution.pressure[0].dtype).nmant),
        "macro_points": macro.points,
        "macro_face_offsets": np.r_[0, np.cumsum([len(ids) for ids in macro.faces])],
        "macro_face_vertices": np.concatenate(macro.faces),
        "macro_normals": macro.normals,
        "macro_boundary_faces": macro.boundary_faces,
        "macro_cell_face_offsets": np.r_[0, np.cumsum([len(ids) for ids in macro.cell_faces])],
        "macro_cell_faces": np.concatenate(macro.cell_faces),
        "macro_cell_signs": np.concatenate(macro.signs),
        "reference_nodal_barycentric": np.vstack(
            (np.eye(4), [(np.eye(4)[i] + np.eye(4)[j]) / 2 for i, j in _EDGES])
        ),
    }
    arrays.update(precision_fields("face_moments", solution.face_moments))
    for order in norm_orders:
        bary, weights = tetrahedron_quadrature(order)
        basis, derivative = tetra_basis(2, bary)
        arrays[f"q{order}_barycentric"] = bary
        arrays[f"q{order}_weights"] = weights
        arrays[f"q{order}_cardinal_values"] = basis
        arrays[f"q{order}_cardinal_derivatives"] = derivative
    boundary = np.zeros(solution.skeleton.size)
    for face in macro.boundary_faces:
        points, weights, basis = _face_rule(macro, solution.skeleton, int(face), assembly_order)
        ids = solution.skeleton.dofs(int(face))
        boundary[ids] = basis.T @ (weights * scalar_values_3d(dirichlet, points))
        arrays[f"boundary_points_{face}"] = points
        arrays[f"boundary_weights_{face}"] = weights
        arrays[f"boundary_basis_{face}"] = basis
    arrays["audit_boundary_load"] = boundary
    physical_trace = np.zeros(solution.skeleton.size, dtype=np.longdouble)
    seen = np.zeros(solution.skeleton.size, dtype=bool)
    duplicate = np.longdouble(0)
    for cell, (local, pressure, moments) in enumerate(
        zip(solution.local, solution.pressure, solution.cell_moments, strict=True)
    ):
        fine = local.mesh
        dofs, nodes = tetra_nodal_space(fine, 2)
        ids = solution.skeleton.cell_dofs(cell)
        if len(ids) != len(macro.cell_faces[cell]):
            raise ValueError("One unsplit P0 moment per original macroface is required")
        coordinates = np.r_[moments, solution.face_moments[ids]]
        jacobian = (fine.points[fine.cells[:, 1:]] - fine.points[fine.cells[:, :1]]).transpose(
            0, 2, 1
        )
        inverse = np.linalg.inv(jacobian)
        gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
        original_a, mass, unprojected_f = tetra_operators(
            fine, 2, diffusion=solution.permeability, source=source, order=assembly_order
        )
        # The shared owner returns CSC. Persist declared CSR rows explicitly,
        # preserving the original represented matrix's small antisymmetry.
        original_a = original_a.tocsr()
        integral = np.asarray(mass.sum(axis=1)).ravel()
        source_mean = float(np.sum(unprojected_f) / np.sum(fine.volumes))
        if not np.allclose(local.load[:1], [source_mean], rtol=1e-12, atol=1e-12):
            raise ValueError("Declared source differs from the executed constant moment projection")
        signed_b = local.moments[:, 1:] * macro.signs[cell]
        projected_f = local.moments[:, 0] * local.load[0]
        local_trace = -macro.signs[cell] * (local.energy @ coordinates - local.load)[1:]
        first = ~seen[ids]
        if np.any(~first):
            duplicate = max(
                duplicate, np.max(np.abs(physical_trace[ids[~first]] - local_trace[~first]))
            )
        physical_trace[ids[first]] = local_trace[first]
        seen[ids] = True
        owned = {
            "points": fine.points,
            "cells": fine.cells,
            "nodal_dofs": dofs,
            "nodes": nodes,
            "barycentric_gradients": gradients,
            "volumes": fine.volumes,
            "face_dofs": ids,
            "face_signs": macro.signs[cell],
            "audit_a_data": original_a.data,
            "audit_a_indices": original_a.indices,
            "audit_a_indptr": original_a.indptr,
            "audit_b": signed_b,
            "audit_projected_f": projected_f,
            "audit_unprojected_f": unprojected_f,
            "audit_physical_integral": integral,
            "physical_constant": np.ones(len(nodes)),
        }
        arrays.update({f"{name}_{cell}": np.asarray(value) for name, value in owned.items()})
        for name, values in (
            ("pressure", pressure),
            ("cell_moments", moments),
            ("executed_reconstruction", local.reconstruction),
            ("executed_moments", local.moments),
            ("executed_energy", local.energy),
            ("executed_load", local.load),
            ("executed_integral", local.integral),
        ):
            arrays.update(precision_fields(f"{name}_{cell}", values))
        for order in norm_orders:
            physical = np.einsum(
                "qi,tia->tqa", arrays[f"q{order}_barycentric"], fine.points[fine.cells]
            )
            arrays[f"q{order}_physical_points_{cell}"] = physical
            arrays[f"q{order}_material_{cell}"] = tensor_values_3d(
                solution.permeability, physical.reshape(-1, 3)
            ).reshape(*physical.shape[:2], 3, 3)
    if not seen.all():
        raise ValueError("Physical trace injection is incomplete")
    arrays.update(precision_fields("physical_trace", physical_trace))
    arrays["duplicate_trace_absolute_difference"] = np.asarray(duplicate, dtype=float)
    if display_refinement:
        from examples.mshho3d_sections import capture_section

        arrays.update(capture_section(solution, display_refinement))
    validate_arrays(arrays)
    return arrays


def _same(actual: np.ndarray, expected: np.ndarray, message: str, tolerance: float = 1e-10) -> None:
    """Require equal array shapes and the stated absolute and relative value tolerance."""
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=tolerance, atol=tolerance
    ):
        raise ValueError(message)


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Check archived geometry, full P2 identities, actual moments and reconstructions.

    This operation constructs no numerical basis or local solution. Cardinal
    ordering is bound to declared nodal barycentric locations through all
    degree-two monomials, including their three physical reference derivatives.
    """
    if any(a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Finite portable real numerical arrays required")
    for name, expected in (("degree", 2), ("cell_degree", 0)):
        if (
            arrays[name].shape != ()
            or arrays[name].dtype.kind not in "iu"
            or int(arrays[name]) != expected
        ):
            raise ValueError("This archive supports actual P2/P0 scalar fields")
    count = arrays["local_count"]
    if count.shape != () or count.dtype.kind not in "iu" or int(count) < 1:
        raise ValueError("Positive integral macrocell count required")
    precision = arrays["coefficient_nmant"]
    if (
        precision.shape != ()
        or precision.dtype.kind not in "iu"
        or not 52 <= int(precision) <= np.finfo(np.longdouble).nmant
    ):
        raise ValueError("The replay host cannot preserve the declared coefficient precision")
    reference_nodes = arrays["reference_nodal_barycentric"]
    if (
        reference_nodes.shape != (10, 4)
        or np.any(reference_nodes < 0)
        or not np.allclose(reference_nodes.sum(axis=1), 1, rtol=0, atol=1e-14)
        or len(np.unique(reference_nodes, axis=0)) != 10
    ):
        raise ValueError("Ten physical P2 nodal barycentric locations required")
    orders = arrays["norm_orders"]
    if (
        orders.ndim != 1
        or orders.dtype.kind not in "iu"
        or not len(orders)
        or np.any(orders < 3)
        or len(np.unique(orders)) != len(orders)
    ):
        raise ValueError("Distinct sufficient integer norm orders required")
    face_moments = restore(arrays, "face_moments")
    for order in arrays["norm_orders"]:
        order = int(order)
        bary, weights = arrays[f"q{order}_barycentric"], arrays[f"q{order}_weights"]
        values = arrays[f"q{order}_cardinal_values"]
        derivatives = arrays[f"q{order}_cardinal_derivatives"]
        if (
            bary.shape != (order**3, 4)
            or weights.shape != (order**3,)
            or np.any(weights <= 0)
            or np.any(bary <= 0)
        ):
            raise ValueError("Positive three-dimensional Duffy quadrature required")
        _same(bary.sum(axis=1), np.ones(len(weights)), "Barycentric quadrature points differ")
        _same(np.asarray(weights.sum()), np.asarray(1.0), "Volume quadrature normalization differs")
        if values.shape != (len(weights), 10) or derivatives.shape != (len(weights), 10, 4):
            raise ValueError("Executed P2 cardinal dimensions differ")
        for a in range(3):
            for b in range(3 - a):
                for c in range(3 - a - b):
                    powers = np.asarray([a, b, c])
                    coefficients = np.prod(reference_nodes[:, 1:] ** powers, axis=1)
                    target = np.prod(bary[:, 1:] ** powers, axis=1)
                    _same(
                        values @ coefficients,
                        target,
                        "Archived cardinal values fail a P2 monomial identity",
                    )
                    for axis in range(3):
                        reduced = powers.copy()
                        reduced[axis] = max(0, reduced[axis] - 1)
                        gradient = powers[axis] * np.prod(bary[:, 1:] ** reduced, axis=1)
                        _same(
                            (derivatives[..., axis + 1] - derivatives[..., 0]) @ coefficients,
                            gradient,
                            "Archived cardinal derivatives fail a P2 monomial identity",
                        )
    for cell in range(int(count)):
        points, cells = arrays[f"points_{cell}"], arrays[f"cells_{cell}"]
        nodes, dofs = arrays[f"nodes_{cell}"], arrays[f"nodal_dofs_{cell}"]
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or cells.ndim != 2
            or cells.shape[1] != 4
            or cells.dtype.kind not in "iu"
            or np.any(cells < 0)
            or np.any(cells >= len(points))
        ):
            raise ValueError("Finite tetrahedral physical geometry required")
        if (
            dofs.shape != (len(cells), 10)
            or dofs.dtype.kind not in "iu"
            or np.any(dofs < 0)
            or np.any(dofs >= len(nodes))
        ):
            raise ValueError("Executed P2 nodal injection differs")
        vertices = points[cells]
        _same(
            nodes[dofs],
            np.einsum("qi,tia->tqa", reference_nodes, vertices),
            "Physical nodal map differs",
        )
        homogeneous = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
        inverse = np.linalg.inv(homogeneous)
        gradients = arrays[f"barycentric_gradients_{cell}"]
        _same(
            gradients, inverse[:, 1:, :].transpose(0, 2, 1), "Physical barycentric gradients differ"
        )
        determinants = np.linalg.det((vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1))
        if np.any(determinants <= 0):
            raise ValueError("Positive oriented fine tetrahedra required")
        _same(arrays[f"volumes_{cell}"], determinants / 6, "Physical tetrahedral volumes differ")
        pressure = restore(arrays, f"pressure_{cell}")
        coordinates = np.r_[
            restore(arrays, f"cell_moments_{cell}"), face_moments[arrays[f"face_dofs_{cell}"]]
        ]
        reconstruction = restore(arrays, f"executed_reconstruction_{cell}")
        moments = restore(arrays, f"executed_moments_{cell}")
        _same(
            reconstruction @ coordinates, pressure, "Pressure differs from the executed moment lift"
        )
        _same(
            moments.T @ pressure,
            coordinates,
            "Pressure differs from the physical moment coordinates",
        )
        _same(
            moments.T @ reconstruction,
            np.eye(len(coordinates)),
            "Executed reconstruction fails its moment identities",
        )
        matrix = sparse.csr_matrix(
            (
                arrays[f"audit_a_data_{cell}"],
                arrays[f"audit_a_indices_{cell}"],
                arrays[f"audit_a_indptr_{cell}"],
            ),
            shape=(len(pressure), len(pressure)),
        )
        _same(
            restore(arrays, f"executed_energy_{cell}"),
            reconstruction.T @ (matrix @ reconstruction),
            "Executed Galerkin action differs from the reassembled original operator",
        )
        _same(
            restore(arrays, f"executed_integral_{cell}"),
            arrays[f"audit_physical_integral_{cell}"] @ reconstruction,
            "Declared physical pressure integral differs",
        )
        _same(
            moments[:, 0],
            arrays[f"audit_physical_integral_{cell}"],
            "Constant volume moment differs from the physical pressure integral",
        )
        _same(
            arrays[f"physical_constant_{cell}"],
            np.ones(len(nodes)),
            "Declared physical constant differs",
        )
        for order in arrays["norm_orders"]:
            order = int(order)
            _same(
                arrays[f"q{order}_physical_points_{cell}"],
                np.einsum("qi,tia->tqa", arrays[f"q{order}_barycentric"], vertices),
                "Physical quadrature map differs",
            )
    from examples.mshho3d_sections import validate_section

    validate_section(arrays)


def original_checks(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Measure reassembled original projected-source equations in physical RHS norm.

    Boundary moments prescribe pressure and fix its physical level. The recovered
    multiplier is physical normal Darcy flux, separate from the raw gradient.
    No energy-action scaling or artificial pressure gauge is applied.
    """
    trace = restore(arrays, "physical_trace")
    boundary = arrays["audit_boundary_load"].astype(np.longdouble)
    weak = -boundary.copy()
    defects = rhs = np.longdouble(0)
    for cell in range(int(arrays["local_count"])):
        pressure = restore(arrays, f"pressure_{cell}")
        matrix = sparse.csr_matrix(
            (
                arrays[f"audit_a_data_{cell}"],
                arrays[f"audit_a_indices_{cell}"],
                arrays[f"audit_a_indptr_{cell}"],
            ),
            shape=(len(pressure), len(pressure)),
        )
        coupling, load = arrays[f"audit_b_{cell}"], arrays[f"audit_projected_f_{cell}"]
        ids = arrays[f"face_dofs_{cell}"]
        operator = sparse.hstack((matrix, sparse.csr_matrix(coupling)), format="csr")
        defect = accurate_residual(operator, load, np.r_[pressure, trace[ids]])
        defects += np.sum(defect**2)
        rhs += np.sum(load.astype(np.longdouble) ** 2)
        np.add.at(
            weak,
            ids,
            -accurate_residual(sparse.csr_matrix(coupling.T), np.zeros(len(ids)), pressure),
        )
    scale = np.sqrt(rhs + np.sum(boundary**2))
    if scale <= 0:
        raise ValueError("Nonzero source or physical Dirichlet data required")
    return {
        "original_projected_source_relative_residual": float(
            np.sqrt(defects + np.sum(weak**2)) / scale
        ),
        "original_projected_source_residual_norm": float(np.sqrt(defects + np.sum(weak**2))),
        "physical_rhs_norm": float(scale),
        "duplicate_trace_absolute_difference": float(arrays["duplicate_trace_absolute_difference"]),
    }


def replay(
    arrays: Mapping[str, np.ndarray], cell: int, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate pressure, raw physical gradient and Darcy flux using stored tables only."""
    if (
        type(cell) is not int
        or not 0 <= cell < int(arrays["local_count"])
        or order not in arrays["norm_orders"]
    ):
        raise ValueError("Stored macrocell and executed quadrature order required")
    coefficients = restore(arrays, f"pressure_{cell}")[arrays[f"nodal_dofs_{cell}"]]
    pressure = np.einsum(
        "ti,qi->tq", coefficients, arrays[f"q{order}_cardinal_values"], optimize=False
    )
    reference_gradient = np.einsum(
        "ti,qia->tqa", coefficients, arrays[f"q{order}_cardinal_derivatives"], optimize=False
    )
    gradient = np.einsum(
        "tqa,tab->tqb", reference_gradient, arrays[f"barycentric_gradients_{cell}"], optimize=False
    )
    flux = -np.einsum(
        "tqab,tqb->tqa", arrays[f"q{order}_material_{cell}"], gradient, optimize=False
    )
    return pressure, gradient, flux


def write_field(
    path: Path,
    solution: MsHHO3DSolution,
    configuration: Mapping[str, Any],
    *,
    acquisition_uuid: str,
    source_sha256: Mapping[str, str],
    assembly_order: int,
    source: Any,
    dirichlet: Any = 0.0,
    norm_orders: tuple[int, ...] = (10, 12),
    display_refinement: int = 0,
) -> dict[str, Any]:
    """Atomically archive a fresh accepted field and its executed source/basis contract."""
    if (
        path.exists()
        or path.with_suffix(".json").exists()
        or not acquisition_uuid
        or not source_sha256
    ):
        raise ValueError("Fresh paths, acquisition UUID and executed source digests required")

    def check_sources() -> None:
        """Require every recorded numerical owner to retain its literal acquisition bytes."""
        for name, expected in source_sha256.items():
            actual = (source_file(name, root=ROOT)).resolve()
            if file_digest(local_resource(actual)) != expected:
                raise ValueError("Executed numerical source differs from its recorded bytes")

    check_sources()
    arrays = field_arrays(
        solution,
        assembly_order=assembly_order,
        source=source,
        dirichlet=dirichlet,
        norm_orders=norm_orders,
        display_refinement=display_refinement,
    )
    checks = original_checks(arrays)
    if checks["original_projected_source_relative_residual"] > 1e-10:
        raise LinearSolveError("Original physical projected-source residual exceeds1e-10")
    record = {
        "schema": SCHEMA,
        "configuration": dict(configuration),
        "acquisition_uuid": acquisition_uuid,
        "source_sha256": dict(source_sha256),
        "array_sha256": {k: array_identity(v) for k, v in arrays.items()},
        "coefficient_nmant": int(arrays["coefficient_nmant"]),
        "local_refinement_precision": solution.local_refinement_precision,
        "native_factor_precision": "binary64",
        "original_checks": checks,
        "original_operator_provenance": (
            "Diagnostic reassembly by shared tetra_operators; "
            "returned actual R/C/E are archived separately"
        ),
        "scope": (
            "Finite Galerkin P2/P0 projected-source method; no exact local solve, "
            "H(div) raw flux or uniform inf-sup assertion"
        ),
    }
    saved = checkpoint_field(path, arrays, record)
    check_sources()
    record.update(archive=path.name, archive_sha256=saved["archive_sha256"])
    write_progress(path.with_suffix(".json"), record)
    return record


def read_field(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify full-file/basis digests and physical contracts before data-only replay."""
    record = json.loads(read_resource_text(path.with_suffix(".json")))
    if (
        record.get("schema") != SCHEMA
        or record.get("archive") != path.name
        or file_digest(local_resource(path)) != record["archive_sha256"]
    ):
        raise ValueError("MsHHO3D field archive identity differs")
    with np.load(local_resource(path), allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if set(arrays) != set(record["array_sha256"]) or any(
        array_identity(v) != record["array_sha256"][k] for k, v in arrays.items()
    ):
        raise ValueError("Executed numerical basis or coefficient digest differs")
    if int(arrays["coefficient_nmant"]) != record["coefficient_nmant"]:
        raise ValueError("Declared coefficient precision differs")
    validate_arrays(arrays)
    if original_checks(arrays)["original_projected_source_relative_residual"] > 1e-10:
        raise LinearSolveError("Archived original physical equations fail1e-10")
    return arrays, record
