"""Persist two-dimensional MsHHO fields and their executed reconstruction matrices.

Replay evaluates the archived quadrature tables and physical barycentric maps.
The local energy reconstruction, cell/face moments and nodal coefficients have
separate roles. Moment equilibrium does not imply individual fine-cell balance,
an H(div) raw flux, or exact local solutions of the literature theorem.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from examples.archive_precision import precision_fields, restore_precision
from examples.formulations.local_records import DarcyLocalFactory as _DarcyLocalFactory
from examples.local_response_cache import array_identity
from examples.transport_checkpoints import checkpoint_field, write_progress
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data, p1_geometry, triangle_quadrature
from pymhm.fem.scalar.triangle import multiindices, nodal_space, reference_basis
from pymhm.io.provenance import file_digest
from pymhm.linalg.linear import LinearSolveError, accurate_residual
from pymhm.materials.evaluation import tensor_values
from pymhm.postprocessing.solutions import DarcySolution, MsHHOSolution

SCHEMA = "pymhm-mshho-field-archive-v2"
LEGACY_SCHEMA = "pymhm-mshho-field-archive-v1"
ROOT = Path(__file__).resolve().parents[1]


def field_arrays(solution: MsHHOSolution, order: int = 10) -> dict[str, np.ndarray]:
    """Capture actual nodal, moment and energy bases at a positive executed rule.

    ``order`` is the one-dimensional Duffy Gauss order, at least degree+1 for
    squared polynomial fields. Material tensors are stored at those physical
    quadrature points. Arbitrary unsaved sampling and material interpolation
    are outside this archive's replay contract.
    """
    degree = positive_int(solution.degree, "local degree")
    order = positive_int(order, "quadrature order")
    if order < degree + 1:
        raise ValueError("MsHHO field quadrature requires order at least local degree+1")
    if solution.source_variant not in ("projected", "reconstructed"):
        raise ValueError("MsHHO field archive requires an explicit source variant")
    if not solution.local or len(solution.pressure) != len(solution.local):
        raise ValueError("MsHHO field archive requires every local reconstruction")
    macro = solution.skeleton.mesh
    if np.asarray(macro.points).shape[1] != 2:
        raise ValueError("This MsHHO archive requires a two-dimensional physical mesh")
    bary, weights = triangle_quadrature(order)
    basis, derivative, _ = reference_basis(degree, bary)
    arrays = {
        "degree": np.asarray(degree),
        "quadrature_order": np.asarray(order),
        "macro_points": np.asarray(macro.points),
        "macro_faces": np.asarray(macro.faces),
        "macro_normals": np.asarray(macro.normals),
        "quadrature_barycentric": bary,
        "quadrature_weights": weights,
        "executed_cardinal_values": basis,
        "executed_cardinal_derivatives": derivative,
        "cardinal_multiindices": multiindices(degree),
        "local_count": np.asarray(len(solution.local)),
    }
    arrays.update(precision_fields("face_moments", solution.face_moments))
    for cell, (local, field, cell_moments) in enumerate(
        zip(solution.local, solution.pressure, solution.cell_moments, strict=True)
    ):
        fine = local.mesh
        dofs, nodes = nodal_space(fine, degree)
        face_dofs = solution.skeleton.cell_dofs(cell)
        coordinates = np.r_[cell_moments, solution.face_moments[face_dofs]]
        if (
            field.shape != (len(nodes),)
            or local.reconstruction.shape != (len(nodes), len(coordinates))
            or local.moments.shape != local.reconstruction.shape
            or local.cell_count != len(cell_moments)
        ):
            raise ValueError(
                "MsHHO coefficients do not belong to their executed local moment basis"
            )
        reconstructed = np.einsum("ij,j->i", local.reconstruction, coordinates, optimize=False)
        scale = max(float(np.linalg.norm(field)), np.finfo(float).tiny)
        if float(np.linalg.norm(field - reconstructed)) / scale > 1e-10:
            raise ValueError("MsHHO nodal field differs from its actual energy reconstruction")
        moments = np.einsum("ij,i->j", local.moments, field, optimize=False)
        moment_scale = max(float(np.linalg.norm(coordinates)), np.finfo(float).tiny)
        if float(np.linalg.norm(moments - coordinates)) / moment_scale > 1e-10:
            raise ValueError("MsHHO field does not reproduce its physical integral moments")
        gradient, area = p1_geometry(fine)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        tensors = tensor_values(solution.permeability, points.reshape(-1, 2)).reshape(
            *points.shape[:2], 2, 2
        )
        owned = {
            "points": fine.points,
            "cells": fine.cells,
            "nodal_dofs": dofs,
            "nodes": nodes,
            "physical_barycentric_gradients": gradient,
            "areas": area,
            "physical_quadrature_points": points,
            "permeability_tensors": tensors,
            "executed_moments": local.moments,
            "face_dofs": face_dofs,
            "cell_count": np.asarray(local.cell_count),
            "macro_vertices": macro.points[macro.cells[cell]],
            "macro_face_dofs": macro.cell_faces[cell],
            "macro_face_signs": macro.signs[cell],
        }
        arrays.update({f"{name}_{cell}": np.asarray(value) for name, value in owned.items()})
        for name, value in (
            ("executed_reconstruction", local.reconstruction),
            ("executed_energy", local.energy),
            ("executed_load", local.load),
        ):
            arrays.update(precision_fields(f"{name}_{cell}", value))
        arrays.update(precision_fields(f"pressure_{cell}", field))
        arrays.update(precision_fields(f"cell_moments_{cell}", cell_moments))
    if any(
        value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in arrays.values()
    ):
        raise ValueError("MsHHO archives require finite real executed arrays")
    return arrays


def restore(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Restore the three portable coefficient components of an executed field."""
    return restore_precision(arrays[name], arrays[name + "_correction"], arrays[name + "_tail"])


def _executed(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Retain portable matrix digits; legacy v1 matrices are binary64 only."""
    if name + "_correction" in arrays or name + "_tail" in arrays:
        if name + "_correction" not in arrays or name + "_tail" not in arrays:
            raise ValueError("MsHHO executed matrix precision components are incomplete")
        return restore(arrays, name)
    return arrays[name]


def _integer(arrays: Mapping[str, np.ndarray], name: str, minimum: int = 1) -> int:
    """Require a scalar integer header without truncating a floating value."""
    value = arrays[name]
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"MsHHO {name} requires an actual scalar integer")
    return positive_int(int(value), name, minimum)


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Check physical dof locations and the executed moment/reconstruction contract.

    Coherent reordering of basis columns and coefficients is admissible. A
    changed matrix with unchanged coordinates is rejected even if an archive
    checksum is recomputed. These are data-contract checks, not inf-sup or
    continuum error estimates.
    """
    if any(
        value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in arrays.values()
    ):
        raise ValueError("MsHHO archives require finite real executed arrays")
    degree = _integer(arrays, "degree")
    count = _integer(arrays, "local_count")
    order = _integer(arrays, "quadrature_order")
    if order < degree + 1:
        raise ValueError("MsHHO field quadrature requires order at least local degree+1")
    bary, weights = arrays["quadrature_barycentric"], arrays["quadrature_weights"]
    indices = arrays["cardinal_multiindices"]
    size = (degree + 1) * (degree + 2) // 2
    if (
        weights.shape != (order * order,)
        or bary.shape != (len(weights), 3)
        or indices.shape != (size, 3)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices.sum(axis=1) != degree)
        or len(np.unique(indices, axis=0)) != size
        or arrays["executed_cardinal_values"].shape != (len(weights), size)
        or arrays["executed_cardinal_derivatives"].shape != (len(weights), size, 3)
        or np.any(bary <= 0)
        or np.any(weights <= 0)
        or not np.allclose(bary.sum(axis=1), 1, rtol=0, atol=8 * np.finfo(float).eps)
        or not np.isclose(weights.sum(), 1, rtol=0, atol=8 * np.finfo(float).eps)
    ):
        raise ValueError("MsHHO quadrature or cardinal ordering contract differs")
    values = arrays["executed_cardinal_values"]
    derivative = arrays["executed_cardinal_derivatives"]
    # Reproduce the complete declared polynomial space using its nodal
    # moments. These independent monomial identities check saved tables;
    # they do not construct or substitute a fresh cardinal basis.
    nodal = indices[:, 1:] / degree
    x, y = bary[:, 1:].T
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            coefficient = nodal[:, 0] ** a * nodal[:, 1] ** b
            polynomial = x**a * y**b
            dx = a * x ** max(a - 1, 0) * y**b
            dy = b * x**a * y ** max(b - 1, 0)
            if (
                not np.allclose(values @ coefficient, polynomial, rtol=0, atol=1e-10)
                or not np.allclose(
                    (derivative[..., 1] - derivative[..., 0]) @ coefficient,
                    dx,
                    rtol=0,
                    atol=1e-10,
                )
                or not np.allclose(
                    (derivative[..., 2] - derivative[..., 0]) @ coefficient,
                    dy,
                    rtol=0,
                    atol=1e-10,
                )
            ):
                raise ValueError(
                    "MsHHO saved cardinal tables do not reproduce their declared Pk space"
                )
    face_moments = restore(arrays, "face_moments")
    for cell in range(count):
        points, cells = arrays[f"points_{cell}"], arrays[f"cells_{cell}"]
        dofs, nodes = arrays[f"nodal_dofs_{cell}"], arrays[f"nodes_{cell}"]
        ids = arrays[f"face_dofs_{cell}"]
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or cells.ndim != 2
            or cells.shape[1] != 3
            or cells.dtype.kind not in "iu"
            or np.any(cells < 0)
            or np.any(cells >= len(points))
            or nodes.ndim != 2
            or nodes.shape[1] != 2
            or dofs.shape != (len(cells), size)
            or dofs.dtype.kind not in "iu"
            or np.any(dofs < 0)
            or np.any(dofs >= len(nodes))
            or ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= len(face_moments))
        ):
            raise ValueError("MsHHO physical geometry or nodal/face dof ordering differs")
        vertices = points[cells]
        edges = vertices[:, 1:] - vertices[:, :1]
        determinant = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
        gradients = arrays[f"physical_barycentric_gradients_{cell}"]
        areas = arrays[f"areas_{cell}"]
        physical_points = arrays[f"physical_quadrature_points_{cell}"]
        material = arrays[f"permeability_tensors_{cell}"]
        if (
            np.any(determinant <= 0)
            or areas.shape != (len(cells),)
            or gradients.shape != (len(cells), 3, 2)
            or physical_points.shape != (len(cells), len(weights), 2)
            or material.shape != (len(cells), len(weights), 2, 2)
            or not np.allclose(areas, determinant / 2, rtol=64 * np.finfo(float).eps, atol=0)
        ):
            raise ValueError("MsHHO physical quadrature geometry or material shape differs")
        # Affine barycentric identities check the stored physical map without
        # replacing it with a freshly inverted matrix or rebuilding the basis.
        affine = np.einsum("tia,tib->tab", vertices, gradients, optimize=False)
        affine_scale = np.einsum(
            "tia,tib->tab", np.abs(vertices), np.abs(gradients), optimize=False
        )
        eps = 64 * np.finfo(float).eps
        if np.any(np.abs(affine - np.eye(2)) > eps * np.maximum(affine_scale, 1)) or np.any(
            np.abs(gradients.sum(axis=1)) > eps * np.maximum(np.abs(gradients).sum(axis=1), 1)
        ):
            raise ValueError("MsHHO stored physical barycentric gradients are incompatible")
        expected_nodes = np.einsum("ij,tja->tia", indices / degree, vertices, optimize=False)
        diameter = max(float(np.ptp(points, axis=0).max()), np.finfo(float).tiny)
        expected_points = np.einsum("qi,tia->tqa", bary, vertices, optimize=False)
        if not np.allclose(physical_points, expected_points, rtol=0, atol=1e-12 * diameter):
            raise ValueError("MsHHO quadrature points differ from their physical barycentric map")
        if not np.allclose(nodes[dofs], expected_nodes, rtol=0, atol=1e-12 * diameter):
            raise ValueError("MsHHO nodal coordinates differ from their physical cardinal dof map")
        field = restore(arrays, f"pressure_{cell}")
        local_moments = restore(arrays, f"cell_moments_{cell}")
        coordinates = np.r_[local_moments, face_moments[ids]]
        reconstruction = _executed(arrays, f"executed_reconstruction_{cell}")
        moments = arrays[f"executed_moments_{cell}"]
        if (
            field.shape != (len(nodes),)
            or reconstruction.shape != (len(nodes), len(coordinates))
            or moments.shape != reconstruction.shape
            or _integer(arrays, f"cell_count_{cell}", 0) != len(local_moments)
        ):
            raise ValueError("MsHHO local reconstruction coefficient contract differs")
        reconstructed = np.einsum("ij,j->i", reconstruction, coordinates, optimize=False)
        scale = max(float(np.linalg.norm(field)), np.finfo(float).tiny)
        if float(np.linalg.norm(field - reconstructed)) / scale > 1e-10:
            raise ValueError("MsHHO coefficients differ from their executed reconstruction matrix")
        recovered = np.einsum("ij,i->j", moments, field, optimize=False)
        scale = max(float(np.linalg.norm(coordinates)), np.finfo(float).tiny)
        if float(np.linalg.norm(recovered - coordinates)) / scale > 1e-10:
            raise ValueError("MsHHO coefficients differ from their declared physical moments")


def attach_mhm(
    arrays: dict[str, np.ndarray], solution: DarcySolution, dirichlet: Any
) -> dict[str, float]:
    """Retain both finite fields and check their original all-Dirichlet saddle rows.

    The comparison requires identical primal local spaces, geometry, tensors
    and a constant source in the MsHHO moment space. Original A/B/f/Z/C are
    reassembled by Darcy's shared numerical owner; no field is solved anew.
    The MsHHO physical trace is recovered from its represented Galerkin action,
    taking the first one-sided value and checking every original local row.
    No interface averaging or change of numerical tolerance is performed.
    """
    validate_arrays(arrays)
    degree = _integer(arrays, "degree")
    count = _integer(arrays, "local_count")
    macro = solution.skeleton.mesh
    if (
        solution.formulation != "primal"
        or solution.degree != degree
        or len(solution.local_meshes) != count
        or not np.array_equal(macro.points, arrays["macro_points"])
        or not np.array_equal(macro.faces, arrays["macro_faces"])
        or not np.isscalar(solution.source)
        or solution.point_sources
    ):
        raise ValueError("MHM--MsHHO replay requires matched primal meshes and a constant source")
    boundary, fixed = boundary_data(
        solution.skeleton, dirichlet, None, order=solution.quadrature_order
    )
    if fixed or len(solution.hybrid.gauge_multipliers):
        raise ValueError("This matched comparison requires only physical Dirichlet boundary data")
    factory = _DarcyLocalFactory(
        macro,
        solution.skeleton,
        solution.permeability,
        solution.source,
        tuple(np.empty((0, 3)) for _ in range(count)),
        1,
        solution.local_meshes,
        degree,
        "primal",
        solution.quadrature_order,
    )
    # Restored moment coordinates and their Galerkin actions may retain wider
    # digits. Physical trace recovery must not narrow those one-sided values.
    mshho_trace = np.zeros(solution.skeleton.size, dtype=np.longdouble)
    seen = np.zeros(solution.skeleton.size, dtype=bool)
    for cell, (fine, field) in enumerate(
        zip(solution.local_meshes, solution.pressure, strict=True)
    ):
        dofs, nodes = nodal_space(fine, degree)
        if (
            not np.array_equal(fine.points, arrays[f"points_{cell}"])
            or not np.array_equal(fine.cells, arrays[f"cells_{cell}"])
            or not np.array_equal(dofs, arrays[f"nodal_dofs_{cell}"])
            or not np.array_equal(nodes, arrays[f"nodes_{cell}"])
            or not np.array_equal(field, solution.hybrid.fields[cell])
        ):
            raise ValueError("MHM physical field does not match the saved local cardinal geometry")
        physical = arrays[f"physical_quadrature_points_{cell}"]
        material = tensor_values(solution.permeability, physical.reshape(-1, 2)).reshape(
            *physical.shape[:2], 2, 2
        )
        if not np.array_equal(material, arrays[f"permeability_tensors_{cell}"]):
            raise ValueError("MHM--MsHHO physical materials differ")
        problem = factory(cell).problem
        matrix = problem.matrix.tocsr()
        owned = {
            "original_a_data": matrix.data,
            "original_a_indices": matrix.indices,
            "original_a_indptr": matrix.indptr,
            "original_b": problem.coupling,
            "original_test_b": problem.test_coupling,
            "original_f": problem.load,
            "original_z": problem.kernel,
            "original_constraint": problem.constraints,
            "original_trace_dofs": problem.trace_dofs,
        }
        arrays.update({f"{name}_{cell}": np.asarray(value) for name, value in owned.items()})
        arrays.update(precision_fields(f"mhm_pressure_{cell}", field))
        coordinates = np.r_[
            restore(arrays, f"cell_moments_{cell}"),
            restore(arrays, "face_moments")[arrays[f"face_dofs_{cell}"]],
        ]
        moment_derivative = _executed(arrays, f"executed_energy_{cell}") @ coordinates - _executed(
            arrays, f"executed_load_{cell}"
        )
        signs = np.repeat(
            macro.signs[cell],
            [solution.skeleton.faces[face].size for face in macro.cell_faces[cell]],
        )
        local_trace = -signs * moment_derivative[_integer(arrays, f"cell_count_{cell}", 0) :]
        ids = problem.trace_dofs
        first = ~seen[ids]
        mshho_trace[ids[first]] = local_trace[first]
        seen[ids] = True
    if not seen.all():
        raise ValueError("Matched MHM--MsHHO comparison has an incomplete physical trace map")
    arrays.update(precision_fields("mhm_trace", solution.hybrid.trace))
    arrays.update(precision_fields("mshho_trace", mshho_trace))
    arrays["original_boundary_load"] = boundary
    return original_saddle_residuals(arrays)


def original_saddle_residuals(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Measure original local and weak trace rows in their physical right-hand-side norm.

    These diagnostics use archived numerical operators and the shared accurate
    sparse residual action, including represented nonzero kernel actions.
    They do not prove an inf-sup constant or fine-cell flux conservation.
    """
    count = _integer(arrays, "local_count")
    boundary = arrays["original_boundary_load"].astype(np.longdouble)
    diagnostics = {}
    for method in ("mhm", "mshho"):
        trace = restore(arrays, f"{method}_trace")
        weak = -boundary.copy()
        volume_squared = rhs_squared = np.longdouble(0)
        for cell in range(count):
            field = restore(
                arrays, f"mhm_pressure_{cell}" if method == "mhm" else f"pressure_{cell}"
            )
            size = len(field)
            matrix = sparse.csr_matrix(
                (
                    arrays[f"original_a_data_{cell}"],
                    arrays[f"original_a_indices_{cell}"],
                    arrays[f"original_a_indptr_{cell}"],
                ),
                shape=(size, size),
            )
            coupling = arrays[f"original_b_{cell}"]
            ids = arrays[f"original_trace_dofs_{cell}"]
            load = arrays[f"original_f_{cell}"]
            operator = sparse.hstack((matrix, sparse.csr_matrix(coupling)), format="csr")
            defect = accurate_residual(operator, load, np.r_[field, trace[ids]])
            volume_squared += np.sum(defect**2)
            rhs_squared += np.sum(load.astype(np.longdouble) ** 2)
            test = arrays[f"original_test_b_{cell}"]
            local_weak = -accurate_residual(sparse.csr_matrix(test.T), np.zeros(len(ids)), field)
            np.add.at(weak, ids, local_weak)
        rhs_norm = float(np.sqrt(rhs_squared + np.sum(boundary**2)))
        residual = float(np.sqrt(volume_squared + np.sum(weak**2)))
        if rhs_norm <= 0 or not np.isfinite(residual):
            raise ValueError("Original comparison requires a nonzero physical right-hand side")
        diagnostics[f"{method}_original_relative_residual"] = residual / rhs_norm
        diagnostics[f"{method}_original_residual_norm"] = residual
        diagnostics[f"{method}_original_rhs_norm"] = rhs_norm
    return diagnostics


def replay(
    arrays: Mapping[str, np.ndarray], cell: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pressure, raw physical gradient and Darcy flux at the stored rule.

    This operation uses the saved tables, coefficients, dof order and material
    values. It computes no new reconstruction, basis or interface average.
    Contraction operands use C layout so equivalent archive layouts preserve
    the accumulation order, including on hosts with extended precision.
    """
    cell = positive_int(cell, "macro cell", 0)
    if cell >= int(arrays["local_count"]):
        raise ValueError("MsHHO replay cell is outside the executed macro partition")
    nodal = np.ascontiguousarray(restore(arrays, f"pressure_{cell}")[arrays[f"nodal_dofs_{cell}"]])
    values = np.ascontiguousarray(arrays["executed_cardinal_values"])
    derivatives = np.ascontiguousarray(arrays["executed_cardinal_derivatives"])
    barycentric_gradients = np.ascontiguousarray(arrays[f"physical_barycentric_gradients_{cell}"])
    material = np.ascontiguousarray(arrays[f"permeability_tensors_{cell}"])
    pressure = np.einsum("ti,qi->tq", nodal, values, optimize=False)
    derivative = np.einsum("ti,qia->tqa", nodal, derivatives, optimize=False)
    gradient = np.einsum("tqa,tab->tqb", derivative, barycentric_gradients, optimize=False)
    flux = -np.einsum("tqab,tqb->tqa", material, gradient, optimize=False)
    return pressure, gradient, flux


def write_field(
    path: Path,
    solution: MsHHOSolution,
    configuration: Mapping[str, Any],
    *,
    acquisition_uuid: str,
    source_sha256: Mapping[str, str],
    order: int = 10,
    mhm: DarcySolution | None = None,
    dirichlet: Any = None,
) -> dict[str, Any]:
    """Atomically save one fresh field and a separate immutable acquisition contract."""
    contract = path.with_suffix(".json")
    if path.exists() or contract.exists():
        raise ValueError("MsHHO acquisition requires fresh coefficient and contract paths")
    if not acquisition_uuid or not source_sha256:
        raise ValueError("MsHHO fields require their actual acquisition UUID and numerical sources")

    def check_sources() -> None:
        for name, expected in source_sha256.items():
            owned = (ROOT / name).resolve()
            if not owned.is_relative_to(ROOT) or file_digest(owned) != expected:
                raise ValueError("MsHHO executed numerical source differs from its recorded bytes")

    check_sources()
    json.dumps(dict(configuration), allow_nan=False)
    arrays = field_arrays(solution, order)
    validate_arrays(arrays)
    diagnostics = {} if mhm is None else attach_mhm(arrays, mhm, dirichlet)
    for method in ("mhm", "mshho"):
        residual = diagnostics.get(f"{method}_original_relative_residual", 0.0)
        if residual > 1e-10:
            raise LinearSolveError(
                f"{method} original physical saddle residual {residual:.6e} exceeds 1e-10"
            )
    record = {
        "schema": SCHEMA,
        "acquisition_uuid": acquisition_uuid,
        "configuration": dict(configuration),
        "source_variant": solution.source_variant,
        "local_refinement_precision": solution.local_refinement_precision,
        "coefficient_mantissa_bits": int(np.finfo(solution.pressure[0].dtype).nmant),
        "native_factor_precision": "binary64",
        "source_sha256": dict(source_sha256),
        "array_sha256": {name: array_identity(value) for name, value in arrays.items()},
        "moment_residual": float(solution.residual),
        "original_saddle_diagnostics": diagnostics,
        "scope": (
            "Finite Galerkin moment reconstruction; exact local solves and "
            "individual fine-cell conservation are not asserted"
        ),
    }
    saved = checkpoint_field(path, arrays, record)
    check_sources()
    record.update(archive=path.name, archive_sha256=saved["archive_sha256"])
    write_progress(contract, record)
    return record


def read_field(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Require every actual field, geometry, moment and numerical-basis digest before replay."""
    record = json.loads(path.with_suffix(".json").read_text())
    if record.get("schema") not in (SCHEMA, LEGACY_SCHEMA) or record.get("archive") != path.name:
        raise ValueError("MsHHO field schema or archive identity differs")
    if file_digest(path) != record["archive_sha256"]:
        raise ValueError("MsHHO coefficient archive digest changed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if set(arrays) != set(record["array_sha256"]):
        raise ValueError("MsHHO executed array contract is incomplete")
    for name, value in arrays.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("MsHHO coefficient archive requires finite real arrays")
        if array_identity(value) != record["array_sha256"][name]:
            raise ValueError("MsHHO executed basis, geometry or field digest changed")
    if record["schema"] == SCHEMA:
        for cell in range(_integer(arrays, "local_count")):
            for name in ("executed_reconstruction", "executed_energy", "executed_load"):
                if f"{name}_{cell}_correction" not in arrays or f"{name}_{cell}_tail" not in arrays:
                    raise ValueError(
                        "MsHHO v2 requires actual portable matrix precision components"
                    )
    validate_arrays(arrays)
    if record.get("original_saddle_diagnostics"):
        actual = original_saddle_residuals(arrays)
        for method in ("mhm", "mshho"):
            if actual[f"{method}_original_relative_residual"] > 1e-10:
                raise ValueError("Archived field fails the original physical saddle equations")
    return arrays, record
