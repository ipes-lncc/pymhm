"""Executed operators, bases and one-sided fields of a recursive Cartesian MHM.

The archive contains the actual responses used by the acquisition, including
parent boundary reactions. Leaf traces are injected into the flat mesh's
oriented P1 coordinates; duplicate interfaces are checked, never averaged.
Portable high/correction/tail arrays retain any wider coefficient arithmetic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

try:
    from .archive_precision import precision_fields, restore_precision
except ImportError:
    from archive_precision import precision_fields, restore_precision
from examples.transport_checkpoints import checkpoint_field, write_progress
from pymhm.hybrid import HybridSolution, HybridSystem, LocalResponse
from pymhm.mesh import SkeletonSpace
from pymhm.nested import NestedLocalProblem, NestedSolution
from pymhm.quadrilateral import CartesianMacroMesh, _cardinals, qk_basis, qk_space

SCHEMA = "pymhm-nested-field-archive-v1"


def array_digest(values: np.ndarray) -> str:
    """Bind an array's dtype, shape and logical coefficient bytes to SHA256."""
    value = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def restore(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Restore the complete portable arithmetic of one archived real array."""
    return restore_precision(arrays[name], arrays[f"{name}_correction"], arrays[f"{name}_tail"])


def _responses(
    arrays: dict[str, np.ndarray],
    prefix: str,
    responses: Sequence[LocalResponse],
    fields: Sequence[np.ndarray],
    traces: Sequence[np.ndarray],
    coarse: Sequence[np.ndarray],
) -> None:
    """Copy original local data and their executed responses without recondensation."""
    data = {
        "A": [r.problem.matrix.toarray() for r in responses],
        "B_original": [r.problem.coupling for r in responses],
        "f": [r.problem.load for r in responses],
        "Z": [r.problem.coarse_basis for r in responses],
        "C": [r.problem.constraints for r in responses],
        "W": [r.problem.test_basis for r in responses],
        "test_C": [r.problem.test_constraints for r in responses],
        "test_B": [r.problem.test_coupling for r in responses],
        "E": [r.retained_basis for r in responses],
        "source": [r.source for r in responses],
        "lifts": [r.lifts for r in responses],
        "pressure": fields,
        "local_trace": traces,
        "coarse": coarse,
    }
    for name, values in data.items():
        arrays.update(precision_fields(f"{prefix}_{name}", np.asarray(values)))
    arrays[f"{prefix}_original_trace_dofs"] = np.asarray([r.problem.trace_dofs for r in responses])
    arrays[f"{prefix}_corrected_retained"] = np.asarray(
        [r.coarse_vectors is not None for r in responses]
    )


def replay_responses(arrays: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Reconstruct from the saved source, lifts and executed retained matrix E."""
    return (
        restore(arrays, f"{prefix}_source")
        - np.einsum(
            "cij,cj->ci",
            restore(arrays, f"{prefix}_lifts"),
            restore(arrays, f"{prefix}_local_trace"),
        )
        + np.einsum(
            "cij,cj->ci", restore(arrays, f"{prefix}_E"), restore(arrays, f"{prefix}_coarse")
        )
    )


def capture_display(arrays: dict[str, np.ndarray]) -> None:
    """Capture executed Q2 sampling tables and their one-sided physical injection maps."""
    line = np.linspace(0, 1, 17)
    xx, yy = np.meshgrid(line, line)
    unit = np.column_stack((xx.ravel(), yy.ravel()))
    points, references, dofs, basis, gradient = [], [], [], [], []
    for count, vertices in enumerate(arrays["leaf_points"]):
        spacing = (vertices[8] - vertices[0]) / 2
        physical = vertices[0] + unit * (vertices[8] - vertices[0])
        coordinates = (physical - vertices[0]) / spacing
        indices = np.minimum(coordinates.astype(int), 1)
        reference = np.clip(coordinates - indices, 0, 1)
        local_basis, local_gradient = qk_basis(2, reference)
        cells = indices[:, 1] * 2 + indices[:, 0]
        points.append(physical)
        references.append(reference)
        dofs.append(arrays["leaf_dofs"][count, cells])
        basis.append(local_basis)
        gradient.append(local_gradient / spacing)
    arrays.update(
        display_points=np.asarray(points),
        display_reference=np.asarray(references),
        display_dofs=np.asarray(dofs),
        display_basis=np.asarray(basis),
        display_gradient=np.asarray(gradient),
    )


def display_fields(
    arrays: dict[str, np.ndarray], kind: str = "recursive"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay selected publication samples through saved tables without fresh tabulation."""
    if kind not in {"recursive", "flat"}:
        raise ValueError("recursive or flat displayed field required")
    coefficients = restore(arrays, f"leaf_pressure_{kind}")
    values = coefficients[np.arange(len(coefficients))[:, None, None], arrays["display_dofs"]]
    return (
        arrays["display_points"],
        np.einsum("cqj,cqj->cq", values, arrays["display_basis"]),
        -np.einsum("cqj,cqja->cqa", values, arrays["display_gradient"]),
    )


def _face_transform(
    endpoints: np.ndarray, normal: np.ndarray, flat_endpoints: np.ndarray, flat_normal: np.ndarray
) -> np.ndarray:
    """Convert canonical flat normal-flux coefficients into one inner P1 basis."""
    same = np.array_equal(endpoints, flat_endpoints)
    flipped = np.array_equal(endpoints, flat_endpoints[::-1])
    if not same and not flipped:
        raise ValueError("identical oriented physical endpoints are required")
    sign = float(normal @ flat_normal)
    if abs(sign) != 1:
        raise ValueError("identical or opposite physical normal required")
    return np.diag([sign, sign if same else -sign])


def executed_arrays(
    n: int,
    outer: SkeletonSpace,
    parent: HybridSystem,
    solution: HybridSolution,
    children: Sequence[NestedLocalProblem],
    recovered: Sequence[NestedSolution],
    flat: HybridSystem,
    flat_solution: HybridSolution,
    flat_skeleton: SkeletonSpace,
    boundary_load: np.ndarray,
    parent_boundary_load: np.ndarray,
) -> dict[str, np.ndarray]:
    """Archive actual nested/flat objects, original equations and physical maps.

    Leaf order is x-fast on the (2n)-square flat macro mesh. B_original and its
    executed inner trace are retained. B is that same operator expressed in
    the archived flat normal/parameter coordinates by a signed injection.
    """
    count = (2 * n) ** 2
    arrays: dict[str, np.ndarray] = {}
    leaf_responses: list[Any] = [None] * count
    leaf_fields: list[Any] = [None] * count
    leaf_traces: list[Any] = [None] * count
    leaf_coarse: list[Any] = [None] * count
    meshes: list[Any] = [None] * count
    trace_maps = np.empty((count, 8, 8))
    flat_mesh = flat_skeleton.mesh
    endpoints = flat_mesh.points[flat_mesh.faces]
    face_ids = {tuple(sorted(map(tuple, edge))): i for i, edge in enumerate(endpoints)}
    trace = np.zeros(flat_skeleton.size)
    seen = np.zeros(len(endpoints), dtype=bool)
    duplicate_defect = 0.0
    for cell, (child, inner_result) in enumerate(zip(children, recovered, strict=True)):
        inner_mesh = CartesianMacroMesh(2, bounds=outer.mesh.submesh(cell, 1).bounds)
        inner_skeleton = SkeletonSpace(
            inner_mesh, tuple(flat_skeleton.faces[0] for _ in inner_mesh.faces)
        )
        face_transforms = {}
        for face, edge in enumerate(inner_mesh.points[inner_mesh.faces]):
            fid = face_ids[tuple(sorted(map(tuple, edge)))]
            transform = _face_transform(
                edge, inner_mesh.normals[face], endpoints[fid], flat_mesh.normals[fid]
            )
            face_transforms[face] = transform
            value = transform @ inner_result.unknowns[inner_skeleton.dofs(face)]
            dofs = flat_skeleton.dofs(fid)
            if seen[fid]:
                duplicate_defect = max(duplicate_defect, float(np.max(abs(trace[dofs] - value))))
            else:
                trace[dofs] = value
                seen[fid] = True
        for subcell, response in enumerate(child.inner.responses):
            i, j = 2 * (cell % n) + subcell % 2, 2 * (cell // n) + subcell // 2
            index = j * 2 * n + i
            leaf_responses[index] = response
            leaf_fields[index] = inner_result.fields[subcell]
            leaf_traces[index] = inner_result.unknowns[response.problem.trace_dofs]
            start, end = child.inner.kernel_offsets[subcell : subcell + 2]
            leaf_coarse[index] = inner_result.unknowns[start:end]
            meshes[index] = child.inner.local_metadata[subcell][0]
            transform = np.zeros((8, 8))
            for side, face in enumerate(inner_mesh.cell_faces[subcell]):
                transform[2 * side : 2 * side + 2, 2 * side : 2 * side + 2] = face_transforms[
                    int(face)
                ]
            trace_maps[index] = transform
    if not seen.all() or duplicate_defect > 1e-11 * max(
        np.linalg.norm(trace), np.finfo(float).tiny
    ):
        raise ValueError("all recursive faces must inject consistently without averaging")
    _responses(arrays, "leaf", leaf_responses, leaf_fields, leaf_traces, leaf_coarse)
    _responses(
        arrays,
        "flat",
        flat.responses,
        flat_solution.fields,
        [flat_solution.trace[r.problem.trace_dofs] for r in flat.responses],
        flat_solution.coarse,
    )
    _responses(
        arrays,
        "parent",
        parent.responses,
        solution.fields,
        [solution.trace[r.problem.trace_dofs] for r in parent.responses],
        solution.coarse,
    )
    for name, values in {
        "leaf_B": np.einsum("cij,cjk->cik", restore(arrays, "leaf_B_original"), trace_maps),
        "leaf_pressure_recursive": np.asarray(leaf_fields),
        "leaf_pressure_flat": np.asarray(flat_solution.fields),
        "trace_recursive": trace,
        "trace_flat": flat_solution.trace,
        "boundary_load": boundary_load,
        "parent_boundary_load": parent_boundary_load,
        "outer_trace": solution.trace,
        "inner_unknowns": np.asarray([v.unknowns for v in recovered]),
        "inner_boundary_reactions": np.asarray([v.boundary_reactions for v in recovered]),
        "inner_A": np.asarray([child.inner.matrix.toarray() for child in children]),
        "inner_f": np.asarray([child.inner.rhs for child in children]),
        "nested_trace_map": np.asarray([child.trace_map for child in children]),
        "parent_physical_moment": np.asarray(
            [
                child.moment([metadata[1] for metadata in child.inner.local_metadata])[0]
                for child in children
            ]
        ),
        "parent_physical_moment_offset": np.asarray(
            [
                child.moment([metadata[1] for metadata in child.inner.local_metadata])[1]
                for child in children
            ]
        ),
    }.items():
        arrays.update(precision_fields(name, values))
    arrays.update(
        n=np.asarray(n, dtype=np.int64),
        leaf_degree=np.asarray(2, dtype=np.int64),
        leaf_refinement=np.asarray([2, 2], dtype=np.int64),
        leaf_points=np.asarray([mesh.points for mesh in meshes]),
        leaf_cells=np.asarray([mesh.cells for mesh in meshes]),
        leaf_nodes=np.asarray([qk_space(mesh, 2)[1] for mesh in meshes]),
        leaf_dofs=np.asarray([qk_space(mesh, 2)[0] for mesh in meshes]),
        leaf_trace_dofs=np.asarray([r.problem.trace_dofs for r in flat.responses]),
        leaf_trace_transform=trace_maps,
        nested_boundary_dofs=np.asarray([child.boundary_dofs for child in children]),
        inner_kernel_offsets=np.asarray([child.inner.kernel_offsets for child in children]),
        flat_face_endpoints=endpoints,
        flat_face_normals=flat_mesh.normals,
        flat_cell_faces=flat_mesh.cell_faces,
        flat_cell_signs=flat_mesh.signs,
        outer_face_endpoints=outer.mesh.points[outer.mesh.faces],
        outer_face_normals=outer.mesh.normals,
        outer_cell_faces=outer.mesh.cell_faces,
        outer_cell_signs=outer.mesh.signs,
        q2_cardinal_matrix=np.kron(_cardinals(2), _cardinals(2)),
        q2_monomial_powers=np.asarray([(x, y) for y in range(3) for x in range(3)]),
        recursive_duplicate_trace_defect=np.asarray(duplicate_defect),
    )
    # These tables belong to the executed normal-flux basis, including its
    # parameter convention; later comparisons need no current FaceSpace.
    arrays["trace_cardinal_matrix"] = flat_skeleton.faces[0].evaluate(
        np.array([0.0, 1.0])
    ).T @ np.array([[1.0, -1.0], [0.0, 1.0]])
    for order in (8, 10):
        parameter, weights = flat_skeleton.faces[0].quadrature(order)
        arrays[f"trace_q{order}_parameter"] = parameter
        arrays[f"trace_q{order}_weights"] = weights
        arrays[f"trace_q{order}_basis"] = flat_skeleton.faces[0].evaluate(parameter)
    return arrays


def evaluate(
    arrays: dict[str, np.ndarray], kind: str, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate saved leaf coefficients through the actual saved Q2 quadrature tables."""
    if kind not in {"recursive", "flat"} or order not in {8, 10}:
        raise ValueError("recursive/flat fields and executed norm orders8/10 required")
    coefficients = restore(arrays, f"leaf_pressure_{kind}")
    values = coefficients[np.arange(len(coefficients))[:, None, None], arrays["leaf_dofs"]]
    basis = arrays[f"q{order}_basis"]
    gradient = arrays[f"q{order}_gradient"]
    vertices = arrays["leaf_points"][
        np.arange(len(coefficients))[:, None, None], arrays["leaf_cells"]
    ]
    origins = vertices[:, :, 0]
    widths = vertices[:, :, 2] - origins
    physical = origins[:, :, None] + arrays[f"q{order}_reference"][None, None] * widths[:, :, None]
    pressure = np.einsum("cfj,qj->cfq", values, basis)
    raw_gradient = np.einsum("cfj,qja->cfqa", values, gradient) / widths[:, :, None]
    measure = np.prod(widths, axis=-1)[:, :, None] * arrays[f"q{order}_weights"][None, None]
    return physical, pressure, raw_gradient, measure


def field_norms(
    arrays: dict[str, np.ndarray],
    exact: Callable[[np.ndarray], np.ndarray],
    gradient: Callable[[np.ndarray], np.ndarray],
    order: int,
) -> dict[str, float]:
    """Integrate raw flux and pressure differences on independent one-sided leaf cells."""
    points, recursive, raw, measure = evaluate(arrays, "recursive", order)
    _, flat, flat_raw, _ = evaluate(arrays, "flat", order)
    reference = exact(points.reshape(-1, 2)).reshape(recursive.shape)
    reference_raw = gradient(points.reshape(-1, 2)).reshape(raw.shape)

    def square(values: np.ndarray) -> np.longdouble:
        """Accumulate each physical-cell measure in wider arithmetic."""
        return np.sum(measure * values, dtype=np.longdouble)

    return {
        "pressure_l2": float(np.sqrt(square((recursive - reference) ** 2))),
        "flux_l2": float(np.sqrt(square(np.sum((raw - reference_raw) ** 2, axis=-1)))),
        "pressure_relative_flat_difference": float(
            np.sqrt(square((recursive - flat) ** 2) / square(flat**2))
        ),
        "raw_flux_relative_flat_difference": float(
            np.sqrt(
                square(np.sum((raw - flat_raw) ** 2, axis=-1))
                / square(np.sum(flat_raw**2, axis=-1))
            )
        ),
    }


def original_checks(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Check original volume equations and assembled physical continuity against f/g.

    Denominators contain source and prescribed-boundary norms, without an
    operator-action denominator or cancellation allowance. Dirichlet data fix
    the global pressure constant; local physical moment constraints remain.
    """
    result = {}
    for prefix in ("leaf", "flat", "parent"):
        fields = restore(arrays, f"{prefix}_pressure")
        forcing = restore(arrays, f"{prefix}_f")
        residual = np.einsum("cij,cj->ci", restore(arrays, f"{prefix}_A"), fields)
        residual += np.einsum(
            "cij,cj->ci",
            restore(arrays, f"{prefix}_B_original"),
            restore(arrays, f"{prefix}_local_trace"),
        )
        residual -= forcing
        result[f"{prefix}_original_source_relative_residual"] = float(
            np.linalg.norm(residual) / max(np.linalg.norm(forcing), np.finfo(float).tiny)
        )
    for prefix, pressure, boundary in (
        ("leaf", "recursive", "boundary_load"),
        ("flat", "flat", "boundary_load"),
        ("parent", "parent", "parent_boundary_load"),
    ):
        fields = restore(
            arrays, "parent_pressure" if prefix == "parent" else f"leaf_pressure_{pressure}"
        )
        coupling = restore(arrays, "leaf_B" if prefix == "leaf" else f"{prefix}_B_original")
        dofs = arrays["leaf_trace_dofs" if prefix == "leaf" else f"{prefix}_original_trace_dofs"]
        prescribed = restore(arrays, boundary)
        moments = np.zeros_like(prescribed)
        np.add.at(moments, dofs, np.einsum("cij,ci->cj", coupling, fields))
        result[f"{prefix}_original_constraint_relative_residual"] = float(
            np.linalg.norm(moments - prescribed)
            / max(
                np.linalg.norm(prescribed),
                np.linalg.norm(restore(arrays, f"{prefix}_f")),
                np.finfo(float).tiny,
            )
        )
    return result


def validate_layout(arrays: dict[str, np.ndarray]) -> None:
    """Require the declared scalar Q2/r2 coefficient, trace and basis dimensions."""
    level = arrays.get("n")
    degree = arrays.get("leaf_degree", np.empty(0))
    if (
        level is None
        or level.shape != ()
        or level.dtype.kind not in "iu"
        or int(level) not in {1, 2, 4, 8, 16}
        or degree.shape != ()
        or degree.dtype.kind not in "iu"
        or int(degree) != 2
        or not np.array_equal(arrays.get("leaf_refinement"), [2, 2])
    ):
        raise ValueError("declared supported level and scalar Q2/r2 archive required")
    n = int(level)
    count = (2 * n) ** 2
    faces = 2 * (2 * n) * (2 * n + 1)
    required = {
        "leaf_points": (count, 9, 2),
        "leaf_cells": (count, 4, 4),
        "leaf_nodes": (count, 25, 2),
        "leaf_dofs": (count, 4, 9),
        "leaf_A": (count, 25, 25),
        "leaf_B": (count, 25, 8),
        "leaf_f": (count, 25),
        "leaf_E": (count, 25, 1),
        "leaf_trace_dofs": (count, 8),
        "flat_face_endpoints": (faces, 2, 2),
        "flat_face_normals": (faces, 2),
        "flat_cell_faces": (count, 4),
        "flat_cell_signs": (count, 4),
        "q2_cardinal_matrix": (9, 9),
        "q2_monomial_powers": (9, 2),
        "outer_face_endpoints": (2 * n * (n + 1), 2, 2),
        "outer_face_normals": (2 * n * (n + 1), 2),
        "outer_cell_faces": (n * n, 4),
        "outer_cell_signs": (n * n, 4),
        "display_points": (count, 289, 2),
        "display_reference": (count, 289, 2),
        "display_dofs": (count, 289, 9),
        "display_basis": (count, 289, 9),
        "display_gradient": (count, 289, 9, 2),
        "leaf_trace_transform": (count, 8, 8),
        "inner_A": (n * n, 28, 28),
        "inner_f": (n * n, 28),
        "inner_unknowns": (n * n, 28),
        "inner_boundary_reactions": (n * n, 16),
        "nested_boundary_dofs": (n * n, 16),
        "nested_trace_map": (n * n, 16, 16),
        "inner_kernel_offsets": (n * n, 5),
        "parent_physical_moment": (n * n, 44),
        "parent_physical_moment_offset": (n * n,),
        "trace_cardinal_matrix": (2, 2),
    }
    for prefix, cells, size, trace in (
        ("leaf", count, 25, 8),
        ("flat", count, 25, 8),
        ("parent", n * n, 44, 16),
    ):
        for name, shape in {
            "A": (cells, size, size),
            "B_original": (cells, size, trace),
            "f": (cells, size),
            "Z": (cells, size, 1),
            "C": (cells, size, 1),
            "W": (cells, size, 1),
            "test_C": (cells, size, 1),
            "test_B": (cells, size, trace),
            "E": (cells, size, 1),
            "source": (cells, size),
            "lifts": (cells, size, trace),
            "pressure": (cells, size),
            "local_trace": (cells, trace),
            "coarse": (cells, 1),
        }.items():
            required.update(
                {f"{prefix}_{name}{suffix}": shape for suffix in ("", "_correction", "_tail")}
            )
        required[f"{prefix}_original_trace_dofs"] = (cells, trace)
        required[f"{prefix}_corrected_retained"] = (cells,)
    for field in ("leaf_pressure_recursive", "leaf_pressure_flat", "trace_recursive", "trace_flat"):
        shape = (count, 25) if field.startswith("leaf") else (2 * faces,)
        required.update({field + suffix: shape for suffix in ("", "_correction", "_tail")})
    for field, shape in tuple(required.items()):
        if field.startswith(
            (
                "inner_A",
                "inner_f",
                "inner_unknowns",
                "inner_boundary_reactions",
                "nested_trace_map",
                "parent_physical_moment",
            )
        ):
            required.update({field + suffix: shape for suffix in ("_correction", "_tail")})
    for field, shape in (
        ("boundary_load", (2 * faces,)),
        ("parent_boundary_load", (8 * n * (n + 1),)),
        ("outer_trace", (8 * n * (n + 1),)),
    ):
        required.update({field + suffix: shape for suffix in ("", "_correction", "_tail")})
    for order in (6, 8, 10):
        required.update(
            {
                f"q{order}_reference": (order**2, 2),
                f"q{order}_weights": (order**2,),
                f"q{order}_basis": (order**2, 9),
                f"q{order}_gradient": (order**2, 9, 2),
            }
        )
    for order in (8, 10):
        required.update(
            {
                f"trace_q{order}_parameter": (order,),
                f"trace_q{order}_weights": (order,),
                f"trace_q{order}_basis": (order, 2),
            }
        )
    if any(name not in arrays or arrays[name].shape != shape for name, shape in required.items()):
        raise ValueError("executed coefficient/basis dimensions are incompatible")
    for name, bound in (
        ("leaf_cells", 9),
        ("leaf_dofs", 25),
        ("display_dofs", 25),
        ("leaf_trace_dofs", 2 * faces),
        ("nested_boundary_dofs", 24),
        ("flat_cell_faces", faces),
        ("outer_cell_faces", 2 * n * (n + 1)),
        ("leaf_original_trace_dofs", 24),
        ("flat_original_trace_dofs", 2 * faces),
        ("parent_original_trace_dofs", 8 * n * (n + 1)),
    ):
        values = arrays[name]
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= bound):
            raise ValueError("executed physical injection maps are incompatible")
    for name in ("archive_schema_version", "coefficient_precision_bits"):
        value = arrays.get(name, np.empty(0))
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError("executed schema/UUID/precision identity is incompatible")
    if (
        arrays.get("acquisition_uuid", np.empty(0)).shape != (36,)
        or arrays["acquisition_uuid"].dtype != np.uint8
    ):
        raise ValueError("executed schema/UUID/precision identity is incompatible")


def _require_close(
    actual: np.ndarray, expected: np.ndarray, description: str, tolerance: float = 2e-11
) -> None:
    """Check an archived invariant with a relative floating-point evaluation bound."""
    scale = max(float(np.linalg.norm(expected)), 1.0)
    if actual.shape != expected.shape or np.linalg.norm(actual - expected) > tolerance * scale:
        raise ValueError(f"executed {description} is incompatible")


def _cardinal_tables(
    arrays: dict[str, np.ndarray], points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the archived monomial matrix independently, without current basis code."""
    powers = arrays["q2_monomial_powers"]
    matrix = arrays["q2_cardinal_matrix"]
    monomials = np.prod(points[..., None, :] ** powers, axis=-1)
    derivatives = []
    for axis in range(2):
        exponent = powers.copy()
        factor = exponent[:, axis].copy()
        exponent[:, axis] = np.maximum(exponent[:, axis] - 1, 0)
        derivatives.append(np.prod(points[..., None, :] ** exponent, axis=-1) * factor)
    return monomials @ matrix.T, np.stack(derivatives, axis=-1).swapaxes(-1, -2) @ matrix.T


def _physical_boundary(
    endpoints: np.ndarray, normals: np.ndarray, affine: bool, segments: int
) -> np.ndarray:
    """Integrate the declared affine exterior pressure in oriented P1 moment coordinates."""
    result = np.zeros((len(endpoints), 2 * segments))
    for face, (edge, normal) in enumerate(zip(endpoints, normals, strict=True)):
        axis = 0 if edge[0, 0] == edge[1, 0] else 1
        location = edge[0, axis]
        if location not in {0.0, 1.0}:
            continue
        outward = np.zeros(2)
        outward[axis] = -1 if location == 0 else 1
        sign = normal @ outward
        length = np.linalg.norm(edge[1] - edge[0]) / segments
        for part in range(segments):
            ends = edge[0] + (edge[1] - edge[0]) * np.array([part, part + 1])[:, None] / segments
            g = 1 + ends[:, 0] - ends[:, 1] if affine else np.zeros(2)
            result[face, 2 * part : 2 * part + 2] = (
                sign * length * np.array([(g[0] + g[1]) / 2, (g[1] - g[0]) / 6])
            )
    return result.ravel()


def validate_execution(arrays: dict[str, np.ndarray], record: dict[str, Any]) -> None:
    """Validate saved bases, geometry, moments and original equations without solving.

    Cardinality and derivative checks use the saved monomial coefficients.
    Local operators are checked by integration through saved assembly tables;
    response and nested checks use actual saved E/source/lift coordinates.
    No basis is regenerated or recondensed during this replay validation.
    """
    validate_layout(arrays)
    if any(
        value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in arrays.values()
    ):
        raise ValueError("finite portable real/integer archive arrays required")
    try:
        identifier = UUID(record["acquisition_uuid"])
        valid_identity = (
            record["schema"] == 2
            and record["field_archive_schema"] == SCHEMA
            and str(identifier) == record["acquisition_uuid"]
            and bytes(arrays["acquisition_uuid"]).decode("ascii") == str(identifier)
            and int(arrays["archive_schema_version"]) == 1
            and int(arrays["coefficient_precision_bits"])
            == record["coefficient_precision_bits"]
            == 53
            and record["n"] == int(arrays["n"])
            and record["boundary_case"] in {"sine_zero_dirichlet", "affine_plus_sine"}
            and record["assembly_quadrature_order"] == 6
            and record["boundary_quadrature_order"] == 8
        )
    except (KeyError, ValueError, TypeError, UnicodeError):
        valid_identity = False
    if not valid_identity:
        raise ValueError("executed schema/UUID/precision identity is incompatible")
    powers = np.asarray([(x, y) for y in range(3) for x in range(3)])
    if not np.array_equal(arrays["q2_monomial_powers"], powers):
        raise ValueError("executed Q2 monomial exponents are incompatible")
    nodes = np.asarray([(x / 2, y / 2) for y in range(3) for x in range(3)])
    cardinal, _ = _cardinal_tables(arrays, nodes)
    _require_close(cardinal, np.eye(9), "Q2 cardinal matrix", 1e-13)
    for order in (6, 8, 10):
        reference, weights = arrays[f"q{order}_reference"], arrays[f"q{order}_weights"]
        if np.any(reference < 0) or np.any(reference > 1) or np.any(weights <= 0):
            raise ValueError("executed positive quadrature is incompatible")
        basis, gradient = _cardinal_tables(arrays, reference)
        _require_close(arrays[f"q{order}_basis"], basis, "Q2 values", 1e-13)
        _require_close(
            arrays[f"q{order}_gradient"], gradient.swapaxes(-1, -2), "Q2 derivatives", 1e-13
        )
        # A tensor rule must integrate the supported assembly monomials exactly.
        for x in range(6):
            for y in range(6):
                if (
                    abs(
                        weights @ (reference[:, 0] ** x * reference[:, 1] ** y)
                        - 1 / ((x + 1) * (y + 1))
                    )
                    > 1e-13
                ):
                    raise ValueError("executed quadrature moments are incompatible")
    _require_close(
        arrays["trace_cardinal_matrix"],
        np.array([[1.0, 0.0], [-1.0, 2.0]]),
        "P1 moment matrix",
        1e-13,
    )
    for order in (8, 10):
        parameter, weights = arrays[f"trace_q{order}_parameter"], arrays[f"trace_q{order}_weights"]
        _require_close(
            arrays[f"trace_q{order}_basis"],
            np.column_stack((np.ones(order), parameter)) @ arrays["trace_cardinal_matrix"].T,
            "P1 values",
            1e-13,
        )
        if (
            np.any(weights <= 0)
            or np.any(parameter < 0)
            or np.any(parameter > 1)
            or abs(weights.sum() - 1) > 1e-13
        ):
            raise ValueError("executed P1 quadrature is incompatible")
    count = len(arrays["leaf_points"])
    lower, upper = arrays["leaf_points"][:, 0], arrays["leaf_points"][:, 8]
    m = 2 * int(arrays["n"])
    expected_lower = np.asarray([(i / m, j / m) for j in range(m) for i in range(m)])
    _require_close(lower, expected_lower, "whole unit-square leaf order", 1e-13)
    _require_close(upper, expected_lower + 1 / m, "whole unit-square leaf coverage", 1e-13)
    widths = (upper - lower) / 2
    if np.any(widths <= 0):
        raise ValueError("executed physical cell orientation is incompatible")
    lattice = np.asarray([(x / 2, y / 2) for y in range(3) for x in range(3)])
    _require_close(
        arrays["leaf_points"],
        lower[:, None] + lattice * (upper - lower)[:, None],
        "Cartesian leaf vertices",
        1e-13,
    )
    fine = arrays["leaf_points"][np.arange(count)[:, None, None], arrays["leaf_cells"]]
    fine_widths = fine[:, :, 2] - fine[:, :, 0]
    expected_nodes = fine[:, :, None, 0] + nodes * fine_widths[:, :, None]
    actual_nodes = arrays["leaf_nodes"][np.arange(count)[:, None, None], arrays["leaf_dofs"]]
    _require_close(actual_nodes, expected_nodes, "physical Q2 nodal injection", 1e-13)
    _require_close(
        fine_widths, np.broadcast_to(widths[:, None], fine_widths.shape), "fine cell sizes", 1e-13
    )
    # Both the original face operator and the canonical injection are retained.
    faces = arrays["flat_cell_faces"]
    if not np.array_equal(
        arrays["leaf_trace_dofs"], (2 * faces[:, :, None] + np.arange(2)).reshape(count, 8)
    ):
        raise ValueError("executed trace injection is incompatible")
    normals = arrays["flat_face_normals"]
    edges = arrays["flat_face_endpoints"]
    _require_close(np.sum(normals * normals, axis=-1), np.ones(len(normals)), "unit normals", 1e-13)
    _require_close(
        np.sum(normals * (edges[:, 1] - edges[:, 0]), axis=-1),
        np.zeros(len(normals)),
        "face normal orthogonality",
        1e-13,
    )
    outward = np.asarray([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    _require_close(
        normals[faces] * arrays["flat_cell_signs"][..., None],
        np.broadcast_to(outward, (count, 4, 2)),
        "outward orientations",
        1e-13,
    )
    corners = arrays["leaf_points"][:, [0, 2, 8, 6]]
    for side in range(4):
        expected = corners[:, [side, (side + 1) % 4]]
        actual = edges[faces[:, side]]
        if any(
            not (np.array_equal(a, b) or np.array_equal(a[::-1], b))
            for a, b in zip(actual, expected, strict=True)
        ):
            raise ValueError("executed face endpoints are incompatible")
    _require_close(
        restore(arrays, "leaf_B"),
        restore(arrays, "leaf_B_original") @ arrays["leaf_trace_transform"],
        "signed leaf coupling",
    )
    _require_close(
        restore(arrays, "leaf_local_trace"),
        np.einsum(
            "cij,cj->ci",
            arrays["leaf_trace_transform"],
            restore(arrays, "trace_recursive")[arrays["leaf_trace_dofs"]],
        ),
        "mapped recursive traces",
    )
    _require_close(
        restore(arrays, "flat_local_trace"),
        restore(arrays, "trace_flat")[arrays["flat_original_trace_dofs"]],
        "flat traces",
    )
    _require_close(
        restore(arrays, "leaf_pressure"),
        restore(arrays, "leaf_pressure_recursive"),
        "recursive coefficients",
    )
    _require_close(
        restore(arrays, "flat_pressure"), restore(arrays, "leaf_pressure_flat"), "flat coefficients"
    )
    affine = record["boundary_case"] == "affine_plus_sine"
    _require_close(
        restore(arrays, "boundary_load"),
        _physical_boundary(edges, normals, affine, 1),
        "physical boundary load",
        1e-13,
    )
    _require_close(
        restore(arrays, "parent_boundary_load"),
        _physical_boundary(arrays["outer_face_endpoints"], arrays["outer_face_normals"], affine, 2),
        "parent boundary load",
        1e-13,
    )
    reference, weights = arrays["q6_reference"], arrays["q6_weights"]
    basis, derivatives = arrays["q6_basis"], arrays["q6_gradient"]
    physical = fine[:, :, None, 0] + reference * fine_widths[:, :, None]
    measures = np.prod(fine_widths, axis=-1)[..., None] * weights
    forcing = 2 * np.pi**2 * np.sin(np.pi * physical[..., 0]) * np.sin(np.pi * physical[..., 1])
    fine_f = np.einsum("cfq,qj,cfq->cfj", measures, basis, forcing)
    gradients = derivatives[None, None] / fine_widths[:, :, None, None]
    fine_A = np.einsum("cfq,cfqia,cfqja->cfij", measures, gradients, gradients)
    expected_A, expected_f, expected_moment = (
        np.zeros((count, 25, 25)),
        np.zeros((count, 25)),
        np.zeros((count, 25)),
    )
    fine_moment = np.einsum("cfq,qj->cfj", measures, basis)
    for cell in range(count):
        for dofs, operator, load, moment in zip(
            arrays["leaf_dofs"][cell], fine_A[cell], fine_f[cell], fine_moment[cell], strict=True
        ):
            expected_A[cell][np.ix_(dofs, dofs)] += operator
            np.add.at(expected_f[cell], dofs, load)
            np.add.at(expected_moment[cell], dofs, moment)
    for prefix in ("leaf", "flat"):
        _require_close(restore(arrays, f"{prefix}_A"), expected_A, "original Q2 volume operator")
        _require_close(restore(arrays, f"{prefix}_f"), expected_f, "original sine source", 1e-13)
        _require_close(
            restore(arrays, f"{prefix}_C")[..., 0], expected_moment, "physical volume moment", 1e-13
        )
    expected_B = np.zeros((count, 25, 8))
    parameter, face_weights = arrays["trace_q8_parameter"], arrays["trace_q8_weights"]
    for side, fine_ids in enumerate(((0, 1), (1, 3), (2, 3), (0, 2))):
        reference = (
            np.column_stack((parameter, np.zeros_like(parameter)))
            if side == 0
            else np.column_stack((np.ones_like(parameter), parameter))
            if side == 1
            else np.column_stack((parameter, np.ones_like(parameter)))
            if side == 2
            else np.column_stack((np.zeros_like(parameter), parameter))
        )
        face_basis, _ = _cardinal_tables(arrays, reference)
        for local_fine in fine_ids:
            physical = fine[:, local_fine, None, 0] + reference * fine_widths[:, local_fine, None]
            edge = edges[faces[:, side]]
            tangent = edge[:, 1] - edge[:, 0]
            length_squared = np.sum(tangent * tangent, axis=-1)
            global_parameter = (
                np.sum((physical - edge[:, None, 0]) * tangent[:, None], axis=-1)
                / length_squared[:, None]
            )
            trace_basis = (
                np.stack((np.ones_like(global_parameter), global_parameter), axis=-1)
                @ arrays["trace_cardinal_matrix"].T
            )
            measure = (
                np.sqrt(length_squared)[:, None]
                / 2
                * face_weights
                * arrays["flat_cell_signs"][:, side, None]
            )
            block = np.einsum("cq,qi,cqj->cij", measure, face_basis, trace_basis)
            for cell in range(count):
                expected_B[cell][
                    np.ix_(arrays["leaf_dofs"][cell, local_fine], [2 * side, 2 * side + 1])
                ] += block[cell]
    _require_close(restore(arrays, "leaf_B"), expected_B, "original physical face integral", 1e-13)
    _require_close(
        restore(arrays, "flat_B_original"), expected_B, "flat physical face integral", 1e-13
    )
    display_basis, display_gradient = _cardinal_tables(arrays, arrays["display_reference"])
    _require_close(arrays["display_basis"], display_basis, "display Q2 values", 1e-13)
    _require_close(
        arrays["display_gradient"],
        display_gradient.swapaxes(-1, -2) / widths[:, None, None],
        "display derivatives",
        1e-13,
    )
    display_nodes = arrays["leaf_nodes"][np.arange(count)[:, None, None], arrays["display_dofs"]]
    display_origin = display_nodes[:, :, 0]
    _require_close(
        arrays["display_points"],
        display_origin + arrays["display_reference"] * widths[:, None],
        "display physical injection",
        1e-13,
    )
    for prefix in ("leaf", "flat", "parent"):
        A, Z, C, E = (restore(arrays, f"{prefix}_{name}") for name in ("A", "Z", "C", "E"))
        _require_close(A, A.swapaxes(-1, -2), "symmetric physical operator")
        _require_close(
            restore(arrays, f"{prefix}_test_B"),
            restore(arrays, f"{prefix}_B_original"),
            "symmetric trace pairing",
        )
        _require_close(restore(arrays, f"{prefix}_W"), Z, "declared symmetric retained test")
        _require_close(restore(arrays, f"{prefix}_test_C"), C, "symmetric physical moment")
        _require_close(A @ Z, np.zeros_like(Z), "declared local kernel")
        _require_close(A @ E, np.zeros_like(E), "executed retained kernel")
        _require_close(C.swapaxes(-1, -2) @ E, C.swapaxes(-1, -2) @ Z, "retained physical moments")
        if np.any(abs((C.swapaxes(-1, -2) @ Z).ravel()) <= np.finfo(float).tiny):
            raise ValueError("executed kernel moment is singular")
        _require_close(
            np.einsum("cik,ci->ck", C, restore(arrays, f"{prefix}_source")),
            np.zeros((len(C), 1)),
            "source physical moment",
        )
        _require_close(
            C.swapaxes(-1, -2) @ restore(arrays, f"{prefix}_lifts"),
            np.zeros((len(C), 1, arrays[f"{prefix}_lifts"].shape[-1])),
            "lift physical moments",
        )
        _require_close(
            replay_responses(arrays, prefix),
            restore(arrays, f"{prefix}_pressure"),
            "executed response replay",
            1e-11,
        )
    parent = restore(arrays, "parent_pressure")
    n = int(arrays["n"])
    expected_outer_dofs = (4 * arrays["outer_cell_faces"][..., None] + np.arange(4)).reshape(
        n * n, 16
    )
    if not np.array_equal(arrays["parent_original_trace_dofs"], expected_outer_dofs):
        raise ValueError("executed parent trace injection is incompatible")
    _require_close(
        restore(arrays, "parent_local_trace"),
        restore(arrays, "outer_trace")[expected_outer_dofs],
        "parent trace coordinates",
    )
    _require_close(parent[:, :28], restore(arrays, "inner_unknowns"), "inner coordinates")
    _require_close(parent[:, 28:], restore(arrays, "inner_boundary_reactions"), "parent reactions")
    for cell in range(int(arrays["n"]) ** 2):
        selected = arrays["nested_boundary_dofs"][cell]
        extension = np.zeros((28, 16))
        extension[selected, np.arange(16)] = 1
        _require_close(
            restore(arrays, "parent_A")[cell],
            np.block(
                [[-restore(arrays, "inner_A")[cell], extension], [extension.T, np.zeros((16, 16))]]
            ),
            "nested original operator",
        )
        _require_close(
            restore(arrays, "parent_B_original")[cell],
            np.vstack((np.zeros((28, 16)), -restore(arrays, "nested_trace_map")[cell])),
            "nested trace restriction",
        )
        _require_close(
            restore(arrays, "parent_f")[cell],
            np.r_[-restore(arrays, "inner_f")[cell], np.zeros(16)],
            "nested original source",
        )
        leaves = [
            (2 * (cell // n) + j) * 2 * n + 2 * (cell % n) + i for j in range(2) for i in range(2)
        ]
        inner_matrix, inner_load = np.zeros((28, 28)), np.zeros(28)
        moment, offset = np.zeros(44), 0.0
        inner_faces = {}
        for local, leaf in enumerate(leaves):
            B, test_B, A, W, E, Z, lifts, response, forcing, C = (
                restore(arrays, f"leaf_{name}")[leaf]
                for name in (
                    "B_original",
                    "test_B",
                    "A",
                    "W",
                    "E",
                    "Z",
                    "lifts",
                    "source",
                    "f",
                    "C",
                )
            )
            trace_dofs = arrays["leaf_original_trace_dofs"][leaf]
            for side in range(4):
                dofs_pair = trace_dofs[2 * side : 2 * side + 2]
                if dofs_pair[0] % 2 or dofs_pair[1] != dofs_pair[0] + 1:
                    raise ValueError("executed inner face injection is incompatible")
                fid = arrays["flat_cell_faces"][leaf, side]
                transform = arrays["leaf_trace_transform"][
                    leaf, 2 * side : 2 * side + 2, 2 * side : 2 * side + 2
                ]
                if (
                    np.count_nonzero(transform) != 2
                    or abs(transform[0, 0]) != 1
                    or abs(transform[1, 1]) != 1
                ):
                    raise ValueError("executed signed P1 injection is incompatible")
                edge = edges[fid] if transform[0, 0] == transform[1, 1] else edges[fid][::-1]
                normal = transform[0, 0] * normals[fid]
                key = int(dofs_pair[0])
                if key in inner_faces:
                    _require_close(inner_faces[key][0], edge, "duplicate inner endpoints", 1e-13)
                    _require_close(inner_faces[key][1], normal, "duplicate inner normals", 1e-13)
                inner_faces[key] = edge, normal
            coarse_dofs = arrays["inner_kernel_offsets"][cell, local : local + 2]
            if not np.array_equal(coarse_dofs, [24 + local, 25 + local]):
                raise ValueError("executed inner kernel injection is incompatible")
            dofs = np.r_[trace_dofs, coarse_dofs[0]]
            corrected = bool(arrays["leaf_corrected_retained"][leaf])
            coarse_trace = W.T @ B - (A.T @ W).T @ lifts if corrected else W.T @ B
            coarse_matrix = (A.T @ W).T @ E if corrected else np.zeros((1, 1))
            block = np.block([[test_B.T @ lifts, -test_B.T @ E], [-coarse_trace, -coarse_matrix]])
            load = np.r_[
                test_B.T @ response,
                (A.T @ W).T @ response - W.T @ forcing if corrected else -W.T @ forcing,
            ]
            inner_matrix[np.ix_(dofs, dofs)] += block
            np.add.at(inner_load, dofs, load)
            np.add.at(moment, trace_dofs, -lifts.T @ C[:, 0])
            moment[coarse_dofs[0]] = (E.T @ C[:, 0]).item()
            offset += (C[:, 0] @ response).item()
            _require_close(
                restore(arrays, "leaf_local_trace")[leaf],
                restore(arrays, "inner_unknowns")[cell, trace_dofs],
                "inner leaf traces",
            )
            _require_close(
                restore(arrays, "leaf_coarse")[leaf],
                restore(arrays, "inner_unknowns")[cell, coarse_dofs[0] : coarse_dofs[1]],
                "inner retained coordinates",
            )
        _require_close(
            restore(arrays, "inner_A")[cell], inner_matrix, "inner executed reduced operator"
        )
        _require_close(
            restore(arrays, "inner_f")[cell], inner_load, "inner executed reduced source"
        )
        _require_close(
            restore(arrays, "parent_physical_moment")[cell], moment, "nested moment lift", 1e-13
        )
        _require_close(
            restore(arrays, "parent_physical_moment_offset")[cell],
            np.asarray(offset),
            "nested source moment offset",
            1e-13,
        )
        if len(np.unique(selected)) != 16 or any(
            selected[2 * k] % 2 or selected[2 * k + 1] != selected[2 * k] + 1 for k in range(8)
        ):
            raise ValueError("executed selected inner boundary injection is incompatible")
        mapping = np.zeros((16, 16))
        for k in range(8):
            edge, normal = inner_faces[int(selected[2 * k])]
            matches = []
            for side, fid in enumerate(arrays["outer_cell_faces"][cell]):
                outer_edge = arrays["outer_face_endpoints"][fid]
                tangent = outer_edge[1] - outer_edge[0]
                parameter = (edge - outer_edge[0]) @ tangent / (tangent @ tangent)
                if (
                    np.max(abs(edge - outer_edge[0] - parameter[:, None] * tangent)) < 1e-13
                    and parameter.min() >= 0
                    and parameter.max() <= 1
                ):
                    matches.append((side, fid, parameter))
            if len(matches) != 1:
                raise ValueError("executed nested boundary geometry is incompatible")
            side, fid, parameter = matches[0]
            segment = int(np.floor(parameter.mean() * 2))
            sign = normal @ arrays["outer_face_normals"][fid]
            column = 4 * side + 2 * segment
            mapping[2 * k : 2 * k + 2, column : column + 2] = np.diag(
                [sign, sign * 2 * (parameter[1] - parameter[0])]
            )
        _require_close(
            restore(arrays, "nested_trace_map")[cell],
            mapping,
            "geometric nested P1 restriction",
            1e-13,
        )
        _require_close(
            restore(arrays, "inner_unknowns")[cell, selected],
            mapping @ restore(arrays, "parent_local_trace")[cell],
            "prescribed inner boundary trace",
        )
    _require_close(
        restore(arrays, "parent_C")[..., 0],
        restore(arrays, "parent_physical_moment"),
        "nested physical moment",
        1e-13,
    )
    checks = original_checks(arrays)
    if max(checks.values()) > 1e-10:
        raise ValueError("executed original physical equations failed")
    recorded_checks = record.get("original_physical_checks", {})
    if set(checks) != set(recorded_checks):
        raise ValueError("executed original-check record is incompatible")
    _require_close(
        np.asarray(list(recorded_checks.values())),
        np.asarray([checks[key] for key in recorded_checks]),
        "original-check record",
        1e-13,
    )

    def exact(points: np.ndarray) -> np.ndarray:
        """Check the declared physical analytical pressure independently of driver callbacks."""
        values = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
        return values + 1 + points[:, 0] - points[:, 1] if affine else values

    def gradient(points: np.ndarray) -> np.ndarray:
        """Independently differentiate the declared analytical pressure."""
        x, y = np.pi * points.T
        values = np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))
        return values + np.array([1.0, -1.0]) if affine else values

    norms = {
        f"quadrature_{order}": field_norms(arrays, exact, gradient, order) for order in (8, 10)
    }
    for key, values in norms.items():
        recorded = record.get("norms", {}).get(key, {})
        if set(values) != set(recorded):
            raise ValueError("executed physical-norm record is incompatible")
        _require_close(
            np.asarray(list(recorded.values())),
            np.asarray([values[name] for name in recorded]),
            "physical-norm record",
            1e-13,
        )
        if (
            max(
                values["pressure_relative_flat_difference"],
                values["raw_flux_relative_flat_difference"],
            )
            > 1e-9
        ):
            raise ValueError("executed physical field agreement failed")
    before, after = norms.values()
    if (
        max(
            abs(before[key] - after[key]) / max(abs(after[key]), np.finfo(float).tiny)
            for key in ("pressure_l2", "flux_l2")
        )
        > 1e-9
    ):
        raise ValueError("executed physical norm quadrature failed")


def write_archive(
    path: Path, arrays: dict[str, np.ndarray], record: dict[str, Any]
) -> dict[str, Any]:
    """Write a fresh archive with per-array digests and an immutable complete-file digest."""
    metadata = path.with_suffix(".json")
    if path.exists() or metadata.exists():
        raise ValueError("a fresh field archive and record are required")
    if any(a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("finite portable real/integer archive arrays required")
    identifier = str(UUID(record["acquisition_uuid"]))
    arrays = dict(arrays)
    arrays.update(
        acquisition_uuid=np.frombuffer(identifier.encode("ascii"), dtype=np.uint8).copy(),
        archive_schema_version=np.asarray(1, dtype=np.int64),
        coefficient_precision_bits=np.asarray(53, dtype=np.int64),
    )
    result = dict(
        record,
        field_archive_schema=SCHEMA,
        coefficient_precision_bits=53,
        arrays_sha256={k: array_digest(v) for k, v in arrays.items()},
    )
    validate_execution(arrays, result)
    checkpoint = checkpoint_field(path, arrays, result)
    result.update(archive=checkpoint["archive"], archive_sha256=checkpoint["archive_sha256"])
    write_progress(metadata, result)
    return result


def read_archive(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reject changed basis/operator/coefficient bytes before replaying any field."""
    record = json.loads(path.with_suffix(".json").read_text())
    if (
        record["archive"] != path.name
        or hashlib.sha256(path.read_bytes()).hexdigest() != record["archive_sha256"]
    ):
        raise ValueError("executed field archive digest changed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if set(arrays) != set(record["arrays_sha256"]) or any(
        array_digest(v) != record["arrays_sha256"][name] for name, v in arrays.items()
    ):
        raise ValueError("executed array/basis digest changed")
    validate_execution(arrays, record)
    return record, arrays
