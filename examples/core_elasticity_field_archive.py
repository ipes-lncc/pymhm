"""Persist executed weak-symmetry elasticity coordinates, bases and physical rows.

The skeleton multiplier is negative Cauchy traction in the canonical face
normal, rather than an H(div) Darcy flux. Stress uses row-wise contravariant
Piola mapping; displacement and independent weak rotation are broken fields.
Only current serial BDM2/P1/P1 and RT1/Q1/total-P1 producers are supported.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy import sparse

from examples.archive_precision import precision_fields, restore_precision
from examples.local_response_cache import array_identity
from examples.transport_checkpoints import checkpoint_field, write_progress
from pymhm.core.system import HybridSystem
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.tensor_rt import tensor_rt_basis, tensor_rt_dofs
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.io.provenance import file_digest
from pymhm.linalg.linear import accurate_residual

SCHEMA = "pymhm-core-elasticity-executed-field-v1"


@dataclass
class ProductionObservation:
    """Literal system, applied boundary moments, mean rows and executing BDM matrix."""

    system: HybridSystem | None = None
    boundary_load: np.ndarray | None = None
    applied_boundary: np.ndarray | None = None
    fixed: dict[int, float] = field(default_factory=dict)
    mean_weights: list[tuple[np.ndarray, ...]] = field(default_factory=list)
    mean_values: list[float] = field(default_factory=list)
    constraints: list[tuple[np.ndarray, float]] = field(default_factory=list)
    bdm_coefficients: np.ndarray | None = None
    compliance: np.ndarray | None = None
    assembly_tables: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    assembly_scalar: dict[str, np.ndarray] = field(default_factory=dict)
    assembly_rotation: dict[str, np.ndarray] = field(default_factory=dict)


@contextmanager
def observe_system() -> Iterator[ProductionObservation]:
    """Observe one unchanged serial solve in an isolated process; restore every owner.

    Delegation preserves each original argument and return value. This process
    scoped observer must not overlap another production solve in that process.
    Assembly tables and BDM coefficient bytes come from the original calls.
    """
    import examples.formulations.weak_stress as declared
    import pymhm._legacy.models.elasticity.stress as triangle
    import pymhm._legacy.models.elasticity.stress_tensor as rectangle
    import pymhm.fem.hdiv.bdm_family as bdm
    import pymhm.fem.vector.stress as triangle_forms
    import pymhm.fem.vector.stress_tensor as rectangle_forms
    from pymhm.core.multiscale import MultiscaleSystem

    observed = ProductionObservation()
    original_solve, original_assemble = HybridSystem.solve, HybridSystem._assemble_global
    original_multiscale_solve = MultiscaleSystem.solve
    original_mean, original_boundary = HybridSystem.mean_constraint, triangle.boundary_data
    original_dual, original_basis = bdm._full_dual, BDMFamily.basis
    original_rt = rectangle_forms.tensor_rt_basis
    original_compliance = triangle_forms.compliance_products
    original_scalar, original_rotation = (
        triangle_forms.reference_basis,
        rectangle_forms.complete_rotation_basis,
    )

    def scalar(degree: int, points: np.ndarray) -> Any:
        """Capture the cardinal displacement table actually consumed by triangular assembly."""
        result = original_scalar(degree, points)
        if degree != 1:
            raise ValueError("The selected producer requires cardinal P1")
        observed.assembly_scalar[array_identity(points)] = result[0].copy()
        return result

    def rotation(degree: int, points: np.ndarray) -> Any:
        """Capture the total-degree rotation table used by rectangular assembly."""
        result = original_rotation(degree, points)
        observed.assembly_rotation[array_identity(points)] = result.copy()
        return result

    def compliance(material: Any, *args: Any, **kwargs: Any) -> Any:
        """Capture the constant material actually accepted by the compliance owner."""
        result = original_compliance(material, *args, **kwargs)
        value = np.asarray(material)
        if value.shape != (4, 4):
            raise ValueError("The selected producer requires constant Cartesian compliance")
        if observed.compliance is not None and not np.array_equal(observed.compliance, value):
            raise ValueError("The executed material changed during acquisition")
        observed.compliance = value.copy()
        return result

    def assemble(system: HybridSystem, responses: Any, metadata: Any, boundary_load: Any) -> None:
        """Capture the actual signed boundary load supplied to global assembly."""
        original_assemble(system, responses, metadata, boundary_load)
        if boundary_load is not None:
            if observed.applied_boundary is not None:
                raise ValueError("One explicit production boundary vector is required")
            observed.applied_boundary = np.asarray(boundary_load).copy()

    def boundary(*args: Any, **kwargs: Any) -> Any:
        """Capture actual Dirichlet moments and prescribed negative-traction coordinates."""
        result = original_boundary(*args, **kwargs)
        if observed.boundary_load is not None:
            raise ValueError("One production boundary call is required")
        observed.boundary_load, observed.fixed = result[0].copy(), dict(result[1])
        return result

    def mean(system: HybridSystem, local_weights: Any, value: float = 0.0) -> Any:
        """Capture physical moment weights and their actual reduced row."""
        result = original_mean(system, local_weights, value)
        observed.mean_weights.append(tuple(np.asarray(w).copy() for w in local_weights))
        observed.mean_values.append(value)
        return result

    def solve(system: HybridSystem, *args: Any, **kwargs: Any) -> Any:
        """Retain the actual solved system and gauge rows, without changing its solve."""
        result = (
            original_multiscale_solve if isinstance(system, MultiscaleSystem) else original_solve
        )(system, *args, **kwargs)
        if isinstance(system, MultiscaleSystem):
            observed.applied_boundary = -system.global_load[: system.trace_size].copy()
        if observed.system is not None:
            raise ValueError("One production system is required")
        observed.system = system
        observed.constraints = [
            (np.asarray(row).copy(), value) for row, value in kwargs.get("constraints", ())
        ]
        return result

    def dual(degree: int) -> np.ndarray:
        """Capture the actual BDM moment matrix returned during evaluation."""
        result = original_dual(degree)
        if degree != 2:
            raise ValueError("The current producer contract requires BDM2")
        if observed.bdm_coefficients is not None and not np.array_equal(
            observed.bdm_coefficients, result
        ):
            raise ValueError("Executing BDM coordinates changed inside one acquisition")
        observed.bdm_coefficients = result.copy()
        return result

    def basis(family: BDMFamily, mesh: Any, points: np.ndarray) -> Any:
        """Capture the first actual fine-mesh assembly evaluation."""
        result = original_basis(family, mesh, points)
        if id(mesh) not in observed.assembly_tables:
            observed.assembly_tables[id(mesh)] = (points.copy(), result[0].copy(), result[1].copy())
        return result

    def rt(mesh: Any, degree: int, enrichment: int, points: np.ndarray) -> Any:
        """Capture the actual RT operator evaluation, before rigid-mode projection calls."""
        result = original_rt(mesh, degree, enrichment, points)
        if id(mesh) not in observed.assembly_tables:
            observed.assembly_tables[id(mesh)] = (points.copy(), result[0].copy(), result[1].copy())
        observed.assembly_scalar[array_identity(points)] = result[2].copy()
        return result

    with (
        patch.object(HybridSystem, "_assemble_global", assemble),
        patch.object(HybridSystem, "mean_constraint", mean),
        patch.object(HybridSystem, "solve", solve),
        patch.object(MultiscaleSystem, "solve", solve),
        patch.object(declared, "boundary_data", boundary),
        patch.object(triangle, "boundary_data", boundary),
        patch.object(rectangle, "boundary_data", boundary),
        patch.object(bdm, "_full_dual", dual),
        patch.object(BDMFamily, "basis", basis),
        patch.object(rectangle_forms, "tensor_rt_basis", rt),
        patch.object(triangle_forms, "compliance_products", compliance),
        patch.object(rectangle_forms, "compliance_products", compliance),
        patch.object(triangle_forms, "reference_basis", scalar),
        patch.object(rectangle_forms, "complete_rotation_basis", rotation),
    ):
        yield observed


def restore(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Restore finite high/correction/tail coordinates without narrowing their mantissa."""
    return restore_precision(arrays[name], arrays[name + "_correction"], arrays[name + "_tail"])


def _put_csr(arrays: dict[str, np.ndarray], name: str, matrix: Any) -> None:
    """Store the literal represented sparse matrix without symmetrizing entries."""
    matrix = sparse.csr_matrix(matrix)
    arrays.update(
        {
            name + "_data": matrix.data.copy(),
            name + "_indices": matrix.indices.copy(),
            name + "_indptr": matrix.indptr.copy(),
            name + "_shape": np.asarray(matrix.shape, dtype=np.int64),
        }
    )


def _csr(arrays: Mapping[str, np.ndarray], name: str) -> sparse.csr_matrix:
    """Restore a checked literal sparse row representation."""
    shape = arrays[name + "_shape"]
    indices, indptr, data = (arrays[name + suffix] for suffix in ("_indices", "_indptr", "_data"))
    if (
        shape.shape != (2,)
        or shape.dtype.kind not in "iu"
        or np.any(shape <= 0)
        or indices.dtype.kind not in "iu"
        or indptr.dtype.kind not in "iu"
        or indptr.shape != (int(shape[0]) + 1,)
        or indptr[0] != 0
        or indptr[-1] != len(data)
        or indices.shape != data.shape
        or np.any(np.diff(indptr) < 0)
        or np.any(indices < 0)
        or np.any(indices >= shape[1])
    ):
        raise ValueError("Archived sparse layout is inconsistent")
    return sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))


def _tables(solution: Any, mesh: Any, points: np.ndarray) -> tuple[np.ndarray, ...]:
    """Execute the shared producer's basis operations once for this acquisition's tables."""
    if hasattr(solution, "family"):
        values, divergence = solution.family.basis(mesh, points)
        displacement = rotation = reference_basis(1, points)[0]
    else:
        from pymhm.fem.vector.stress_tensor import complete_rotation_basis as _rotation_basis

        values, divergence, displacement = tensor_rt_basis(mesh, 1, 0, points)
        rotation = _rotation_basis(1, points)
    return values, divergence, displacement, rotation


def field_arrays(
    solution: Any,
    observed: ProductionObservation,
    *,
    assembly_order: int = 8,
    norm_orders: tuple[int, ...] = (9, 10),
) -> dict[str, np.ndarray]:
    """Capture executing A/B/f/Z/C/E, physical fields, moments, maps and norm tables.

    Norm and face tables are generated by the shared producer inside the fresh
    acquisition; replay consumes their literal bytes. The original assembly
    tables and BDM C are observed from production, never retrospectively fitted.
    """
    system = observed.system
    if (
        system is None
        or observed.boundary_load is None
        or observed.applied_boundary is None
        or observed.compliance is None
        or type(assembly_order) is not int
        or assembly_order < 5
        or not norm_orders
        or len(set(norm_orders)) != len(norm_orders)
        or any(type(order) is not int or order < 5 for order in norm_orders)
        or len(observed.constraints) != len(observed.mean_weights)
        or len(system.responses) != len(solution.local_meshes)
    ):
        raise ValueError("An actual serial production solve and sufficient quadratures required")
    triangle = hasattr(solution, "family")
    if (triangle and (solution.family.degree, solution.family.enrichment) != (2, 0)) or (
        not triangle and (solution.degree, solution.enrichment) != (1, 0)
    ):
        raise ValueError("Only executing BDM2 or RT1 without enrichment is supported")
    macro, skeleton = solution.skeleton.mesh, solution.skeleton
    arrays = {
        "kind": np.asarray(0 if triangle else 1),
        "local_count": np.asarray(len(solution.local_meshes)),
        "assembly_order": np.asarray(assembly_order),
        "norm_orders": np.asarray(norm_orders),
        "macro_points": macro.points.copy(),
        "macro_face_endpoints": macro.points[macro.faces].copy(),
        "macro_normals": macro.normals.copy(),
        "macro_cell_face_offsets": np.r_[0, np.cumsum([len(f) for f in macro.cell_faces])],
        "macro_cell_faces": np.concatenate(macro.cell_faces),
        "macro_signs": np.concatenate(macro.signs),
        "macro_boundary_faces": macro.boundary_faces.copy(),
        "trace_offsets": skeleton.offsets.copy(),
        "kernel_offsets": system.kernel_offsets.copy(),
        "trace_size": np.asarray(system.trace_size),
        "gauge_count": np.asarray(len(observed.constraints)),
        "coefficient_nmant": np.asarray(np.finfo(solution.hybrid.trace.dtype).nmant),
        "cartesian_compliance": observed.compliance.copy(),
        "stress_normal_degree": np.asarray(2 if triangle else 1),
        "stress_enrichment": np.asarray(0),
        "rotation_total_degree": np.asarray(1),
    }
    if triangle:
        if observed.bdm_coefficients is None:
            raise ValueError("The executing BDM moment matrix was not observed")
        arrays["bdm_coefficients"] = observed.bdm_coefficients.copy()
    _put_csr(arrays, "global_matrix", system.matrix)
    for name, value in (
        ("global_trace", solution.hybrid.trace),
        ("global_rhs", system.rhs),
        ("physical_dirichlet_moments", observed.boundary_load),
        ("applied_boundary_load", observed.applied_boundary),
        ("gauge_multipliers", solution.hybrid.gauge_multipliers),
        ("gauge_rows", np.array([r for r, _ in observed.constraints]).reshape(-1, len(system.rhs))),
        ("gauge_targets", np.asarray([v for _, v in observed.constraints])),
        ("physical_mean_values", np.asarray(observed.mean_values)),
    ):
        arrays.update(precision_fields(name, value))
    arrays["fixed_trace_indices"] = np.asarray(sorted(observed.fixed), dtype=np.int64)
    arrays["fixed_trace_values"] = np.asarray([observed.fixed[i] for i in sorted(observed.fixed)])
    parameter, face_weights = leggauss(9)
    parameter, face_weights = (parameter + 1) / 2, face_weights / 2
    arrays["face_parameter"], arrays["face_weights"] = parameter, face_weights
    arrays["trace_degrees"] = np.asarray([max(face.degrees) for face in skeleton.faces])
    for face, space in enumerate(skeleton.faces):
        arrays[f"trace_breaks_{face}"] = np.asarray(space.breaks)
        arrays[f"trace_degrees_{face}"] = np.asarray(space.degrees)
        arrays[f"trace_values_{face}"] = space.evaluate(parameter)
    for order in norm_orders:
        points, weights = (triangle_quadrature if triangle else quadrilateral_quadrature)(order)
        arrays[f"q{order}_reference_points"], arrays[f"q{order}_weights"] = points, weights
    for cell, (mesh, response, internal) in enumerate(
        zip(solution.local_meshes, system.responses, solution.hybrid.fields, strict=True)
    ):
        problem = response.problem
        ns, nu, nr = (
            solution.stress[cell].size,
            solution.displacement[cell].size,
            solution.rotation[cell].size,
        )
        dofs = solution.family.dofs(mesh) if triangle else tensor_rt_dofs(mesh, 1, 0)
        vertices = mesh.points[mesh.cells]
        jacobian = (
            (vertices[:, 1:3] - vertices[:, :1]).swapaxes(1, 2)
            if triangle
            else (np.broadcast_to(np.diag(mesh.spacing), (len(mesh.cells), 2, 2)).copy())
        )
        normals = np.stack(
            (
                np.diff(vertices[:, np.r_[np.arange(vertices.shape[1]), 0]], axis=1)[..., 1],
                -np.diff(vertices[:, np.r_[np.arange(vertices.shape[1]), 0]], axis=1)[..., 0],
            ),
            axis=-1,
        )
        owned = {
            "points": mesh.points,
            "cells": mesh.cells,
            "faces": mesh.faces,
            "cell_faces": mesh.cell_faces,
            "signs": mesh.signs,
            "areas": mesh.areas,
            "jacobian": jacobian,
            "stress_dofs": dofs,
            "trace_dofs": problem.trace_dofs,
            "block_sizes": np.asarray([ns, nu, nr, len(internal) - ns - nu - nr]),
            "physical_outward_measure_normals": normals,
            "internal_nmant": np.asarray(np.finfo(internal.dtype).nmant),
        }
        arrays.update({f"{name}_{cell}": np.asarray(value).copy() for name, value in owned.items()})
        _put_csr(arrays, f"original_a_{cell}", problem.matrix)
        for name, value in (
            ("original_b", problem.coupling),
            ("test_coupling", problem.test_coupling),
            ("original_f", problem.load),
            ("kernel_z", problem.kernel),
            ("coarse_basis_z", problem.coarse_basis),
            ("constraints_c", problem.constraints),
            ("test_basis", problem.test_basis),
            ("test_constraints", problem.test_constraints),
            ("retained_basis_e", response.retained_basis),
            ("source_lift", response.source),
            ("trace_lifts", response.lifts),
            ("internal_field", internal),
            ("stress", solution.stress[cell]),
            ("displacement", solution.displacement[cell]),
            ("rotation", solution.rotation[cell]),
            ("coarse", solution.hybrid.coarse[cell]),
        ):
            arrays.update(precision_fields(f"{name}_{cell}", value))
        for gauge, gauge_weights in enumerate(observed.mean_weights):
            arrays.update(
                precision_fields(f"physical_mean_weights_{gauge}_{cell}", gauge_weights[cell])
            )
        actual = observed.assembly_tables.get(id(mesh))
        expected_points = (triangle_quadrature if triangle else quadrilateral_quadrature)(
            assembly_order
        )[0]
        if actual is None or not np.array_equal(actual[0], expected_points):
            raise ValueError("Executing operator quadrature was not captured")
        scalar = observed.assembly_scalar.get(array_identity(expected_points))
        rotation = (
            scalar if triangle else observed.assembly_rotation.get(array_identity(expected_points))
        )
        if scalar is None or rotation is None:
            raise ValueError("Executing displacement/rotation quadrature tables were not captured")
        arrays[f"assembly_displacement_basis_{cell}"] = scalar.copy()
        arrays[f"assembly_rotation_basis_{cell}"] = rotation.copy()
        for name, value in zip(("points", "stress_basis", "divergence_basis"), actual, strict=True):
            arrays[f"assembly_{name}_{cell}"] = value
        for order in norm_orders:
            points = arrays[f"q{order}_reference_points"]
            values, div, scalar, rot = _tables(solution, mesh, points)
            physical = (
                np.einsum("qi,tij->tqj", points, vertices)
                if triangle
                else (vertices[:, :1] + points * mesh.spacing)
            )
            for name, value in (
                ("stress_basis", values),
                ("divergence_basis", div),
                ("displacement_basis", scalar),
                ("rotation_basis", rot),
                ("physical_points", physical),
            ):
                arrays[f"q{order}_{name}_{cell}"] = value
        refvertices = (
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            if triangle
            else (np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        )
        for side in range(len(refvertices)):
            edgepoints = (1 - parameter[:, None]) * refvertices[side] + parameter[:, None] * (
                refvertices[(side + 1) % len(refvertices)]
            )
            reference = (
                np.column_stack((1 - edgepoints.sum(axis=1), edgepoints))
                if triangle
                else edgepoints
            )
            arrays[f"face_reference_points_{side}"] = reference
            arrays[f"face_stress_basis_{side}_{cell}"] = _tables(solution, mesh, reference)[0]
    return arrays


def replay(
    arrays: Mapping[str, np.ndarray], cell: int, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate displacement, full stress, div(stress) and rotation from literal tables.

    No element, numerical basis, Piola map or orientation is regenerated. The
    saved stress table already includes its executed physical orientation.
    """
    if cell not in range(int(arrays["local_count"])) or order not in arrays["norm_orders"]:
        raise ValueError("An executed macro and norm quadrature are required")
    coefficients = restore(arrays, f"stress_{cell}")[arrays[f"stress_dofs_{cell}"]]
    scalar = arrays[f"q{order}_displacement_basis_{cell}"]
    rotation = arrays[f"q{order}_rotation_basis_{cell}"]
    return (
        np.einsum("qi,tia->tqa", scalar, restore(arrays, f"displacement_{cell}")),
        np.einsum("tqib,tia->tqab", arrays[f"q{order}_stress_basis_{cell}"], coefficients),
        np.einsum("tqi,tia->tqa", arrays[f"q{order}_divergence_basis_{cell}"], coefficients),
        np.einsum("qi,ti->tq", rotation, restore(arrays, f"rotation_{cell}")),
    )


def _rt_polynomial_audit(
    arrays: Mapping[str, np.ndarray], cell: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Check RT1's polynomial identity against its literal executing assembly table.

    The least-squares polynomial coefficients are diagnostics only: no field
    coordinates or replay basis are produced from them. RT1's component spaces
    are Q(2,1) and Q(1,2); independently differentiating these polynomials checks
    literal divergence and connects norm tables to actual assembly arithmetic.
    """
    assembly = arrays[f"assembly_points_{cell}"]
    values = arrays[f"assembly_stress_basis_{cell}"]
    jacobian = arrays[f"jacobian_{cell}"]
    tables = np.zeros((len(values), len(points), 12, 2))
    divergence = np.zeros((len(values), len(points), 12))
    for axis in range(2):
        powers = [
            (a, b) for b in range(2 if axis == 0 else 3) for a in range(3 if axis == 0 else 2)
        ]
        monomials = np.column_stack([assembly[:, 0] ** a * assembly[:, 1] ** b for a, b in powers])
        coefficients = np.linalg.lstsq(
            monomials, np.moveaxis(values[..., axis], 0, 1).reshape(len(assembly), -1), rcond=None
        )[0]
        represented = monomials @ coefficients
        scale = max(float(np.max(abs(values[..., axis]))), np.finfo(float).tiny)
        if (
            np.max(
                abs(represented - np.moveaxis(values[..., axis], 0, 1).reshape(len(assembly), -1))
            )
            > 1e-12 * scale
        ):
            raise ValueError("Executing RT1 assembly table leaves its tensor polynomial space")
        test = np.column_stack([points[:, 0] ** a * points[:, 1] ** b for a, b in powers])
        derivative = np.column_stack(
            [
                (a * points[:, 0] ** (a - 1) * points[:, 1] ** b if a else np.zeros(len(points)))
                if axis == 0
                else (
                    b * points[:, 0] ** a * points[:, 1] ** (b - 1) if b else np.zeros(len(points))
                )
                for a, b in powers
            ]
        )
        tables[..., axis] = np.moveaxis(
            (test @ coefficients).reshape(len(points), len(values), 12), 0, 1
        )
        differentiated = np.moveaxis(
            (derivative @ coefficients).reshape(len(points), len(values), 12), 0, 1
        )
        divergence += differentiated / jacobian[:, axis, axis, None, None]
    return tables, divergence


def basis_checks(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Audit finite polynomial/normal-moment identities using the archived representation.

    Independent moment and divergence-theorem checks use literal face/volume
    tables. BDM tables additionally contract the saved numerical C with raw
    shared monomials; no new BDM inverse or fitted field coordinates are used.
    These finite identities do not establish a uniform inf-sup constant.
    """
    from pymhm.fem.hdiv.bdm_family import _polynomials

    triangle = int(arrays["kind"]) == 0
    degree, sides = (2, 3) if triangle else (1, 4)
    parameter, weights = arrays["face_parameter"], arrays["face_weights"]
    maximum = maximum_divergence = maximum_table = 0.0
    compliance = arrays["cartesian_compliance"]
    if (
        compliance.shape != (4, 4)
        or not np.allclose(compliance, compliance.T, rtol=1e-13, atol=1e-14)
        or np.min(np.linalg.eigvalsh(compliance)) <= 0
    ):
        raise ValueError("Archived Cartesian compliance is not positive self-adjoint")
    for cell in range(int(arrays["local_count"])):
        vertices = arrays[f"points_{cell}"][arrays[f"cells_{cell}"]]
        tangent = np.roll(vertices, -1, axis=1) - vertices
        normals = np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
        if not np.array_equal(normals, arrays[f"physical_outward_measure_normals_{cell}"]):
            raise ValueError("Archived outward face orientation differs from actual vertices")
        signs = arrays[f"signs_{cell}"]
        if signs.shape != vertices.shape[:2] or not np.all(np.isin(signs, (-1, 1))):
            raise ValueError("Invalid cell-normal orientation")
        jacobian = arrays[f"jacobian_{cell}"]
        geometric_jacobian = (
            (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
            if triangle
            else np.stack(
                (vertices[:, 1] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]), axis=-1
            )
        )
        if not np.array_equal(jacobian, geometric_jacobian):
            raise ValueError("Archived Piola map differs from physical cell vertices")
        determinant = np.linalg.det(jacobian)
        area = arrays[f"areas_{cell}"]
        if np.any(determinant <= 0) or not np.allclose(
            determinant, area * (2 if triangle else 1), rtol=1e-13, atol=1e-15
        ):
            raise ValueError("Archived Piola determinant and physical measure differ")
        edge_integral = np.zeros((len(vertices), 12))
        if triangle:
            coefficients = arrays["bdm_coefficients"]
            if coefficients.shape != (12, 12):
                raise ValueError("Executing BDM2 matrix has incompatible shape")
            orientation = np.ones((len(vertices), 12))
            orientation[:, :9] = (signs[:, :, None] ** np.arange(1, 4)).reshape(-1, 9)
        for side in range(sides):
            values = arrays[f"face_stress_basis_{side}_{cell}"]
            if triangle:
                raw = _polynomials(arrays[f"face_reference_points_{side}"][:, 1:], 2)[0]
                reference = np.einsum("qja,ji->qia", raw, coefficients)
                expected_values = np.einsum(
                    "tab,qib,ti,t->tqia", jacobian, reference, orientation, 1 / determinant
                )
                maximum_table = max(maximum_table, float(np.max(abs(values - expected_values))))
            else:
                expected_values, _ = _rt_polynomial_audit(
                    arrays, cell, arrays[f"face_reference_points_{side}"]
                )
                if np.max(abs(values - expected_values)) > 1e-11 * max(
                    float(np.max(abs(values))), np.finfo(float).tiny
                ):
                    raise ValueError("RT1 face tables differ from executing polynomial identity")
            normal_value = np.einsum("tqia,ta->tqi", values, normals[:, side])
            canonical = np.where(signs[:, side, None] > 0, parameter, 1 - parameter)
            tests = legvander(2 * canonical - 1, degree)
            moments = np.einsum("q,tqj,tqi,t->tji", weights, tests, normal_value, signs[:, side])
            expected = np.zeros_like(moments)
            expected[:, :, side * (degree + 1) : (side + 1) * (degree + 1)] = np.eye(degree + 1)
            maximum = max(maximum, float(np.max(abs(moments - expected))))
            edge_integral += np.einsum("q,tqi->ti", weights, normal_value)
        for order in arrays["norm_orders"]:
            order = int(order)
            points, qweights = arrays[f"q{order}_reference_points"], arrays[f"q{order}_weights"]
            scalar, rotation = (
                arrays[f"q{order}_displacement_basis_{cell}"],
                arrays[f"q{order}_rotation_basis_{cell}"],
            )
            if triangle:
                if not np.allclose(scalar, points, rtol=1e-13, atol=1e-14) or not np.array_equal(
                    scalar, rotation
                ):
                    raise ValueError(
                        "Triangle displacement and rotation require actual cardinal P1"
                    )
                raw, raw_div = _polynomials(points[:, 1:], 2)
                coefficients = arrays["bdm_coefficients"]
                if coefficients.shape != (12, 12):
                    raise ValueError("Executing BDM2 matrix has incompatible shape")
                reference = np.einsum("qja,ji->qia", raw, coefficients)
                orientation = np.ones((len(vertices), 12))
                orientation[:, :9] = (signs[:, :, None] ** np.arange(1, 4)).reshape(-1, 9)
                physical = np.einsum(
                    "tab,qib,ti,t->tqia", jacobian, reference, orientation, 1 / determinant
                )
                divergence = np.einsum(
                    "qi,ti,t->tqi", raw_div @ coefficients, orientation, 1 / determinant
                )
                maximum_table = max(
                    maximum_table,
                    float(np.max(abs(physical - arrays[f"q{order}_stress_basis_{cell}"]))),
                    float(np.max(abs(divergence - arrays[f"q{order}_divergence_basis_{cell}"]))),
                )
            else:
                x, y = 2 * points.T - 1
                expected_scalar = np.column_stack((np.ones(len(x)), x, y, x * y))
                expected_rotation = expected_scalar[:, :3]
                if not np.allclose(
                    scalar, expected_scalar, rtol=1e-13, atol=1e-14
                ) or not np.allclose(rotation, expected_rotation, rtol=1e-13, atol=1e-14):
                    raise ValueError("Rectangle requires Q1 displacement and total P1 rotation")
                expected_values, expected_div = _rt_polynomial_audit(arrays, cell, points)
                for expected, actual in (
                    (expected_values, arrays[f"q{order}_stress_basis_{cell}"]),
                    (expected_div, arrays[f"q{order}_divergence_basis_{cell}"]),
                ):
                    if np.max(abs(expected - actual)) > 1e-11 * max(
                        float(np.max(abs(actual))), np.finfo(float).tiny
                    ):
                        raise ValueError(
                            "RT1 norm tables differ from executing polynomial/Piola/divergence"
                        )
            divergence_integral = np.einsum(
                "q,tqi,t->ti", qweights, arrays[f"q{order}_divergence_basis_{cell}"], area
            )
            maximum_divergence = max(
                maximum_divergence, float(np.max(abs(divergence_integral - edge_integral)))
            )
            expected_physical = (
                np.einsum("qi,tij->tqj", points, vertices)
                if triangle
                else (vertices[:, :1] + np.einsum("tab,qb->tqa", jacobian, points))
            )
            if not np.allclose(
                expected_physical,
                arrays[f"q{order}_physical_points_{cell}"],
                rtol=1e-13,
                atol=1e-14,
            ):
                raise ValueError("Norm quadrature has inconsistent physical coordinates")
    if max(maximum, maximum_divergence) > 1e-10 or maximum_table > 1e-10:
        raise ValueError("Archived basis violates its canonical moment/Piola/divergence identities")
    return {
        "maximum_canonical_normal_moment_defect": maximum,
        "maximum_divergence_theorem_defect": maximum_divergence,
        "maximum_executed_bdm_table_defect": maximum_table,
    }


def _validate_envelope(arrays: Mapping[str, np.ndarray]) -> None:
    """Reject nonfinite/complex arrays and incompatible degree, precision or map dimensions."""
    for value in arrays.values():
        if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise ValueError("Every archived array must be finite and real")
    for name in (
        "kind",
        "local_count",
        "assembly_order",
        "trace_size",
        "gauge_count",
        "coefficient_nmant",
        "stress_normal_degree",
        "stress_enrichment",
        "rotation_total_degree",
    ):
        if arrays[name].shape != () or arrays[name].dtype.kind not in "iu":
            raise ValueError("Executed scalar orders and dimensions require integer scalars")
    triangle = int(arrays["kind"]) == 0
    if (
        int(arrays["kind"]) not in (0, 1)
        or int(arrays["local_count"]) <= 0
        or int(arrays["trace_size"]) <= 0
        or int(arrays["gauge_count"]) not in range(4)
        or int(arrays["stress_normal_degree"]) != (2 if triangle else 1)
        or int(arrays["stress_enrichment"]) != 0
        or int(arrays["rotation_total_degree"]) != 1
        or int(arrays["coefficient_nmant"]) > np.finfo(np.longdouble).nmant
    ):
        raise ValueError("Unsupported executed elasticity family, dimensions or precision")
    orders = arrays["norm_orders"]
    if (
        orders.ndim != 1
        or orders.dtype.kind not in "iu"
        or len(orders) == 0
        or len(np.unique(orders)) != len(orders)
        or np.any(orders < 5)
    ):
        raise ValueError("Executed norm quadratures require distinct sufficient integer orders")
    for cell in range(int(arrays["local_count"])):
        cells, points, faces, incidence = (
            arrays[f"{name}_{cell}"] for name in ("cells", "points", "faces", "cell_faces")
        )
        sides = 3 if triangle else 4
        if (
            cells.ndim != 2
            or cells.shape[1] != sides
            or cells.dtype.kind not in "iu"
            or len(cells) == 0
            or points.ndim != 2
            or points.shape[1] != 2
            or np.any(cells < 0)
            or np.any(cells >= len(points))
            or faces.ndim != 2
            or faces.shape[1] != 2
            or faces.dtype.kind not in "iu"
            or np.any(faces < 0)
            or np.any(faces >= len(points))
            or incidence.shape != cells.shape
            or incidence.dtype.kind not in "iu"
            or np.any(incidence < 0)
            or np.any(incidence >= len(faces))
        ):
            raise ValueError("Invalid executed fine geometry/incidence")
        count, interior = (3, 3) if triangle else (2, 4)
        edge_dofs = (count * incidence[:, :, None] + np.arange(count)).reshape(len(cells), -1)
        cell_dofs = (
            count * len(faces) + interior * np.arange(len(cells))[:, None] + np.arange(interior)
        )
        if not np.array_equal(
            arrays[f"stress_dofs_{cell}"], np.column_stack((edge_dofs, cell_dofs))
        ):
            raise ValueError("Executed stress component DOFs differ from oriented fine moments")
        scalar_size = 3 if triangle else 4
        if (
            restore(arrays, f"stress_{cell}").shape
            != (count * len(faces) + interior * len(cells), 2)
            or restore(arrays, f"displacement_{cell}").shape != (len(cells), scalar_size, 2)
            or restore(arrays, f"rotation_{cell}").shape != (len(cells), 3)
        ):
            raise ValueError("Executed stress/displacement/rotation coefficient dimensions differ")


def original_checks(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Check actual local/full/gauge equations with source-normalized criterion 1e-10.

    All four local blocks are retained: constitutive, fine force moments,
    weak symmetry moments and negative-traction boundary compatibility.
    Block backward errors and the compact solve are reported separately from
    the original full saddle normalized by physical force/boundary data.
    """
    trace, rhs = restore(arrays, "global_trace"), restore(arrays, "global_rhs")
    boundary = restore(arrays, "applied_boundary_load")
    if not np.array_equal(boundary, -restore(arrays, "physical_dirichlet_moments")):
        raise ValueError(
            "Applied elasticity boundary sign differs from negative-traction convention"
        )
    fixed = arrays["fixed_trace_indices"]
    if (
        fixed.dtype.kind not in "iu"
        or np.any(fixed < 0)
        or np.any(fixed >= len(trace))
        or len(np.unique(fixed)) != len(fixed)
    ):
        raise ValueError("Invalid prescribed trace coordinates")
    if not np.array_equal(trace[fixed], arrays["fixed_trace_values"]):
        raise ValueError("Prescribed negative-traction coordinates changed")
    free = np.ones(len(trace), dtype=bool)
    free[fixed] = False
    weak = -boundary.copy()
    defect_squared = load_squared = body_load_squared = np.longdouble(0)
    rows, maximum = [], 0.0
    means = np.zeros(int(arrays["gauge_count"]), dtype=np.longdouble)
    mean_scales = np.zeros_like(means)
    for cell in range(int(arrays["local_count"])):
        matrix, coupling = _csr(arrays, f"original_a_{cell}"), restore(arrays, f"original_b_{cell}")
        field, load = (
            restore(arrays, f"internal_field_{cell}"),
            restore(arrays, f"original_f_{cell}"),
        )
        dofs, sizes = arrays[f"trace_dofs_{cell}"], arrays[f"block_sizes_{cell}"]
        if (
            sizes.shape != (4,)
            or sizes.dtype.kind not in "iu"
            or np.any(sizes <= 0)
            or sum(sizes) != len(field)
            or coupling.shape != (len(field), len(dofs))
            or dofs.dtype.kind not in "iu"
            or np.any(dofs < 0)
            or np.any(dofs >= len(trace))
            or len(np.unique(dofs)) != len(dofs)
        ):
            raise ValueError("Invalid local original block/trace map")
        combined = sparse.hstack((matrix, sparse.csr_matrix(coupling)), format="csr")
        values = np.r_[field, trace[dofs]]
        defect = accurate_residual(combined, load, values)
        action = abs(combined) @ abs(values) + abs(load)
        cuts = np.r_[0, np.cumsum(sizes)]
        blocks = [
            float(
                np.linalg.norm(defect[a:b]) / max(np.linalg.norm(action[a:b]), np.finfo(float).tiny)
            )
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]
        maximum = max(maximum, *blocks)
        defect_squared += np.sum(defect**2, dtype=np.longdouble)
        local_fixed = np.isin(dofs, fixed)
        physical_forcing = load - coupling[:, local_fixed] @ trace[dofs[local_fixed]]
        load_squared += np.sum(physical_forcing**2, dtype=np.longdouble)
        body_load_squared += np.sum(load**2, dtype=np.longdouble)
        np.add.at(weak, dofs, restore(arrays, f"test_coupling_{cell}").T @ field)
        ns, nu, nr, _ = sizes
        for name, actual in (
            ("stress", field[:ns]),
            ("displacement", field[ns : ns + nu]),
            ("rotation", field[ns + nu : ns + nu + nr]),
        ):
            if not np.array_equal(restore(arrays, f"{name}_{cell}").ravel(), actual):
                raise ValueError("Physical coefficients differ from executing original coordinates")
        z, c = restore(arrays, f"coarse_basis_z_{cell}"), restore(arrays, f"constraints_c_{cell}")
        e, lifts, source = (
            restore(arrays, f"{name}_{cell}")
            for name in ("retained_basis_e", "trace_lifts", "source_lift")
        )
        if (
            z.shape != c.shape
            or z.shape != e.shape
            or z.shape != (len(field), 3)
            or np.linalg.matrix_rank(np.asarray(c.T @ z, dtype=float)) != 3
        ):
            raise ValueError("Executing rigid moments do not pair with three local modes")
        kernel = restore(arrays, f"kernel_z_{cell}")
        kernel_action = np.asarray(matrix @ kernel, dtype=np.longdouble)
        kernel_scale = np.maximum(
            np.linalg.norm(matrix.data) * np.linalg.norm(kernel, axis=0), np.finfo(float).tiny
        )
        if np.max(np.linalg.norm(kernel_action, axis=0) / kernel_scale) > 1e-10:
            raise ValueError("Archived local rigid modes are not an operator kernel")
        pairing = c.T @ z
        moment_defects = (c.T @ source, c.T @ lifts, c.T @ e - pairing)
        moment_scale = max(float(np.linalg.norm(pairing)), np.finfo(float).tiny)
        if max(float(np.linalg.norm(d)) / moment_scale for d in moment_defects) > 1e-10:
            raise ValueError("Executed source/lift/retained physical moment pairing differs")
        reconstruction = source - lifts @ trace[dofs] + e @ restore(arrays, f"coarse_{cell}")
        reconstruction_error = float(
            np.linalg.norm(reconstruction - field)
            / max(np.linalg.norm(field), np.finfo(float).tiny)
        )
        if reconstruction_error > 1e-10:
            raise ValueError("Executed retained basis/lift reconstruction differs")
        for gauge in range(len(means)):
            weights = restore(arrays, f"physical_mean_weights_{gauge}_{cell}")
            means[gauge] += weights @ field
            mean_scales[gauge] += abs(weights) @ abs(field)
        rows.append(
            {
                "macro": cell,
                "physical_block_backward_errors": blocks,
                "executed_reconstruction_relative_error": reconstruction_error,
            }
        )
    denominator_squared = load_squared + np.sum(boundary[free] ** 2, dtype=np.longdouble)
    full_defect_squared = defect_squared + np.sum(weak[free] ** 2, dtype=np.longdouble)
    full = (
        float(np.sqrt(full_defect_squared / denominator_squared))
        if denominator_squared
        else float(np.sqrt(full_defect_squared))
    )
    coarse = np.concatenate(
        [restore(arrays, f"coarse_{cell}") for cell in range(int(arrays["local_count"]))]
    )
    global_values = np.r_[trace, coarse]
    global_matrix = _csr(arrays, "global_matrix")
    compact_defect = accurate_residual(global_matrix, rhs, global_values)
    compact_free = np.r_[free, np.ones(len(coarse), dtype=bool)]
    fixed_values = np.zeros_like(global_values)
    fixed_values[fixed] = trace[fixed]
    effective_rhs = rhs - global_matrix @ fixed_values
    denominator = max(
        np.linalg.norm(effective_rhs[compact_free]),
        np.linalg.norm(rhs[compact_free]),
        np.finfo(float).tiny,
    )
    compact = float(np.linalg.norm(compact_defect[compact_free]) / denominator)
    gauge_rows, targets = restore(arrays, "gauge_rows"), restore(arrays, "gauge_targets")
    multipliers = restore(arrays, "gauge_multipliers")
    gauge_defect = gauge_rows @ global_values - targets
    physical_targets = restore(arrays, "physical_mean_values")
    gauge_relative = float(
        np.max(
            abs(gauge_defect)
            / np.maximum(abs(targets) + abs(gauge_rows) @ abs(global_values), np.finfo(float).tiny),
            initial=0,
        )
    )
    physical_mean_relative = float(
        np.max(
            abs(means - physical_targets)
            / np.maximum(mean_scales + abs(physical_targets), np.finfo(float).tiny),
            initial=0,
        )
    )
    augmented = float(
        np.linalg.norm((compact_defect - gauge_rows.T @ multipliers)[compact_free]) / denominator
    )
    if max(maximum, full, compact, gauge_relative, physical_mean_relative, augmented) > 1e-10:
        raise ValueError("Original elasticity equations or physical mean exceed unchanged 1e-10")
    return {
        "accepted": True,
        "criterion": 1e-10,
        "full_uncondensed_relative_to_physical_rhs": full,
        "local_volume_relative_to_force": float(np.sqrt(defect_squared / load_squared))
        if load_squared
        else float(np.sqrt(defect_squared)),
        "body_force_norm": float(np.sqrt(body_load_squared)),
        "physical_rhs_with_prescribed_traction_norm": float(np.sqrt(denominator_squared)),
        "maximum_physical_block_backward_error": maximum,
        "compact_physical_relative_residual": compact,
        "augmented_compact_relative_residual": augmented,
        "physical_mean_relative_defect": physical_mean_relative,
        "reduced_gauge_relative_defect": gauge_relative,
        "physical_mean_values": means.astype(float).tolist(),
        "physical_mean_targets": physical_targets.astype(float).tolist(),
        "gauge_multipliers": multipliers.astype(float).tolist(),
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
    """Atomically write fresh actual operator/basis bytes after original and moment checks."""
    if path.exists() or path.with_suffix(".json").exists() or not acquisition_uuid:
        raise ValueError("A fresh archive and nonempty acquisition UUID are required")
    _validate_envelope(arrays)
    checks, bases = original_checks(arrays), basis_checks(arrays)
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
        "coefficient_nmant": int(arrays["coefficient_nmant"]),
        "original_checks": checks,
        "basis_checks": bases,
        "multiplier_convention": "negative Cauchy traction in the canonical face normal",
        "executed_basis": (
            "observed BDM2 monomial C or executing RT1 tensor recipe; "
            "literal oriented Piola/field/face tables"
        ),
        "physical_blocks": [
            "constitutive",
            "fine force moments",
            "weak symmetry moments",
            "negative-traction compatibility",
        ],
        "native_whole_field_agreement_verified": False,
    }
    write_progress(path.with_suffix(".json"), metadata)
    return metadata


def read_field(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Check executed identities, physical originals and finite basis semantics without a solve."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("archive_sha256") != file_digest(path):
        raise ValueError("Executed elasticity archive schema or digest differs")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if (
        arrays["schema"].tobytes().decode() != SCHEMA
        or arrays["acquisition_uuid"].tobytes().decode() != metadata["acquisition_uuid"]
        or {key: array_identity(value) for key, value in arrays.items()}
        != metadata["arrays_sha256"]
        or int(arrays["coefficient_nmant"]) != metadata["coefficient_nmant"]
        or int(arrays["coefficient_nmant"]) > np.finfo(np.longdouble).nmant
        or int(arrays["kind"]) not in (0, 1)
        or int(arrays["local_count"]) <= 0
    ):
        raise ValueError("Executed elasticity identities or coefficient precision differ")
    _validate_envelope(arrays)
    basis_checks(arrays)
    original_checks(arrays)
    return arrays, metadata
