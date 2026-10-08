"""General field and boundary contracts across polynomial families and PDE terms."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.boundary_forms import (
    boundary_bilinear,
    boundary_functional,
    boundary_quadrature,
)
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField, FieldDefinition, solution_field
from pymhm.postprocessing.nodal import nodal_field


def _mesh(kind: str) -> Any:
    """Use physical domains with anisotropic Jacobians and multiple fine cells."""
    if kind == "triangle":
        return TriangleMesh.unit_square(2)
    if kind == "tetrahedron":
        return TetraMesh.unit_cube()
    return CartesianMacroMesh(2, 1, (-1.0, 3.0, 0.0, 2.0))


@pytest.mark.parametrize("kind", ["triangle", "tetrahedron", "quadrilateral"])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("discontinuous", [False, True])
def test_nodal_values_gradients_and_recorded_replay(
    kind: str, components: int, discontinuous: bool
) -> None:
    """Quadratic scalar/vector fields replay actual basis matrices under BLAS thread changes."""
    mesh = _mesh(kind)
    degree = 3
    definition = nodal_field(
        "field", mesh, degree, components=components, discontinuous=discontinuous
    )
    space = {"triangle": nodal_space, "tetrahedron": tetra_nodal_space, "quadrilateral": qk_space}[
        kind
    ]
    dofs, nodes = space(mesh, degree)
    field_nodes = nodes[dofs].reshape(-1, nodes.shape[1]) if discontinuous else nodes
    coefficients = np.column_stack(
        [(axis + 1) * np.sum(field_nodes**2, axis=1) for axis in range(components)]
    ).ravel()
    points = mesh.points[mesh.cells].mean(axis=1)
    field = DiscreteField(definition, coefficients)
    expected = np.column_stack(
        [(axis + 1) * np.sum(points**2, axis=1) for axis in range(components)]
    )
    gradient = np.stack([(axis + 1) * 2 * points for axis in range(components)], axis=1)
    if components == 1:
        expected, gradient = expected[:, 0], gradient[:, 0]
    with threadpool_limits(1):
        assert_allclose(field.evaluate(points), expected, atol=1e-12, rtol=1e-10)
        assert_allclose(field.gradient(points), gradient, atol=1e-12, rtol=1e-10)
    replay = pickle.loads(pickle.dumps(field))
    with threadpool_limits(2):
        assert_allclose(replay.evaluate(points), expected, atol=1e-12, rtol=1e-10)
        assert_allclose(replay.gradient(points), gradient, atol=1e-12, rtol=1e-10)
    assert replay.basis_digest == field.basis_digest
    assert not replay.definition.evaluator.basis_matrix.flags.writeable
    assert not replay.definition.evaluator.dofs.flags.writeable


@pytest.mark.parametrize("kind", ["triangle", "tetrahedron", "quadrilateral"])
def test_piecewise_constant_fields_preserve_one_sided_values(kind: str) -> None:
    """DG interfaces retain the caller's owner instead of averaging separate coefficients."""
    mesh = _mesh(kind)
    values = np.arange(len(mesh.cells), dtype=float) + 1
    field = DiscreteField(nodal_field("piecewise", mesh, 0, discontinuous=True), values)
    point = mesh.points[mesh.cells[0]].mean(axis=0)[None]
    assert_allclose(field.evaluate(point, cells=np.array([0])), values[:1], atol=1e-12, rtol=1e-10)
    assert_allclose(field.gradient(point, cells=np.array([0])), 0, atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="declared one-sided"):
        field.evaluate(point, cells=np.array([len(mesh.cells) - 1]))


@pytest.mark.parametrize("kind", ["triangle", "tetrahedron", "quadrilateral"])
def test_boundary_forms_reuse_rules_for_reaction_and_affine_robin(kind: str) -> None:
    """Boundary mass/load consistency and divergence theorem exercise distinct physical uses."""
    mesh = _mesh(kind)
    size, rules = boundary_quadrature(mesh, 3, order=1)
    ones = np.ones(size)
    mass = boundary_bilinear(size, rules)
    load = boundary_functional(size, rules, 1)
    assert_allclose(mass @ ones, load, atol=1e-12, rtol=1e-10)
    surface = sum(float(rule.weights.sum()) for rule in rules)
    assert_allclose(ones @ mass @ ones, surface, atol=1e-12, rtol=1e-10)
    dimension = mesh.points.shape[1]
    robin = boundary_bilinear(size, rules, lambda points, normal: (points @ normal) / dimension)
    volumes = mesh.volumes if kind == "tetrahedron" else mesh.areas
    assert_allclose(ones @ robin @ ones, volumes.sum(), atol=1e-12, rtol=1e-10)
    projected = boundary_functional(size, rules, lambda points, normal: points @ normal)
    assert_allclose(projected, dimension * (robin @ ones), atol=1e-12, rtol=1e-10)
    assert boundary_bilinear(size, ()).nnz == 0
    assert_array_equal(boundary_functional(size, ()), np.zeros(size))


def test_field_reconstruction_combines_local_trace_orientation_and_affine_offset() -> None:
    """Generic moment recovery includes local, global trace and prescribed-field parts."""
    local = sparse.csr_matrix([[2.0, 0.0], [0.0, -1.0]])
    trace = sparse.csr_matrix([[0.0, -3.0], [4.0, 0.0]])
    definition = FieldDefinition(
        "pressure",
        reconstruction=local,
        trace_reconstruction=trace,
        trace_dofs=[2, 0],
        offset=[1.0, 2.0],
    )
    solution = SimpleNamespace(
        field_data=((definition,),), fields=(np.array([5.0, 6.0]),), trace=np.array([7.0, 8.0, 9.0])
    )
    field = solution_field(solution, "pressure")[0]
    assert_array_equal(field.coefficients, [-10.0, 32.0])
    replay = pickle.loads(pickle.dumps(solution))
    assert_array_equal(solution_field(replay, "pressure")[0].coefficients, field.coefficients)
    assert replay.field_data[0][0].basis_digest == definition.basis_digest
    assert not replay.field_data[0][0].trace_reconstruction.data.flags.writeable
    assert not replay.field_data[0][0].trace_dofs.flags.writeable
    assert not replay.field_data[0][0].offset.flags.writeable
    assert definition.basis_digest != FieldDefinition("pressure", reconstruction=local).basis_digest
    empty = FieldDefinition(
        "face_only",
        reconstruction=np.empty((2, 0)),
        trace_reconstruction=np.eye(2),
        trace_dofs=[0, 2],
    )
    empty_solution = SimpleNamespace(
        field_data=((empty,),), fields=(np.empty(0),), trace=solution.trace
    )
    assert_array_equal(solution_field(empty_solution, "face_only")[0].coefficients, [7.0, 9.0])


@pytest.mark.parametrize(
    "options",
    [
        {"trace_reconstruction": np.eye(2)},
        {"trace_dofs": [0, 1]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": [[0, 1]]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": [-1, 1]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": [0, 0]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": [0.0, 1.0]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": [0]},
        {"trace_reconstruction": np.ones(2), "trace_dofs": [0, 1]},
        {"reconstruction": np.eye(3), "trace_reconstruction": np.eye(2), "trace_dofs": [0, 1]},
        {"offset": np.eye(2)},
        {"reconstruction": np.eye(2), "offset": [1.0]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": [0, 1], "offset": [1.0]},
        {"trace_reconstruction": np.eye(2), "trace_dofs": np.array([0, 2**63], dtype=np.uint64)},
    ],
)
def test_invalid_field_recovery_contracts(options: dict[str, Any]) -> None:
    """Reject ambiguous orientation, dimension, affine and trace-coordinate declarations."""
    with pytest.raises(ValueError):
        FieldDefinition("field", **options)


def test_field_recovery_rejects_trace_and_offset_outside_executed_solution() -> None:
    """Validation uses the executed solution rather than a plausible but unrelated vector."""
    definitions = [
        FieldDefinition("p", trace_reconstruction=np.eye(1), trace_dofs=[2]),
        FieldDefinition("p", offset=[1.0, 2.0]),
    ]
    for definition in definitions:
        solution = SimpleNamespace(
            field_data=((definition,),), fields=(np.ones(1),), trace=np.ones(1)
        )
        with pytest.raises(ValueError):
            solution_field(solution, "p")
    empty = FieldDefinition(
        "p",
        reconstruction=np.empty((1, 0)),
        trace_reconstruction=np.empty((1, 0)),
        trace_dofs=[],
        offset=[3.0],
    )
    solution = SimpleNamespace(field_data=((empty,),), fields=(np.empty(0),), trace=np.empty(0))
    assert_array_equal(solution_field(solution, "p")[0].coefficients, [3.0])


@pytest.mark.parametrize(
    "option", [{"degree": 0}, {"degree": 1, "discontinuous": 1}, {"degree": 1, "mesh": object()}]
)
def test_invalid_nodal_field_contract(option: dict[str, Any]) -> None:
    """Constant and non-mesh fields require an explicit supported evaluation convention."""
    mesh = option.pop("mesh", TriangleMesh.unit_square())
    with pytest.raises((ValueError, TypeError)):
        nodal_field("field", mesh, **option)


@pytest.mark.parametrize(
    "points, owners",
    [
        ([[2.0, 2.0]], None),
        ([[0.1]], None),
        ([[0.1, 0.1]], [99]),
        ([[0.1, 0.1]], [0.0]),
        ([[0.1, 0.1]], [-1]),
        ([[0.1, 0.1]], [[0]]),
    ],
)
def test_invalid_nodal_evaluation_contract(points: Any, owners: Any) -> None:
    """Spatial samples reject missing support and noninteger/out-of-range one-sided owners."""
    mesh = TriangleMesh.unit_square()
    field = DiscreteField(nodal_field("p", mesh, 1), np.ones(len(mesh.points)))
    with pytest.raises(ValueError):
        field.evaluate(points, cells=owners)
    bad = DiscreteField(field.definition, np.ones(1))
    with pytest.raises(ValueError):
        bad.evaluate([[0.1, 0.1]])


def test_boundary_forms_reject_unavailable_geometry_and_nonreal_coefficients() -> None:
    """Surface integration rejects missing geometry and coefficients in another scalar field."""
    with pytest.raises(TypeError):
        boundary_quadrature(object(), 1)
    mesh = TriangleMesh.unit_square()
    size, rules = boundary_quadrature(mesh, 1)
    with pytest.raises(ValueError):
        boundary_bilinear(size, rules, 1j)
