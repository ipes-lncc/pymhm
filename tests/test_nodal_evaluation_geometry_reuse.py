"""Nodal geometry reuse preserves literal gradients and incident point order."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pymhm.fem.geometry import pullback_points
from pymhm.fem.reference import tabulate_archived_nodal_basis
from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field


def simplex_field(kind: str, components: int) -> tuple[DiscreteField, np.ndarray, np.ndarray]:
    """Use a nonorthogonal affine mesh and interleaved quadratic nodal values."""
    mesh: TriangleMesh | TetraMesh
    if kind == "triangle":
        triangle = TriangleMesh.unit_square(2)
        mesh = TriangleMesh(triangle.points @ np.array([[1.2, 0.3], [0.2, 0.9]]), triangle.cells)
        _, nodes = nodal_space(mesh, 2)
    else:
        tetrahedron = TetraMesh.unit_cube(1)
        mesh = TetraMesh(
            tetrahedron.points @ np.array([[1.2, 0.3, 0.1], [0.2, 0.9, 0.2], [0.1, 0.1, 1.1]]),
            tetrahedron.cells,
        )
        _, nodes = tetra_nodal_space(mesh, 2)
    value = nodes[:, 0] ** 2 + 2 * nodes[:, 0] * nodes[:, 1] + 3 * nodes[:, 1] ** 2
    coefficients = value if components == 1 else np.column_stack((value, 1 - 2 * value)).ravel()
    field = DiscreteField(nodal_field("quadratic", mesh, 2, components=components), coefficients)
    owners = np.array([len(mesh.cells) - 1, 0, 2, 0, 2, len(mesh.cells) - 1])
    vertices = mesh.points[mesh.cells[owners]]
    points = np.einsum("i,tia->ta", np.arange(1, vertices.shape[1] + 1), vertices)
    points /= sum(range(1, vertices.shape[1] + 1))
    return field, points, owners


@pytest.mark.parametrize("kind", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("components", [1, 2])
def test_affine_nodal_gradients_match_literal_per_point_inversion(
    kind: str, components: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated unsorted owners preserve old inverse arithmetic, basis and DOF order."""
    field, points, owners = simplex_field(kind, components)
    definition = field.definition
    dofs = (
        nodal_space(field.mesh, 2)[0] if kind == "triangle" else tetra_nodal_space(field.mesh, 2)[0]
    )
    _, reference, jacobian, _ = pullback_points(field.mesh, points, cells=owners)
    tables = tabulate_archived_nodal_basis(kind, 2, definition.basis_matrix, reference, nderiv=1)
    dimension = points.shape[1]
    # This literal per-point operation is the pre-reuse gradient oracle.
    transformed = np.einsum("dqi,qda->qia", tables[1 : dimension + 1], np.linalg.inv(jacobian))
    local = field.coefficients.reshape(-1, components)[dofs[owners]]
    literal = np.einsum("qia,qic->qca", transformed, local)
    if components == 1:
        literal = literal[:, 0]
    inverse = np.linalg.inv
    calls = []

    def counted_inverse(matrix: np.ndarray) -> np.ndarray:
        calls.append(len(matrix))
        return inverse(matrix)

    monkeypatch.setattr(np.linalg, "inv", counted_inverse)
    actual = field.gradient(points, cells=owners)
    np.testing.assert_array_equal(actual.view(np.uint8), literal.view(np.uint8))
    assert calls == [len(np.unique(owners)), len(np.unique(owners))]
    np.testing.assert_array_equal(field.definition.basis_matrix, definition.basis_matrix)


@pytest.mark.parametrize("kind", ["triangle", "tetrahedron", "quadrilateral"])
def test_value_callback_never_inverts_returned_jacobians(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values need the reference coordinates, with no subsequent inverse operation."""
    if kind == "quadrilateral":
        mesh = CartesianMacroMesh(2, 2)
        _, nodes = qk_space(mesh, 2)
        field = DiscreteField(nodal_field("quadratic", mesh, 2), nodes[:, 0] ** 2)
        owners = np.array([3, 0, 3, 1])
        points = mesh.points[mesh.cells[owners]].mean(axis=1)
    else:
        field, points, owners = simplex_field(kind, 1)
    expected = field.evaluate(points, cells=owners)
    geometry = pullback_points(field.mesh, points, cells=owners)

    def declared_pullback(*args: Any, **kwargs: Any) -> Any:
        return geometry

    def forbidden_inverse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the value callback must not invert returned Jacobians")

    monkeypatch.setattr("pymhm.postprocessing.nodal.pullback_points", declared_pullback)
    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    actual = field.evaluate(points, cells=owners)
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def test_unrecognized_geometry_retains_its_pointwise_jacobians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A third-party geometry callback can vary its map at repeated cell owners."""
    field, points, owners = simplex_field("triangle", 1)
    _, reference, jacobian, _ = pullback_points(field.mesh, points, cells=owners)
    jacobian = jacobian * np.linspace(0.9, 1.4, len(points))[:, None, None]
    tables = tabulate_archived_nodal_basis(
        "triangle", 2, field.definition.basis_matrix, reference, nderiv=1
    )
    dofs = nodal_space(field.mesh, 2)[0]
    transformed = np.einsum("dqi,qda->qia", tables[1:3], np.linalg.inv(jacobian))
    literal = np.einsum("qia,qi->qa", transformed, field.coefficients[dofs[owners]])
    third_party = SimpleNamespace(points=field.mesh.points)
    custom = DiscreteField(replace(field.definition, mesh=third_party), field.coefficients)
    determinant = np.linalg.det(jacobian)

    def pointwise_pullback(*args: Any, **kwargs: Any) -> Any:
        return owners, reference, jacobian, determinant

    inverse = np.linalg.inv
    calls = []

    def counted_inverse(matrix: np.ndarray) -> np.ndarray:
        calls.append(len(matrix))
        return inverse(matrix)

    monkeypatch.setattr("pymhm.postprocessing.nodal.pullback_points", pointwise_pullback)
    monkeypatch.setattr(np.linalg, "inv", counted_inverse)
    actual = custom.gradient(points, cells=owners)
    np.testing.assert_array_equal(actual.view(np.uint8), literal.view(np.uint8))
    assert calls == [len(points)]
