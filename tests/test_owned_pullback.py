"""Explicit field owners preserve one-sided geometry with bounded point storage."""

from typing import Any

import numpy as np
import pytest

from pymhm.fem.geometry import pullback_points
from pymhm.meshes.hexahedron import HexMesh, hexahedral_mapping
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field


@pytest.mark.parametrize(
    "mesh",
    [TriangleMesh.unit_square(3), TetraMesh.unit_cube(2), AffineMixedMesh.unit_cube(2, "prism")],
)
def test_declared_affine_owners_recover_actual_geometry(mesh: Any) -> None:
    """Repeated and unsorted owners retain their independent physical maps."""
    owners = np.array([len(mesh.cells) - 1, 0, 2, 0])
    vertices = mesh.points[mesh.cells[owners]]
    points = vertices.mean(axis=1)
    expected = pullback_points(mesh, points)
    actual = pullback_points(mesh, points, cells=owners)
    np.testing.assert_array_equal(actual[0], owners)
    for found, reference in zip(actual[1:], expected[1:], strict=True):
        np.testing.assert_allclose(found, reference, atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="one-sided"):
        pullback_points(mesh, points + 10, cells=owners)
    empty = pullback_points(
        mesh, np.empty((0, mesh.points.shape[1])), cells=np.array([], dtype=int)
    )
    assert empty[1].shape == (0, mesh.points.shape[1])


def test_point_owners_bound_affine_work_independently_of_mesh_size(monkeypatch: Any) -> None:
    """A small query never inverts maps for thousands of unrelated cells."""
    mesh = TriangleMesh.unit_square(80)
    owners = np.array([0, 1, 100, len(mesh.cells) - 1, 100])
    points = mesh.points[mesh.cells[owners]].mean(axis=1)
    invert = np.linalg.inv
    sizes = []

    def bounded_inverse(matrix: np.ndarray) -> np.ndarray:
        sizes.append(len(matrix))
        assert len(matrix) <= len(np.unique(owners))
        return invert(matrix)

    monkeypatch.setattr(np.linalg, "inv", bounded_inverse)
    definition = nodal_field("linear", mesh, 1)
    field = DiscreteField(definition, mesh.points[:, 0] + 2 * mesh.points[:, 1])
    values, gradient = field.values_and_gradient(points, cells=owners)
    np.testing.assert_allclose(values, points[:, 0] + 2 * points[:, 1], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(gradient, np.tile([1.0, 2.0], (len(points), 1)), atol=1e-12)
    assert sizes and max(sizes) == len(np.unique(owners))


@pytest.mark.parametrize(
    "mesh",
    [TriangleMesh.unit_square(3), TetraMesh.unit_cube(2), AffineMixedMesh.unit_cube(2, "prism")],
)
def test_affine_pullback_reuses_literal_maps_without_roundoff_changes(
    mesh: Any, monkeypatch: Any
) -> None:
    """Unique-owner inversion preserves the literal per-point inverse and signed det."""
    owners = np.array([len(mesh.cells) - 1, 0, 2, 0, 2, 0])
    vertices = mesh.points[mesh.cells[owners]]
    points = vertices.mean(axis=1)
    jacobian = (
        mesh.jacobian[owners]
        if hasattr(mesh, "jacobian")
        else (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
    )
    expected_reference = np.einsum("qij,qj->qi", np.linalg.inv(jacobian), points - vertices[:, 0])
    expected_determinant = np.linalg.det(jacobian)
    invert, determinant = np.linalg.inv, np.linalg.det
    counts: dict[str, list[int]] = {"inverse": [], "determinant": []}

    def counted_inverse(matrix: np.ndarray) -> np.ndarray:
        counts["inverse"].append(len(matrix))
        return invert(matrix)

    def counted_determinant(matrix: np.ndarray) -> np.ndarray:
        counts["determinant"].append(len(matrix))
        return determinant(matrix)

    monkeypatch.setattr(np.linalg, "inv", counted_inverse)
    monkeypatch.setattr(np.linalg, "det", counted_determinant)
    actual = pullback_points(mesh, points, cells=owners)
    for found, expected in zip(
        actual, (owners, expected_reference, jacobian, expected_determinant), strict=True
    ):
        np.testing.assert_array_equal(found, expected)
    assert counts == {"inverse": [3], "determinant": [3]}


def test_trilinear_paired_maps_and_owned_inverse_preserve_curvature() -> None:
    """Curved cells return each point's own Jacobian, including repeated owners."""
    base = HexMesh.unit_cube(2)
    coordinates = base.points.copy()
    coordinates[:, 0] += 0.2 * coordinates[:, 1] * coordinates[:, 2]
    mesh = HexMesh(coordinates, base.cells)
    owners = np.array([6, 0, 6, 3])
    corners = mesh.points[mesh.cells[owners]]
    reference = np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3], [1.0, 0.3, 0.4], [0.6, 0.5, 0.1]])
    physical, jacobians, determinant = hexahedral_mapping(corners, reference, paired=True)
    batch = hexahedral_mapping(corners, reference)
    for value, expanded in zip((physical, jacobians, determinant), batch, strict=True):
        np.testing.assert_allclose(value, expanded[np.arange(4), np.arange(4)], atol=1e-12)
    actual = pullback_points(mesh, physical, cells=owners)
    np.testing.assert_array_equal(actual[0], owners)
    for value, expected in zip(actual[1:], (reference, jacobians, determinant), strict=True):
        np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="one-sided"):
        pullback_points(mesh, physical + 2, cells=owners)
    with pytest.raises(ValueError, match="one cell per"):
        hexahedral_mapping(corners, reference[:1], paired=True)
    with pytest.raises(TypeError, match="boolean"):
        hexahedral_mapping(corners, reference, paired=1)


@pytest.mark.parametrize("owners", [[-1], [2], [0.5], [True], [0, 1]])
def test_declared_owners_reject_invalid_cell_selections(owners: Any) -> None:
    """Explicit ownership requires one actual integer cell per point."""
    with pytest.raises(ValueError, match="valid integer"):
        pullback_points(TriangleMesh.unit_square(), [[0.2, 0.1]], cells=owners)


def test_discontinuous_field_keeps_both_interface_values() -> None:
    """Fast explicit ownership never averages independent traces at a shared edge."""
    mesh = TriangleMesh.unit_square()
    field = DiscreteField(nodal_field("jump", mesh, 0, discontinuous=True), [2.0, 5.0])
    points = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(field.evaluate(points, cells=[0, 1]), [2.0, 5.0], atol=1e-12)
