"""Strict geometric ownership distinguishes roundoff from a declared tolerance."""

from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.introduction.scalar import ScalarField, evaluate_incident_reference, evaluate_scalar
from examples.introduction.vector import TriangleVectorEvaluator
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.sampling import TrianglePointLocator


def _determinant(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Fraction:
    """Use the exact rational values of the supplied binary64 coordinates."""
    ax, ay = map(Fraction, map(float, a))
    bx, by = map(Fraction, map(float, b))
    cx, cy = map(Fraction, map(float, c))
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _contains(vertices: np.ndarray, point: np.ndarray) -> bool:
    """Check exact closed-triangle membership independently of affine inverses."""
    orientation = _determinant(*vertices)
    return all(
        _determinant(vertices[i], vertices[(i + 1) % 3], point) * orientation >= 0 for i in range(3)
    )


@pytest.mark.parametrize("clockwise", [False, True])
def test_strict_reported_triangle_accepts_its_literal_vertices(clockwise: bool) -> None:
    """Every stored mesh vertex belongs to its incident triangle at zero tolerance."""
    points = np.array(
        [
            [0.2809263548406665, 0.2751172739539216],
            [0.1321905042821152, 0.9979412786111252],
            [0.11075820432308381, 0.2049380724672395],
        ]
    )
    cells = np.array([[0, 2, 1] if clockwise else [0, 1, 2]])
    mesh = TriangleMesh(points, cells)
    locator = TrianglePointLocator(mesh, tolerance=0)
    assert_array_equal(locator.locate(points, allow_outside=True), np.zeros(3, dtype=int))
    owners, bary = locator.coordinates(points, cells=np.zeros(3, dtype=int))
    assert_array_equal(owners, np.zeros(3, dtype=int))
    assert_allclose(bary @ mesh.points[mesh.cells[0]], points, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("kind", ["non_dyadic", "refined", "translated", "scaled", "sheared"])
def test_strict_nondyadic_mesh_nodes_and_incident_coordinates(kind: str) -> None:
    """Nonbinary subdivisions and affine maps retain all original node ownership."""
    mesh = TriangleMesh.unit_square(9)
    if kind == "refined":
        mesh = TriangleMesh.unit_square(3).submesh(0, 7)
    elif kind == "translated":
        mesh = TriangleMesh(mesh.points + [0.1, -0.3], mesh.cells)
    elif kind == "scaled":
        mesh = TriangleMesh(mesh.points * [0.6, 1.7], mesh.cells)
    elif kind == "sheared":
        mesh = TriangleMesh(mesh.points @ np.array([[1.0, 0.3], [-0.2, 1.1]]), mesh.cells)
    locator = TrianglePointLocator(mesh, candidates=1, tolerance=0)
    assert np.all(locator.locate(mesh.points, allow_outside=True) >= 0)
    incident: np.ndarray = np.repeat(np.arange(len(mesh.cells)), 3)
    vertices = mesh.points[mesh.cells].reshape(-1, 2)
    owners, bary = locator.coordinates(vertices, cells=incident)
    assert_array_equal(owners, incident)
    assert_allclose(
        np.einsum("qi,qia->qa", bary, mesh.points[mesh.cells[owners]]),
        vertices,
        atol=1e-12,
        rtol=1e-10,
    )


@pytest.mark.parametrize("clockwise", [False, True])
def test_strict_exact_edges_and_nearby_interior_use_physical_signs(clockwise: bool) -> None:
    """Dyadic edges include exact points while an immediately exterior float stays out."""
    mesh = TriangleMesh(
        np.array([[0.25, -0.125], [2.25, -0.125], [0.75, 0.875]]),
        np.array([[0, 2, 1] if clockwise else [0, 1, 2]]),
    )
    vertices = mesh.points[mesh.cells[0]]
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    edges = np.concatenate(
        [
            (1 - fractions[:, None]) * vertices[i] + fractions[:, None] * vertices[(i + 1) % 3]
            for i in range(3)
        ]
    )
    points = np.vstack(
        (edges, [[0.75, np.nextafter(-0.125, np.inf)], [0.75, np.nextafter(-0.125, -np.inf)]])
    )
    expected = np.array([0 if _contains(vertices, point) else -1 for point in points])
    assert np.all(expected[:-1] == 0)
    assert expected[-1] == -1
    locator = TrianglePointLocator(mesh, tolerance=0)
    assert_array_equal(locator.locate(points, allow_outside=True), expected)
    locator.coordinates(points[:-1], cells=np.zeros(len(points) - 1, dtype=int))
    with pytest.raises(ValueError, match="incident"):
        locator.coordinates(points[-1:], cells=[0])


def test_strict_nextafter_exterior_is_not_an_implicit_roundoff_tolerance() -> None:
    """Even subnormal displacement outside an exact edge remains geometrically exterior."""
    mesh = TriangleMesh(np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 1, 2]]))
    points = np.array(
        [
            [np.nextafter(0.5, 0), 0.5],
            [0.5, 0.5],
            [np.nextafter(0.5, 1), 0.5],
            [np.nextafter(0.0, -np.inf), 0.25],
            [0.25, np.nextafter(0.0, -np.inf)],
            [0.0, np.nextafter(1.0, np.inf)],
        ]
    )
    expected = np.array([0 if _contains(mesh.points, point) else -1 for point in points])
    assert_array_equal(expected, [0, 0, -1, -1, -1, -1])
    locator = TrianglePointLocator(mesh, tolerance=0)
    assert_array_equal(locator.locate(points, allow_outside=True), expected)
    with pytest.raises(ValueError, match="outside"):
        locator.locate(points)
    for point in points[2:]:
        with pytest.raises(ValueError, match="incident"):
            locator.coordinates(point[None], cells=[0])


def test_positive_tolerance_preserves_literal_boundary_and_unclamped_coordinates() -> None:
    """Tolerance is the explicit barycentric bound, including its immediately excluded float."""
    mesh = TriangleMesh(np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 1, 2]]))
    tolerance = 2.0**-12
    points = np.array(
        [
            [-tolerance, 0.25],
            [np.nextafter(-tolerance, -np.inf), 0.25],
            [0.5, 0.5 + tolerance],
            [0.5, np.nextafter(0.5 + tolerance, np.inf)],
        ]
    )
    locator = TrianglePointLocator(mesh, tolerance=tolerance)
    assert_array_equal(locator.locate(points, allow_outside=True), [0, -1, 0, -1])
    owners, bary = locator.coordinates(points[[0, 2]], cells=[0, 0])
    assert_array_equal(owners, [0, 0])
    assert_array_equal(bary.min(axis=1), [-tolerance, -tolerance])
    local = np.einsum(
        "qba,qa->qb", locator.inverse[owners], points[[0, 2]] - locator.origins[owners]
    )
    assert_array_equal(bary, np.column_stack((1 - local.sum(axis=1), local)))
    for point in points[[1, 3]]:
        with pytest.raises(ValueError, match="incident"):
            locator.coordinates(point[None], cells=[0])


@pytest.mark.parametrize("kind", ["original", "translated", "refined", "arbitrary"])
def test_scalar_helper_evaluates_literal_nodes_without_graphics_or_point_motion(kind: str) -> None:
    """A strict introductory scalar evaluator reproduces an independently defined affine field."""
    mesh = TriangleMesh.unit_square(9)
    if kind == "translated":
        mesh = TriangleMesh(mesh.points + [0.1, -0.3], mesh.cells)
    elif kind == "refined":
        mesh = TriangleMesh.unit_square(3).submesh(0, 7)
    elif kind == "arbitrary":
        mesh = TriangleMesh(
            np.array(
                [
                    [0.2809263548406665, 0.2751172739539216],
                    [0.1321905042821152, 0.9979412786111252],
                    [0.11075820432308381, 0.2049380724672395],
                ]
            ),
            np.array([[0, 1, 2]]),
        )
    _, nodes = nodal_space(mesh, 1)
    gradient = np.array([3.0, -2.0])
    field = ScalarField(mesh, 1, 1 + nodes @ gradient)
    points = np.vstack((mesh.points, mesh.points[mesh.cells].mean(axis=1)))
    value, derivative = evaluate_scalar(field, points)
    assert_allclose(value, 1 + points @ gradient, atol=1e-12, rtol=1e-10)
    assert_allclose(derivative, np.tile(gradient, (len(points), 1)), atol=1e-12, rtol=1e-10)
    outside = np.array([[mesh.points[:, 0].min() - 1, mesh.points[:, 1].min()]])
    assert_array_equal(field.locator(*outside.T), [-1])
    with pytest.raises(ValueError, match="outside"):
        evaluate_scalar(field, outside)


def test_duplicated_interfaces_retain_incident_scalar_gradients_and_vector_curls() -> None:
    """Two coincident queries preserve distinct physical derivatives on their declared sides."""
    mesh = TriangleMesh.unit_square()
    points = np.tile([[0.5, 0.5]], (2, 1))
    owners = np.array([0, 1])
    locator = TrianglePointLocator(mesh, tolerance=0)
    selected, bary = locator.coordinates(points, cells=owners)
    assert_array_equal(selected, owners)
    assert_allclose(bary.sum(axis=1), 1, atol=1e-12, rtol=1e-10)
    assert_array_equal(locator.locate(points), locator.locate(points))
    _, nodes = nodal_space(mesh, 1)
    cusp = np.abs(nodes[:, 0] - nodes[:, 1])
    scalar = ScalarField(mesh, 1, cusp)
    value, gradient = evaluate_scalar(scalar, points, cells=owners)
    assert_allclose(value, 0, atol=1e-12, rtol=1e-10)
    assert_allclose(gradient, [[1, -1], [-1, 1]], atol=1e-12, rtol=1e-10)
    centers = mesh.points[mesh.cells].mean(axis=1)
    incident_value, incident_gradient = evaluate_incident_reference(scalar, points, centers)
    assert_array_equal(incident_value, value)
    assert_array_equal(incident_gradient, gradient)
    vector = TriangleVectorEvaluator(mesh, 1, np.column_stack((np.zeros(len(nodes)), cusp)))
    velocity, jacobian = vector.field.values_and_gradient(points, cells=selected)
    assert_allclose(velocity, 0, atol=1e-12, rtol=1e-10)
    assert_allclose(jacobian[:, 1, 0] - jacobian[:, 0, 1], [1, -1], atol=1e-12, rtol=1e-10)
