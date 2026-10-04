"""Canonical RT moments, physical Piola transforms and divergence commutation."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander
from numpy.testing import assert_allclose

from pymhm.fem.hdiv.rt import (
    rt_basis,
    rt_degree,
    rt_dofs,
    rt_evaluate,
    rt_interior_tests,
    rt_interpolate,
)
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_canonical_normal_moments_on_skew_triangle(degree):
    """The Piola-mapped reference basis has identity physical normal moments."""
    mesh = TriangleMesh(np.array([[0.2, -0.4], [1.8, -0.1], [-0.2, 1.3]]), np.array([[0, 1, 2]]))
    x, weights = leggauss(5)
    width = degree + 1
    for side, face in enumerate(mesh.cell_faces[0]):
        bary = np.zeros((len(x), 3))
        bary[:, side], bary[:, (side + 1) % 3] = (1 - x) / 2, (1 + x) / 2
        basis, _ = rt_basis(mesh, degree, bary)
        moments = legvander(x, degree).T @ (weights[:, None] / 2 * (basis[0] @ mesh.normals[face]))
        expected = np.eye(width * (degree + 3))[width * side : width * (side + 1)]
        assert_allclose(mesh.lengths[face] * moments, expected, atol=3e-13, rtol=0)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_polynomial_reproduction_and_interpolation_divergence(degree):
    """Reproduce a radial RT polynomial and commute divergence on a skew mesh."""
    mesh = TriangleMesh(np.array([[0.2, -0.4], [1.8, -0.1], [-0.2, 1.3]]), np.array([[0, 1, 2]]))

    def exact(points):
        """A vector in RT_m, including its homogeneous radial enrichment."""
        x, y = points.T
        scalar = x**degree + y**degree
        return np.column_stack((1 + x * scalar, -2 + y * scalar))

    bary, weights = triangle_quadrature(6)
    coefficients = rt_interpolate(mesh, exact, degree)
    values, divergence = rt_evaluate(mesh, coefficients, degree, bary)
    points = bary @ mesh.points[mesh.cells[0]]
    expected_divergence = (degree + 2) * (points[:, 0] ** degree + points[:, 1] ** degree)
    assert_allclose(values[0], exact(points), atol=4e-12, rtol=0)
    assert_allclose(divergence[0], expected_divergence, atol=4e-12, rtol=0)

    def outside(points):
        """A cubic field not generally contained in the lower RT spaces."""
        x, y = points.T
        return np.column_stack((x**3 + y * y, x * x * y + y**3))

    field = rt_interpolate(mesh, outside, degree)
    _, interpolated_divergence = rt_evaluate(mesh, field, degree, bary)
    exact_divergence = 4 * points[:, 0] ** 2 + 3 * points[:, 1] ** 2
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            moment = weights @ (
                (interpolated_divergence[0] - exact_divergence)
                * points[:, 0] ** i
                * points[:, 1] ** j
            )
            assert abs(moment) < 5e-12
    assert rt_dofs(mesh, degree).shape == (1, (degree + 1) * (degree + 3))


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_normal_continuity_for_random_coefficients(degree):
    """The odd-moment orientation reversal gives identical two-sided normal values."""
    mesh = TriangleMesh.unit_square()
    coefficients = np.random.default_rng(19).normal(
        size=(degree + 1) * len(mesh.faces) + degree * (degree + 1) * len(mesh.cells)
    )
    face = np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]
    edge = mesh.points[mesh.faces[face]]
    points = edge[0] + np.linspace(0, 1, 7)[:, None] * (edge[1] - edge[0])
    normal = []
    for cell in mesh.face_cells[face]:
        vertices = mesh.points[mesh.cells[cell]]
        bary = np.linalg.solve(
            np.vstack((vertices.T, np.ones(3))), np.column_stack((points, np.ones(len(points)))).T
        ).T
        values, _ = rt_evaluate(mesh, coefficients, degree, bary)
        normal.append(values[cell] @ mesh.normals[face])
    assert_allclose(*normal, atol=3e-12, rtol=0)


def test_rt_validation_and_empty_interior_space():
    """Reject unsupported orders, malformed coordinates and incompatible coefficients."""
    mesh = TriangleMesh.unit_square()
    for degree in (-1, True, 1.5):
        with pytest.raises(ValueError, match="RT degree"):
            rt_degree(degree)
    for bary in ([1, 0, 0], [[0, 0]], [[0, 0, 0]], [[np.nan, 0, 1]], [[1j, 0, 1]]):
        with pytest.raises(ValueError, match="barycentric"):
            rt_basis(mesh, 1, bary)
    size = 2 * len(mesh.faces) + 2 * len(mesh.cells)
    for coefficients in (
        np.zeros(2),
        np.zeros((size, 1)),
        np.ones(size, dtype=complex),
        np.full(size, np.inf),
    ):
        with pytest.raises(ValueError, match="coefficients"):
            rt_evaluate(mesh, coefficients, 1, np.array([[1 / 3] * 3]))
    with pytest.raises(ValueError, match="quadrature"):
        rt_interpolate(mesh, [1, 2], 2, order=3)
    assert rt_interior_tests(0, np.zeros((3, 2))).shape == (3, 0)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_incident_point_evaluation_matches_cellwise_tabulation(degree):
    """Arbitrary point order and explicit interface sides preserve physical RT fields."""
    from pymhm.fem.hdiv.rt import rt_evaluate_points

    mesh = TriangleMesh.unit_square(2)
    size = (degree + 1) * len(mesh.faces) + degree * (degree + 1) * len(mesh.cells)
    coefficients = np.random.default_rng(381).normal(size=size)
    bary = np.array([[0.1, 0.2, 0.7], [0.0, 0.5, 0.5], [0.2, 0.6, 0.2]])
    points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells]).reshape(-1, 2)
    owners = np.repeat(np.arange(len(mesh.cells)), len(bary))
    expected, expected_div = rt_evaluate(mesh, coefficients, degree, bary)
    actual, actual_div = rt_evaluate_points(mesh, coefficients, degree, points[::-1], owners[::-1])
    assert_allclose(actual[::-1], expected.reshape(-1, 2), rtol=1e-12, atol=1e-12)
    assert_allclose(actual_div[::-1], expected_div.ravel(), rtol=1e-12, atol=2e-11)
    with pytest.raises(ValueError, match="coefficients"):
        rt_evaluate_points(mesh, coefficients[:-1], degree, points, owners)
    for bad_points, bad_owners in (
        (points + 1j, owners),
        (points, owners.astype(float)),
        (points, owners + len(mesh.cells)),
    ):
        with pytest.raises(ValueError, match="incident-cell"):
            rt_evaluate_points(mesh, coefficients, degree, bad_points, bad_owners)
    with pytest.raises(ValueError, match="outside"):
        rt_evaluate_points(mesh, coefficients, degree, points + 3, owners)
