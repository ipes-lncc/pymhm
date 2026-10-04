"""Polynomial reproduction and orientation tests for the quadratic BDM element."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander
from numpy.testing import assert_allclose

from pymhm.fem.hdiv.bdm import bdm2_basis, bdm2_dofs, bdm2_evaluate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.meshes.triangle import TriangleMesh


def test_face_moments_and_affine_piola():
    """All twelve basis functions have the prescribed physical normal moments."""
    mesh = TriangleMesh(np.array([[1.0, -0.3], [2.5, 0.1], [0.5, 2.0]]), np.array([[0, 1, 2]]))
    x, w = leggauss(5)
    for side, edge in enumerate(mesh.cell_faces[0]):
        bary = np.zeros((len(x), 3))
        bary[:, side], bary[:, (side + 1) % 3] = (1 - x) / 2, (1 + x) / 2
        basis, _ = bdm2_basis(mesh, bary)
        moments = legvander(x, 2).T @ (w[:, None] / 2 * (basis[0] @ mesh.normals[edge]))
        expected = np.eye(12)[3 * side : 3 * side + 3]
        assert_allclose(mesh.lengths[edge] * moments, expected, atol=2e-14, rtol=0)


def test_normal_continuity_including_odd_edge_moments():
    """Reversing both normal and edge coordinates preserves the global P1 moment."""
    mesh = TriangleMesh.unit_square()
    coefficients = np.random.default_rng(15).normal(size=3 * (len(mesh.faces) + len(mesh.cells)))
    face = np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]
    points = mesh.points[mesh.faces[face]]
    parameter = np.linspace(0, 1, 7)
    coordinates = points[0] + parameter[:, None] * (points[1] - points[0])
    traces = []
    for cell in mesh.face_cells[face]:
        vertices = mesh.points[mesh.cells[cell]]
        bary = np.linalg.solve(
            np.vstack((vertices.T, np.ones(3))),
            np.column_stack((coordinates, np.ones(len(coordinates)))).T,
        ).T
        values, _ = bdm2_evaluate(mesh, coefficients, bary)
        traces.append(values[cell] @ mesh.normals[face])
    assert_allclose(*traces, atol=2e-13, rtol=0)


def test_quadratic_vectors_and_divergence_reproduce_exactly():
    """Fit physical quadratic coefficients and independently differentiate the vector."""
    mesh = TriangleMesh(np.array([[0.2, -0.7], [1.4, -0.1], [-0.3, 0.8]]), np.array([[0, 1, 2]]))
    bary, _ = triangle_quadrature(5)
    points = bary @ mesh.points[mesh.cells[0]]
    x, y = points.T
    exact = np.column_stack((1 + x + 2 * y + 3 * x * x - x * y, -1 + 2 * x - y + x * y + 2 * y * y))
    basis, _ = bdm2_basis(mesh, bary)
    local = np.linalg.lstsq(basis[0].transpose(0, 2, 1).reshape(-1, 12), exact.ravel(), rcond=None)[
        0
    ]
    coefficients = np.empty(12)
    coefficients[bdm2_dofs(mesh)[0]] = local
    values, divergence = bdm2_evaluate(mesh, coefficients, bary)
    assert_allclose(values[0], exact, atol=2e-13, rtol=0)
    assert_allclose(divergence[0], 7 * x + 3 * y, atol=2e-13, rtol=0)
    tensor, tensor_div = bdm2_evaluate(
        mesh, np.column_stack((coefficients, 2 * coefficients)), bary
    )
    assert_allclose(tensor[:, :, 1], 2 * values, atol=1e-13)
    assert_allclose(tensor_div[:, :, 1], 2 * divergence, atol=1e-13)


@pytest.mark.parametrize(
    "bary", [np.ones(3), np.ones((2, 2)), [[0, 0, 0]], [[np.nan, 0, 1]], [[1j, 0, 1]]]
)
def test_invalid_barycentric_coordinates(bary):
    with pytest.raises(ValueError, match="barycentric"):
        bdm2_basis(TriangleMesh.unit_square(), bary)


@pytest.mark.parametrize(
    "coefficients",
    [0, np.zeros(3), np.zeros((21, 2, 2)), np.full(21, np.nan), np.ones(21, dtype=complex)],
)
def test_invalid_coefficient_layout(coefficients):
    with pytest.raises(ValueError, match="coefficients"):
        bdm2_evaluate(TriangleMesh.unit_square(), coefficients, np.array([[1 / 3] * 3]))
