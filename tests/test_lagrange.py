"""Polynomial completeness, geometric continuity and signed high-order traces."""

from math import factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import (
    multiindices,
    nodal_space,
    reference_basis,
    tabulate,
    trace_coupling,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_cardinal_reference_basis_and_simplex_lattice(degree):
    indices = multiindices(degree)
    assert len(indices) == (degree + 1) * (degree + 2) // 2
    assert_allclose(indices.sum(axis=1), degree)
    assert_allclose(indices[:3], degree * np.eye(3))
    values, _, _ = reference_basis(degree, indices / degree)
    assert_allclose(values, np.eye(len(indices)), atol=3e-14)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_physical_basis_reproduces_all_polynomials_and_two_derivatives(degree):
    square = TriangleMesh.unit_square(2)
    mesh = TriangleMesh(
        square.points @ np.array([[1.3, 0.2], [-0.4, 0.9]]) + [0.13, -0.27], square.cells
    )
    bary, _ = triangle_quadrature(4)
    dofs, points, basis, gradient, hessian = tabulate(mesh, degree, bary)
    physical = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    assert_allclose(basis.sum(axis=1), 1.0, atol=2e-14)
    assert_allclose(gradient.sum(axis=2), 0.0, atol=5e-14)
    assert_allclose(hessian.sum(axis=2), 0.0, atol=6e-13)
    assert_allclose(hessian, hessian.swapaxes(-1, -2), atol=1e-14)
    for xpower in range(degree + 1):
        for ypower in range(degree + 1 - xpower):
            values = points[:, 0] ** xpower * points[:, 1] ** ypower
            interpolated = values[dofs] @ basis.T
            exact = physical[:, :, 0] ** xpower * physical[:, :, 1] ** ypower
            assert_allclose(interpolated, exact, atol=6e-14)
            for derivative_order in (1, 2):
                for directions in np.ndindex(*(2,) * derivative_order):
                    dx, dy = directions.count(0), directions.count(1)
                    if xpower < dx or ypower < dy:
                        exact_derivative = np.zeros(physical.shape[:2])
                    else:
                        exact_derivative = (
                            factorial(xpower)
                            / factorial(xpower - dx)
                            * factorial(ypower)
                            / factorial(ypower - dy)
                            * physical[:, :, 0] ** (xpower - dx)
                            * physical[:, :, 1] ** (ypower - dy)
                        )
                    derivative = gradient if derivative_order == 1 else hessian
                    evaluated = np.einsum(
                        "tqi,ti->tq", derivative[(..., *directions)], values[dofs]
                    )
                    # Derivatives of high-order cardinal functions cancel at
                    # low-degree polynomials; scale roundoff by the summed
                    # absolute action, rather than by the possibly zero result.
                    action = np.einsum(
                        "tqi,ti->tq", np.abs(derivative[(..., *directions)]), np.abs(values[dofs])
                    )
                    bound = 128 * np.finfo(float).eps * (action + np.abs(exact_derivative) + 1)
                    assert np.all(np.abs(evaluated - exact_derivative) <= bound)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_continuous_numbering_uses_shared_oriented_edge_nodes(degree):
    mesh = TriangleMesh.unit_square(2)
    dofs, nodes = nodal_space(mesh, degree)
    expected = len(mesh.points) + (degree - 1) * len(mesh.faces)
    expected += (degree - 1) * (degree - 2) * len(mesh.cells) // 2
    assert len(nodes) == expected
    assert len(np.unique(dofs)) == expected
    expected_coordinates = np.einsum(
        "qi,tij->tqj", multiindices(degree) / degree, mesh.points[mesh.cells]
    )
    assert_allclose(nodes[dofs], expected_coordinates, atol=2e-16)
    for face in np.flatnonzero(mesh.face_cells[:, 1] >= 0):
        first, second = mesh.face_cells[face]
        assert len(np.intersect1d(dofs[first], dofs[second])) == degree + 1


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_trace_moments_match_independent_physical_edge_integrals(degree):
    mesh = TriangleMesh.unit_square()
    faces = tuple(FaceSpace((0.0, 0.231, 0.68, 1.0), (0, 2, 1)) for _ in mesh.faces)
    skeleton = SkeletonSpace(mesh, faces)
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 3)
        _, nodes = nodal_space(fine, degree)
        matrix = trace_coupling(mesh, cell, fine, skeleton, degree)
        for xp in range(degree + 1):
            yp = degree - xp
            values = nodes[:, 0] ** xp * nodes[:, 1] ** yp
            expected = []
            for side, face in enumerate(mesh.cell_faces[cell]):
                parameter, weights = faces[face].quadrature(8)
                start, end = mesh.points[mesh.faces[face]]
                physical = start + parameter[:, None] * (end - start)
                polynomial = physical[:, 0] ** xp * physical[:, 1] ** yp
                expected.extend(
                    mesh.signs[cell, side]
                    * np.linalg.norm(end - start)
                    * ((weights * polynomial) @ faces[face].evaluate(parameter))
                )
            assert_allclose(values @ matrix, expected, atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize("degree", [0, -1, True, 1.5])
def test_nonpositive_or_noninteger_element_degrees_are_rejected(degree):
    with pytest.raises(ValueError, match="integer"):
        multiindices(degree)
