"""Native element providers through real affine physical tabulation callers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.fem.scalar.tetrahedron import tetra_element_tabulate, tetra_tabulate
from pymhm.fem.scalar.triangle import element_tabulate, tabulate
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", range(1, 7))
def test_triangle_public_native_tabulation_preserves_topology_and_physical_derivatives(
    degree: int,
) -> None:
    pytest.importorskip("basix")
    mesh = TriangleMesh(
        np.array([[0.1, 0.2], [1.4, 0.3], [0.3, 1.2], [1.5, 1.5]]),
        np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )
    bary = np.vstack((np.eye(3), np.random.default_rng(2026).dirichlet(np.ones(3), size=11)))
    expected = tabulate(mesh, degree, bary)
    actual = tabulate(mesh, degree, bary, backend="basix")
    for a, b in zip(actual[:2], expected[:2], strict=True):
        assert_array_equal(a, b)
    for a, b in zip(actual[2:], expected[2:], strict=True):
        assert_allclose(a, b, atol=3e-11, rtol=8e-12)
    for points in (
        np.broadcast_to(bary, (len(mesh.cells), *bary.shape)),
        np.stack((bary, bary[:, [1, 2, 0]])),
    ):
        expected = element_tabulate(mesh, degree, points)
        actual = element_tabulate(mesh, degree, points, backend="basix")
        for a, b in zip(actual[:2], expected[:2], strict=True):
            assert_array_equal(a, b)
        for a, b in zip(actual[2:], expected[2:], strict=True):
            assert_allclose(a, b, atol=3e-11, rtol=8e-12)


@pytest.mark.parametrize("degree", range(1, 7))
def test_tetra_public_native_tabulation_preserves_topology_and_physical_derivatives(
    degree: int,
) -> None:
    pytest.importorskip("basix")
    mesh = TetraMesh(
        [[0.1, 0.2, -0.1], [1.2, 0.3, 0.0], [0.3, 1.1, 0.2], [-0.1, 0.4, 1.0]],
        [[0, 1, 2, 3]],
    )
    bary = np.vstack((np.eye(4), np.random.default_rng(2026).dirichlet(np.ones(4), size=11)))
    expected = tetra_element_tabulate(mesh, degree, bary)
    actual = tetra_element_tabulate(mesh, degree, bary, backend="basix")
    for a, b in zip(actual[:2], expected[:2], strict=True):
        assert_array_equal(a, b)
    for a, b in zip(actual[2:], expected[2:], strict=True):
        assert_allclose(a, b, atol=3e-11, rtol=8e-12)
    without_hessian = tetra_tabulate(mesh, degree, bary, backend="basix")
    for a, b in zip(without_hessian, actual[:4], strict=True):
        assert_allclose(a, b, atol=3e-11, rtol=8e-12)


def test_native_physical_hessian_reproduces_independent_quadratic_field() -> None:
    pytest.importorskip("basix")
    mesh = TetraMesh(
        [[0.1, 0.2, -0.1], [1.2, 0.3, 0.0], [0.3, 1.1, 0.2], [-0.1, 0.4, 1.0]],
        [[0, 1, 2, 3]],
    )
    bary = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    dofs, nodes, values, gradients, hessians = tetra_element_tabulate(
        mesh, 2, bary, backend="basix"
    )
    coefficients = nodes[:, 0] ** 2 + 2 * nodes[:, 0] * nodes[:, 1] + 3 * nodes[:, 2] ** 2
    physical = bary @ mesh.points[mesh.cells[0]]
    expected_values = (
        physical[:, 0] ** 2 + 2 * physical[:, 0] * physical[:, 1] + 3 * physical[:, 2] ** 2
    )
    expected_gradient = np.column_stack(
        (2 * physical[:, 0] + 2 * physical[:, 1], 2 * physical[:, 0], 6 * physical[:, 2])
    )
    expected_hessian = np.array([[2.0, 2.0, 0], [2.0, 0, 0], [0, 0, 6.0]])
    assert_allclose(values @ coefficients[dofs[0]], expected_values, atol=2e-14)
    assert_allclose(
        np.einsum("qna,n->qa", gradients[0], coefficients[dofs[0]]), expected_gradient, atol=4e-14
    )
    assert_allclose(
        np.einsum("qnab,n->qab", hessians[0], coefficients[dofs[0]]),
        np.broadcast_to(expected_hessian, (len(bary), 3, 3)),
        atol=8e-14,
    )


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
def test_public_physical_tabulation_rejects_unknown_backend(cell: str) -> None:
    backend: Any = "unknown"
    if cell == "triangle":
        triangle = TriangleMesh.unit_square()
        with pytest.raises(ValueError, match="backend"):
            tabulate(triangle, 1, np.eye(3), backend=backend)
        with pytest.raises(ValueError, match="backend"):
            element_tabulate(triangle, 1, np.broadcast_to(np.eye(3), (2, 3, 3)), backend=backend)
    else:
        tetrahedron = TetraMesh.unit_cube()
        with pytest.raises(ValueError, match="backend"):
            tetra_tabulate(tetrahedron, 1, np.eye(4), backend=backend)
        with pytest.raises(ValueError, match="backend"):
            tetra_element_tabulate(tetrahedron, 1, np.eye(4), backend=backend)
