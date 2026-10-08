"""Public geometry and residual operations retain their numerical conventions."""

import numpy as np
import pytest
from scipy import sparse

from pymhm import accurate_residual, hexahedral_mapping, require_hermitian, symmetric_equilibration
from pymhm.linalg import linear
from pymhm.meshes.hexahedron import HexMesh


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_rectangular_residual_compensates_cancellation(
    extended: bool, complex_values: bool
) -> None:
    """Real and complex multiple RHS retain small terms in a rectangular row."""
    matrix = sparse.csr_matrix([[1e16, 1.0, -1e16]])
    if complex_values:
        matrix = matrix * (1 + 1j)
    solution = np.column_stack((np.ones(3), 2 * np.ones(3)))
    rhs = np.array([[2.0, 4.0]]) * (1 + 1j if complex_values else 1)
    with pytest.MonkeyPatch.context() as patch:
        # Extended accumulation is selected only on a platform supplying it.
        patch.setattr(linear, "_EXTENDED_PRECISION", extended and linear._EXTENDED_PRECISION)
        actual = accurate_residual(matrix, rhs, solution)
        single = accurate_residual(matrix, rhs[:, 0], solution[:, 0])
    np.testing.assert_array_equal(actual, rhs / 2)
    np.testing.assert_array_equal(single, rhs[:, 0] / 2)
    assert linear._accurate_residual is accurate_residual


def test_hexahedral_map_declares_physical_jacobian_axes() -> None:
    """An affine map and its extension use physical rows and reference columns."""
    mesh = HexMesh.unit_cube()
    transform = np.array([[2.0, 0.4, 0.0], [0.0, 3.0, 0.2], [0.0, 0.0, 4.0]])
    offset = np.array([1.0, -2.0, 3.0])
    vertices = mesh.points[mesh.cells] @ transform.T + offset
    points = np.array([[0.2, 0.4, 0.6], [-0.2, 1.1, 0.3]])
    physical, jacobian, determinant = hexahedral_mapping(vertices, points)
    np.testing.assert_allclose(physical[0], points @ transform.T + offset, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(jacobian, np.broadcast_to(transform, jacobian.shape), atol=1e-12)
    np.testing.assert_allclose(determinant, 24.0, atol=1e-12)


def test_symmetric_scaling_preserves_indefinite_equations() -> None:
    """Congruence scales a Hermitian operator without imposing positive definiteness."""
    matrix = sparse.diags([-2e-8, 8e8], format="csr")
    require_hermitian(matrix)
    balanced, diagonal = symmetric_equilibration(matrix)
    scale = sparse.diags(diagonal)
    np.testing.assert_allclose(balanced.toarray(), (scale @ matrix @ scale).toarray(), atol=1e-12)
    assert np.all(diagonal > 0)
    assert balanced[0, 0] < 0 < balanced[1, 1]
    rhs = matrix @ np.array([-2.0, 3.0])
    solution = diagonal * np.linalg.solve(balanced.toarray(), diagonal * rhs)
    np.testing.assert_allclose(solution, [-2.0, 3.0], atol=1e-12, rtol=1e-10)
    assert linear._symmetric_equilibration is symmetric_equilibration
    assert linear._hermitian is require_hermitian
    with pytest.raises(ValueError, match="operation requires"):
        require_hermitian(sparse.csr_matrix([[1.0, 0.1], [0.0, 1.0]]))


@pytest.mark.parametrize(
    "vertices",
    [np.zeros((8, 3)), np.zeros((1, 7, 3)), np.full((1, 8, 3), np.nan), np.ones((1, 8, 3)) * 1j],
)
def test_hexahedral_map_rejects_invalid_vertices(vertices: np.ndarray) -> None:
    """Malformed or nonphysical vertex data are rejected before map evaluation."""
    with pytest.raises(ValueError, match="vertices"):
        hexahedral_mapping(vertices, np.zeros((1, 3)))


@pytest.mark.parametrize("points", [np.zeros(3), np.zeros((1, 2)), [[np.inf, 0, 0]], [[1j, 0, 0]]])
def test_hexahedral_map_rejects_invalid_reference_coordinates(points: np.ndarray) -> None:
    """Reference coordinates must be finite real triples for every sample."""
    mesh = HexMesh.unit_cube()
    with pytest.raises(ValueError, match="reference coordinates"):
        hexahedral_mapping(mesh.points[mesh.cells], points)
