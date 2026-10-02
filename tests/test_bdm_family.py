"""Physical normal moments, divergence ranges and independently enriched bubbles."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.bdm import bdm2_basis, bdm2_dofs, bdm2_trace_map
from pymhm.bdm_family import BDMFamily
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import reference_basis


@pytest.mark.parametrize(
    "degree,enrichment", [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
)
def test_face_duality_bubble_traces_and_full_divergence_range(degree: int, enrichment: int) -> None:
    """Check physical moments on a sheared triangle and the exact divergence image."""
    family = BDMFamily(degree, enrichment)
    mesh = TriangleMesh([[0.2, -0.1], [1.7, 0.3], [0.4, 1.2]], [[0, 1, 2]])
    x, weights = leggauss(family.polynomial_degree + 3)
    for side in range(3):
        bary = np.zeros((len(x), 3))
        bary[:, side] = (1 - x) / 2
        bary[:, (side + 1) % 3] = (1 + x) / 2
        basis, _ = family.basis(mesh, bary)
        face = mesh.cell_faces[0, side]
        parameter = x * mesh.signs[0, side]
        moment = legvander(parameter, family.polynomial_degree).T @ (
            (weights * mesh.lengths[face] / 2)[:, None] * (basis[0] @ mesh.normals[face])
        )
        expected = np.zeros_like(moment)
        expected[: degree + 1, side * (degree + 1) : (side + 1) * (degree + 1)] = np.eye(degree + 1)
        assert_allclose(moment, expected, atol=2e-10)
    bary, weights = triangle_quadrature(family.polynomial_degree + 3)
    _, divergence = family.basis(mesh, bary)
    pressure_degree = family.polynomial_degree - 1
    scalar = (
        reference_basis(pressure_degree, bary)[0] if pressure_degree else np.ones((len(bary), 1))
    )
    coupling = scalar.T @ (weights[:, None] * divergence[0])
    assert np.linalg.matrix_rank(coupling, tol=1e-9) == scalar.shape[1]
    assert family.size(mesh) == family.local_size
    assert family.dofs(mesh).shape == (1, family.local_size)


def test_bdm2_existing_basis_and_segmented_trace_are_preserved() -> None:
    """The generalized family retains every original BDM2 coefficient convention."""
    coarse = TriangleMesh.unit_square()
    mesh = coarse.submesh(0, 3)
    family = BDMFamily()
    bary = triangle_quadrature(5)[0]
    assert_allclose(family.basis(mesh, bary)[0], bdm2_basis(mesh, bary)[0], atol=1e-12)
    assert_allclose(family.basis(mesh, bary)[1], bdm2_basis(mesh, bary)[1], atol=1e-12)
    assert_allclose(family.dofs(mesh), bdm2_dofs(mesh))
    skeleton = SkeletonSpace(coarse, tuple(FaceSpace.uniform(2, 3) for _ in coarse.faces))
    family.validate_trace(skeleton, 3)
    assert_allclose(
        family.trace_map(coarse, 0, mesh, skeleton), bdm2_trace_map(coarse, 0, mesh, skeleton)
    )
    coefficients = np.arange(family.size(mesh), dtype=float)
    value, divergence = family.evaluate(mesh, coefficients, bary)
    tensor, tensor_div = family.evaluate(
        mesh, np.column_stack((coefficients, 2 * coefficients)), bary
    )
    assert_allclose(tensor[..., 0, :], value)
    assert_allclose(tensor_div[..., 1], 2 * divergence)


@pytest.mark.parametrize("degree,enrichment", [(0, 0), (1.5, 0), (1, -1), (1, 3.5)])
def test_invalid_family(degree: int, enrichment: int) -> None:
    """Reject noninteger or unsupported polynomial family descriptors."""
    with pytest.raises(ValueError):
        BDMFamily(degree, enrichment)


@pytest.mark.parametrize(
    "bary", [np.ones((1, 3)) * 1j, np.ones((1, 2)), [[np.nan, 0, 1]], [[0, 0, 0]]]
)
def test_invalid_barycentric_points(bary: np.ndarray) -> None:
    """Reference coordinates must remain finite, real and normalized."""
    with pytest.raises(ValueError, match="barycentric"):
        BDMFamily().basis(TriangleMesh.unit_square(), bary)


@pytest.mark.parametrize(
    "coefficients", [np.zeros(3), np.zeros((21, 1, 1)), np.full(21, np.nan), np.ones(21) * 1j]
)
def test_invalid_coefficients(coefficients: np.ndarray) -> None:
    """A physical field requires the declared real global coefficient layout."""
    with pytest.raises(ValueError, match="coefficients"):
        BDMFamily().evaluate(TriangleMesh.unit_square(), coefficients, np.array([[1.0, 0, 0]]))


@pytest.mark.parametrize("face", [FaceSpace.uniform(3), FaceSpace.uniform(1, 3)])
def test_unrepresentable_macro_trace(face: FaceSpace) -> None:
    """Reject both polynomial and geometric mismatch of a prescribed normal trace."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="trace degrees"):
        BDMFamily().validate_trace(SkeletonSpace(mesh, tuple(face for _ in mesh.faces)), 2)


def test_cellwise_quadrature_preserves_piola_values() -> None:
    """Separate material-cell integration points do not average one-sided bases."""
    mesh = TriangleMesh.unit_square()
    family = BDMFamily(1, 2)
    bary = np.array([[[0.2, 0.3, 0.5]], [[0.1, 0.7, 0.2]]])
    values, divergence = family.basis(mesh, bary)
    for cell in range(2):
        common = family.basis(mesh, bary[cell])
        assert_allclose(values[cell], common[0][cell])
        assert_allclose(divergence[cell], common[1][cell])
    with pytest.raises(ValueError, match="barycentric"):
        family.basis(mesh, bary[:1])
