"""Physical candidate coordinates remain independent of native basis numbering."""

from itertools import product
from math import factorial, prod

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.bdm import validate_bdm2_trace
from pymhm.hdiv3d_family import face_polynomials, face_size
from pymhm.hdiv_reference import bernstein_tabulation
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.rt3d import _candidates


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_declared_bernstein_coordinates(dimension: int, degree: int) -> None:
    """The archived exponent order has its stated values and Cartesian gradients."""
    points = np.vstack((np.full(dimension, 0.1), np.zeros(dimension), np.eye(dimension)))
    bary = np.column_stack((1 - points.sum(axis=1), points))
    exponents = tuple(
        exponent
        for exponent in product(range(degree + 1), repeat=dimension + 1)
        if sum(exponent) == degree
    )
    expected = np.zeros((len(points), len(exponents)))
    gradients = np.zeros((*expected.shape, dimension))
    for column, exponent in enumerate(exponents):
        multiplier = factorial(degree) / prod(factorial(a) for a in exponent)
        expected[:, column] = multiplier * np.prod(bary**exponent, axis=1)
        for axis in range(dimension):
            for coordinate, sign in ((0, -1), (axis + 1, 1)):
                if exponent[coordinate]:
                    lowered = np.array(exponent)
                    lowered[coordinate] -= 1
                    gradients[:, column, axis] += (
                        sign * exponent[coordinate] * multiplier * np.prod(bary**lowered, axis=1)
                    )
    values, derivatives = bernstein_tabulation(points, exponents)
    assert_allclose(values, expected, rtol=0, atol=3e-14)
    assert_allclose(derivatives, gradients, rtol=0, atol=3e-14)


def test_archived_rt1_candidate_rows_have_their_declared_meaning() -> None:
    """RT archive rows remain barycentric component functions and ordered radial monomials."""
    points = np.array([[0.2, 0.3, 0.1], [0.0, 0.0, 1.0], [-0.1, 0.4, 0.2]])
    scalar = np.column_stack((points[:, 2], points[:, 1], points[:, 0], 1 - points.sum(axis=1)))
    derivatives = np.array([[0, 0, 1, -1], [0, 1, 0, -1], [1, 0, 0, -1]])
    expected = np.concatenate([scalar[..., None] * np.eye(3)[axis] for axis in range(3)], axis=1)
    expected_divergence = np.concatenate(
        [np.broadcast_to(derivatives[axis], scalar.shape) for axis in range(3)], axis=1
    )
    radial = points[:, ::-1]
    expected = np.concatenate((expected, points[:, None] * radial[..., None]), axis=1)
    expected_divergence = np.column_stack((expected_divergence, 4 * radial))
    values, divergence = _candidates(points, 1)
    assert_allclose(values, expected, rtol=0, atol=2e-14)
    assert_allclose(divergence, expected_divergence, rtol=0, atol=2e-14)


def test_quadratic_normal_trace_requires_representable_degree_and_edge_partition() -> None:
    """BDM2 accepts its degree boundary and rejects degree/partition mismatches."""
    mesh = TriangleMesh.unit_square()
    valid = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces))
    validate_bdm2_trace(valid, 2)
    for face in (FaceSpace.uniform(3, 2), FaceSpace.uniform(2, 3)):
        invalid = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
        with pytest.raises(ValueError, match="degrees <=2.*aligned"):
            validate_bdm2_trace(invalid, 2)


@pytest.mark.parametrize("corners", [2, 5])
def test_normal_moment_faces_require_supported_triangle_or_quadrilateral(corners: int) -> None:
    """Moment dimensions and evaluation reject the same unsupported face topology."""
    with pytest.raises(ValueError, match="three or four corners"):
        face_size(corners, 1)
    with pytest.raises(ValueError, match="three or four corners"):
        face_polynomials(np.array([[0.2, 0.3]]), corners, 1)
