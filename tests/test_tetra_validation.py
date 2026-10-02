"""Coverage and geometric adversaries for explicit face and volume partitions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.tetra_validation import validate_face_partition, validate_tetra_submesh
from pymhm.tetrahedral import TetraMesh


def face_fan():
    """Triangulate a unit face around a noncentral interior point."""
    corners, center = np.eye(3), np.array([0.2, 0.3, 0.5])
    return np.array([[corners[i], corners[(i + 1) % 3], center] for i in range(3)])


def test_nonuniform_face_partition_preserves_coordinates_and_weights():
    """Area fractions and original local vertex orders are physical metadata."""
    values = face_fan()
    values[1] = values[1, ::-1]
    actual, fractions = validate_face_partition(values)
    assert_array_equal(actual, values)
    assert_allclose(fractions, [0.5, 0.2, 0.3])
    assert_allclose(fractions.sum(), 1)
    assert not actual.flags.writeable and not fractions.flags.writeable
    centroid = np.einsum("t,tia->a", fractions / 3, actual)
    assert_allclose(centroid, np.full(3, 1 / 3))


@pytest.mark.parametrize("shape", [(0, 3, 3), (3, 3), (1, 2, 3)])
def test_face_partition_shape_rejected(shape):
    """Require one or more complete triangles in macroface barycentric coordinates."""
    with pytest.raises(ValueError, match="shape"):
        validate_face_partition(np.zeros(shape))


def test_face_holes_overlap_and_inconsistent_coordinates_rejected():
    """Equal total area does not authorize hanging internal faces or overlapping cells."""
    fan = face_fan()
    with pytest.raises(ValueError, match="sum to one"):
        validate_face_partition(fan + 0.1)
    with pytest.raises(ValueError, match="cover"):
        validate_face_partition(fan[:2])
    fan[0, 2] += [0.05, -0.05, 0]
    assert_allclose(abs(np.linalg.det(fan)).sum(), 1)
    with pytest.raises(ValueError, match="unmatched interior"):
        validate_face_partition(fan)
    with pytest.raises(ValueError, match="finite and real"):
        validate_face_partition(face_fan().astype(complex))


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (1e-7, 0.0), (1.0, 1e6)])
def test_oblique_tetrahedral_fan_covers_macro(scale, shift):
    """Nonnested interior-star tetrahedra retain exact physical envelopes at different scales."""
    vertices = (
        scale * np.array([[0.0, 0, 0], [1.2, 0.1, 0], [0.2, 1.0, 0.1], [0.1, 0.2, 1.4]]) + shift
    )
    coarse = TetraMesh(vertices, [[0, 1, 2, 3]])
    center = np.array([0.2, 0.3, 0.1, 0.4]) @ vertices
    fine = TetraMesh(
        np.vstack((vertices, center)), [[4, *np.delete(np.arange(4), i)] for i in range(4)]
    )
    validate_tetra_submesh(coarse, 0, fine)
    validate_tetra_submesh(coarse, 0, coarse.submesh(0, 2))


def test_volume_envelope_rejects_overlap_even_with_equal_total_volume():
    """Overlapping disconnected tetrahedra cannot pass through volume equality alone."""
    vertices = np.vstack((np.zeros(3), np.eye(3)))
    coarse = TetraMesh(vertices, [[0, 1, 2, 3]])
    overlap = TetraMesh(np.tile(vertices / 2, (8, 1)), np.arange(32).reshape(8, 4))
    assert_allclose(overlap.volumes.sum(), coarse.volumes[0])
    with pytest.raises(ValueError, match="unmatched interior"):
        validate_tetra_submesh(coarse, 0, overlap)
    with pytest.raises(ValueError, match="cover"):
        validate_tetra_submesh(coarse, 0, TetraMesh(vertices + 0.1, [[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="cover"):
        validate_tetra_submesh(coarse, 0, TetraMesh(vertices / 2, [[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="TetraMesh"):
        validate_tetra_submesh(coarse, 0, object())
    for cell in [-1, 1]:
        with pytest.raises(ValueError, match="cell"):
            validate_tetra_submesh(coarse, cell, coarse)
