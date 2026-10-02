"""Independent moments and geometry checks for crossed/diagonal P1 overlays."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.mh2m_crisscross_norms import PATTERN, CrossedP1, common_triangles
from examples.mh2m_heterogeneous_norms import StructuredP1, difference
from pymhm.mesh import TriangleMesh


def crossed(n, function):
    """Build reversed archived triangles independently of the canonical reader."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vertices = ((np.column_stack((i.ravel(), j.ravel()))[:, None, None] + PATTERN) / n).reshape(
        -1, 3, 2
    )
    pressure = function(vertices.reshape(-1, 2)).reshape(-1, 3)
    return CrossedP1.from_arrays(vertices[::-1, ::-1], pressure[::-1, ::-1])


@pytest.mark.parametrize("resolution", [2, 3, 4])
def test_nonzero_affine_fields_on_distinct_partitions(resolution):
    """Check pressure, gradient, flux and energy against elementary unit-square moments."""
    mesh = TriangleMesh.unit_square(3)
    vertices = mesh.points[mesh.cells]
    reference = StructuredP1.from_arrays(vertices, 1 + vertices @ [2, 3])
    other = crossed(resolution, lambda x: 1 + x @ [1, -1])
    result = difference(reference, other, 2.0, 5, geometry=common_triangles(3, resolution))
    assert_allclose(result["pressure_difference"], np.sqrt(23 / 3), rtol=2e-14)
    assert_allclose(result["reference_pressure_norm"], np.sqrt(40 / 3), rtol=2e-14)
    assert_allclose(result["flux_difference"], 2 * np.sqrt(17), rtol=2e-14)
    assert_allclose(result["energy_difference"], np.sqrt(34), rtol=2e-14)
    assert_allclose(result["overlay_area"], 1, atol=2e-15)


def test_discontinuous_quadrants_are_integrated_without_interpolation():
    """Each independent triangle value contributes its own exact quarter-square mass."""
    reference = StructuredP1(np.zeros((3, 3, 2, 3)))
    other = CrossedP1(np.broadcast_to(np.arange(1, 5)[None, None, :, None], (2, 2, 4, 3)))
    result = difference(reference, other, 1.0, 4, geometry=common_triangles(3, 2))
    assert_allclose(result["pressure_difference"], np.sqrt(7.5), rtol=2e-14)
    assert result["flux_difference"] == 0
    assert result["pressure_relative_difference"] is None


def test_incident_rows_and_incompatible_archive_rejected():
    """Retain distinct horizontal traces and reject duplicated or missing triangles."""
    values = np.ones((2, 2, 4, 3))
    values[:, 1] = 3
    field = CrossedP1(values)
    points = np.array([[0.2, 0.5], [0.7, 0.5]])
    assert_allclose(field.evaluate(points, y_side=-1)[0], 1)
    assert_allclose(field.evaluate(points, y_side=1)[0], 3)
    with pytest.raises(ValueError, match="incident"):
        field.evaluate(points, y_side=0)
    with pytest.raises(ValueError, match="complete"):
        CrossedP1.from_arrays(np.zeros((3, 3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="unique"):
        CrossedP1.from_arrays(np.zeros((4, 3, 2)), np.zeros((4, 3)))


def test_spawned_nonzero_norms_equal_serial():
    """Spawn preserves the same physical points, batches and ordered reductions."""
    reference = StructuredP1(np.ones((3, 3, 2, 3)))
    other = crossed(2, lambda x: 1 + x @ [2, 3])
    serial = difference(reference, other, 2.0, 6, geometry=common_triangles(3, 2))
    parallel = difference(reference, other, 2.0, 6, workers=2, geometry=common_triangles(3, 2))
    assert parallel == serial
