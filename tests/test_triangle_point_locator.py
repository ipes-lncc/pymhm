"""Geometric ownership validates actual triangles and preserves independent sides."""

from typing import Any

import numpy as np
import pytest

from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.sampling import TrianglePointLocator


def test_actual_affine_coordinates_and_deterministic_interface() -> None:
    """Containment, repeated queries and explicit incident sides use the same maps."""
    mesh = TriangleMesh.unit_square(3)
    locator = TrianglePointLocator(mesh)
    points = mesh.points[mesh.cells].mean(axis=1)
    owners, bary = locator.coordinates(points)
    np.testing.assert_array_equal(owners, np.arange(len(mesh.cells)))
    np.testing.assert_allclose(bary, 1 / 3, atol=1e-12, rtol=1e-10)
    interface = mesh.points[mesh.faces[np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]]].mean(axis=0)[
        None
    ]
    np.testing.assert_array_equal(locator.locate(interface), locator.locate(interface))
    adjacent = mesh.face_cells[np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]]
    explicit, coordinates = locator.coordinates(np.repeat(interface, 2, axis=0), cells=adjacent)
    np.testing.assert_array_equal(explicit, adjacent)
    assert np.all(coordinates >= -locator.tolerance)
    assert locator.locate(np.empty((0, 2))).shape == (0,)


def test_skew_triangle_fallback_rejects_nearest_centroid_assumption() -> None:
    """A narrow long triangle contains a point farther from its centroid than a disjoint cell."""
    mesh = TriangleMesh(
        np.array([[0, 0], [100, 0], [0, 1], [0.1, 1.1], [0.2, 1.1], [0.1, 1.2]]),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    locator = TrianglePointLocator(mesh, candidates=1)
    np.testing.assert_array_equal(locator.locate([[0.01, 0.9]]), [0])
    with pytest.raises(ValueError, match="outside"):
        locator.locate([[0.5, 1.1]])
    with pytest.raises(ValueError, match="incident"):
        locator.coordinates([[0.01, 0.9]], cells=[1])


@pytest.mark.parametrize("points", [[[np.nan, 0]], [[1j, 0]], [[1, 2, 3]], [1, 2]])
def test_invalid_physical_points(points: Any) -> None:
    """Queries contain finite real coordinates with explicit planar shape."""
    with pytest.raises((ValueError, TypeError)):
        TrianglePointLocator(TriangleMesh.unit_square()).locate(points)


@pytest.mark.parametrize("cells", [[-1], [2], [0.5], [True], [0, 1]])
def test_invalid_explicit_ownership(cells: Any) -> None:
    """Independent incident cells require a valid integer per point."""
    with pytest.raises(ValueError, match="valid integer"):
        TrianglePointLocator(TriangleMesh.unit_square()).coordinates([[0.2, 0.1]], cells=cells)


@pytest.mark.parametrize("tolerance", [-1, np.inf, np.nan, 1j])
def test_invalid_tolerance(tolerance: Any) -> None:
    """Dimensionless barycentric tolerances are finite, real and nonnegative."""
    with pytest.raises(ValueError, match="tolerance"):
        TrianglePointLocator(TriangleMesh.unit_square(), tolerance=tolerance)


def test_invalid_mesh_and_candidate_count() -> None:
    """Acceleration parameters cannot silently change the geometric convention."""
    with pytest.raises(TypeError, match="TriangleMesh"):
        TrianglePointLocator(None)
    with pytest.raises(ValueError, match="candidates"):
        TrianglePointLocator(TriangleMesh.unit_square(), candidates=0)
