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


def test_exterior_mask_preserves_order_and_strict_default() -> None:
    """Subdomain sampling marks exterior points without assigning a nearest cell."""
    locator = TrianglePointLocator(TriangleMesh.unit_square(), candidates=1)
    points = np.array([[0.7, 0.1], [2, 2], [0.1, 0.7], [-1, -1]])
    owners = locator.locate(points, allow_outside=True)
    np.testing.assert_array_equal(owners[[1, 3]], [-1, -1])
    np.testing.assert_array_equal(owners[[0, 2]], locator.locate(points[[0, 2]]))
    assert locator.locate(np.empty((0, 2)), allow_outside=True).shape == (0,)
    with pytest.raises(ValueError, match="outside"):
        locator.locate(points)
    with pytest.raises(ValueError, match="valid integer"):
        locator.coordinates(points, cells=owners)
    with pytest.raises(TypeError, match="boolean"):
        locator.locate(points, allow_outside=1)


def test_exterior_broadphase_respects_declared_barycentric_tolerance() -> None:
    """A valid tolerant fallback can extend beyond the physical vertex bounding box."""
    mesh = TriangleMesh(
        np.array([[0, 0], [100, 0], [100, 1], [0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    points = np.array([[-0.19, -0.00095], [-0.3, -0.003]])
    locator = TrianglePointLocator(mesh, candidates=1, tolerance=0.001)
    np.testing.assert_array_equal(locator.locate(points, allow_outside=True), [0, -1])
    strict = TrianglePointLocator(mesh, candidates=1, tolerance=0)
    np.testing.assert_array_equal(strict.locate(points, allow_outside=True), [-1, -1])
    owners, bary = locator.coordinates(points[:1])
    np.testing.assert_array_equal(owners, [0])
    np.testing.assert_allclose(bary, [[1.0019, -0.00095, -0.00095]], atol=1e-12, rtol=1e-10)


def test_strictly_exterior_queries_do_not_trigger_exhaustive_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unresolved exterior samples are masked without visiting all mesh cells."""
    locator = TrianglePointLocator(TriangleMesh.unit_square(3), candidates=1)
    original = locator._contains
    visited = []

    def observe(points: Any, candidates: Any) -> Any:
        """Record the number of geometric candidates actually evaluated."""
        visited.append(candidates.shape)
        return original(points, candidates)

    monkeypatch.setattr(locator, "_contains", observe)
    points = np.array([[2, 2], [-1, -1], [0.3, 2]])
    np.testing.assert_array_equal(locator.locate(points, allow_outside=True), [-1, -1, -1])
    assert visited == []


def test_exhaustive_batches_preserve_skew_owners_and_query_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback retains the first valid incident cell and bounds its temporary storage."""
    mesh = TriangleMesh(
        np.array([[0, 0], [100, 0], [0, 1], [0.1, 1.1], [0.2, 1.1], [0.1, 1.2]]),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    locator = TrianglePointLocator(mesh, candidates=1)
    original = locator._contains
    batches = []

    def observe(points: Any, candidates: Any) -> Any:
        """Track exhaustive batches independently of nearest-candidate queries."""
        if candidates.shape[1] == len(mesh.cells):
            batches.append(len(points))
        return original(points, candidates)

    monkeypatch.setattr(locator, "_contains", observe)
    queries = np.tile([[0.01, 0.9], [0.5, 1.1], [0.12, 1.12], [0, 0]], (300, 1))
    np.testing.assert_array_equal(
        locator.locate(queries, allow_outside=True), np.tile([0, -1, 1, 0], 300)
    )
    assert len(batches) > 1
    assert max(batches) <= 256


@pytest.mark.parametrize("scale", [1.0, 1e-160, 1e150])
def test_zero_tolerance_handles_normal_and_subnormal_determinants(scale: float) -> None:
    """Underflow in physical orientation products cannot admit a representable exterior."""
    mesh = TriangleMesh(scale * np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 1, 2]]))
    locator = TrianglePointLocator(mesh, tolerance=0)
    points = np.vstack(
        (
            mesh.points,
            [scale / 2, scale / 2],
            [scale / 2, np.nextafter(scale / 2, np.inf)],
            [scale / 4, np.nextafter(0.0, -np.inf)],
        )
    )
    np.testing.assert_array_equal(locator.locate(points, allow_outside=True), [0, 0, 0, 0, -1, -1])
    owners, bary = locator.coordinates(points[:4], cells=[0, 0, 0, 0])
    np.testing.assert_array_equal(owners, [0, 0, 0, 0])
    np.testing.assert_allclose(bary[-1], [0, 0.5, 0.5], atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="incident"):
        locator.coordinates(points[4:], cells=[0, 0])


def test_large_declared_tolerance_resolves_overflow_exactly() -> None:
    """Finite coordinates and tolerance retain their exact meaning beyond binary64 products."""
    mesh = TriangleMesh(1e150 * np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 1, 2]]))
    locator = TrianglePointLocator(mesh, tolerance=1e200)
    points = np.array([[-1e308, 1e308], [1e308, -1e308]])
    np.testing.assert_array_equal(locator.locate(points, allow_outside=True), [0, 0])
    owners, bary = locator.coordinates(points, cells=[0, 0])
    np.testing.assert_array_equal(owners, [0, 0])
    assert np.isfinite(bary).all()


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
