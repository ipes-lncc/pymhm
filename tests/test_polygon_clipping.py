"""Cartesian half-plane cuts preserve oriented measure and represented boundaries."""

from typing import Any

import numpy as np
import pytest

from pymhm.fem.quadrature.material import _clip_polygon
from pymhm.meshes.geometry import clip_polygon
from pymhm.meshes.triangle import TriangleMesh


def signed_area(vertices: np.ndarray) -> float:
    """Integrate oriented area by the polygon boundary's exact affine segments."""
    next_vertices = np.roll(vertices, -1, axis=0)
    return float(
        np.sum(vertices[:, 0] * next_vertices[:, 1] - vertices[:, 1] * next_vertices[:, 0]) / 2
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("clockwise", [False, True])
def test_half_plane_partition_preserves_oriented_area_and_input(axis: int, clockwise: bool) -> None:
    """Opposite closed cuts partition a nontrivial triangle without changing its orientation."""
    triangle = np.array([[-0.25, 0.125], [1.5, 0.5], [0.25, 1.75]])
    if clockwise:
        triangle = triangle[::-1]
    original = triangle.copy()
    high = clip_polygon(triangle, axis, 0.25, True)
    low = clip_polygon(triangle, axis, 0.25, False)
    assert np.all(high[:, axis] >= 0.25)
    assert np.all(low[:, axis] <= 0.25)
    assert np.any(high[:, axis] == 0.25) and np.any(low[:, axis] == 0.25)
    assert signed_area(high) * signed_area(triangle) > 0
    assert signed_area(low) * signed_area(triangle) > 0
    np.testing.assert_allclose(
        signed_area(high) + signed_area(low), signed_area(triangle), atol=1e-12, rtol=1e-10
    )
    np.testing.assert_array_equal(triangle, original)
    np.testing.assert_array_equal(_clip_polygon(triangle, axis, 0.25, True), high)


def test_empty_degenerate_and_strict_represented_boundary_conventions() -> None:
    """Do not merge a represented narrow cut, and retain touching zero-area intersections."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    smallest_positive = float(np.nextafter(0.0, 1.0))
    cut = clip_polygon(square, 0, smallest_positive, True)
    assert np.min(cut[:, 0]) == smallest_positive
    assert not np.any(cut[:, 0] == 0)
    touching = clip_polygon(square, 0, 1.0, True)
    assert len(touching) > 0 and signed_area(touching) == 0
    assert np.all(touching[:, 0] == 1)
    empty = clip_polygon(square, 1, 2.0, True)
    assert empty.shape == (0, 2)
    assert clip_polygon(empty, 0, 0.5, False).shape == (0, 2)
    unchanged = clip_polygon(square.tolist(), 0, 0.0, True)
    np.testing.assert_array_equal(unchanged, square)
    assert not np.shares_memory(unchanged, square)


def test_scientific_corner_cut_quadrature_preserves_analytic_area_and_first_moments() -> None:
    """The shared public cuts integrate two corner exclusions without transferring fields."""
    from examples.compare_stokes_cavity import interior_quadrature

    mesh = TriangleMesh.unit_square(3)
    cutout = 0.13
    barycentric, weights = interior_quadrature(mesh, 4, cutout)
    integrals = np.zeros(3)
    for vertices, area, local_barycentric, local_weights in zip(
        mesh.points[mesh.cells], mesh.areas, barycentric, weights, strict=True
    ):
        points = local_barycentric @ vertices
        integrals += area * (local_weights @ np.c_[np.ones(len(points)), points])
    expected_area = 1 - 2 * cutout**2
    expected_y = (1 - cutout) ** 2 / 2 + cutout * (1 - 2 * cutout) * (1 - cutout / 2)
    np.testing.assert_allclose(
        integrals, [expected_area, expected_area / 2, expected_y], atol=1e-12, rtol=1e-10
    )


@pytest.mark.parametrize(
    "polygon,axis,level,lower",
    [
        (np.zeros(3), 0, 0.0, True),
        (np.zeros((3, 3)), 0, 0.0, True),
        (np.full((3, 2), np.nan), 0, 0.0, True),
        (np.zeros((3, 2), complex), 0, 0.0, True),
        (np.zeros((3, 2)), -1, 0.0, True),
        (np.zeros((3, 2)), 2, 0.0, True),
        (np.zeros((3, 2)), True, 0.0, True),
        (np.zeros((3, 2)), 0.5, 0.0, True),
        (np.zeros((3, 2)), 0, np.inf, True),
        (np.zeros((3, 2)), 0, 1j, True),
        (np.zeros((3, 2)), 0, [0.0], True),
        (np.zeros((3, 2)), 0, 0.0, 1),
    ],
)
def test_ambiguous_clipping_inputs_are_rejected(
    polygon: Any, axis: Any, level: Any, lower: Any
) -> None:
    """Reject invalid dimensions, nonreal coordinates and ambiguous half-plane choices."""
    with pytest.raises(ValueError):
        clip_polygon(polygon, axis, level, lower)
