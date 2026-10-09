"""Compiled point filters preserve exact geometry and literal affine arithmetic."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing import sampling
from pymhm.postprocessing._sampling_kernels import candidate_barycentric, containment_filter


def exact_membership(vertices: np.ndarray, points: np.ndarray, tolerance: float) -> np.ndarray:
    """Resolve supplied binary coordinates with independent rational determinants."""
    corners = tuple(tuple(Fraction(float(value)) for value in vertex) for vertex in vertices)

    def determinant(first: Any, second: Any, third: Any) -> Fraction:
        """Use the represented input values without an affine inverse or tolerance expansion."""
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )

    area = determinant(*corners)
    sign = 1 if area > 0 else -1
    threshold = Fraction(tolerance) * abs(area)
    output = []
    for point in points:
        exact = tuple(Fraction(float(value)) for value in point)
        output.append(
            all(
                sign * determinant(corners[edge], corners[(edge + 1) % 3], exact) + threshold >= 0
                for edge in range(3)
            )
        )
    return np.asarray(output)


@pytest.mark.parametrize("engine", ["compiled", "interpreted"])
@pytest.mark.parametrize(
    "case",
    ["regular", "clockwise", "skew", "subnormal", "threshold_underflow", "overflow", "tolerant"],
)
def test_filtered_membership_matches_exact_binary_geometry(
    engine: str, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interior, vertices and immediately excluded floats retain their physical membership."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tolerance = 2.0**-12 if case == "tolerant" else 0.0
    points = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.5, np.nextafter(0.5, np.inf)],
            [0.25, np.nextafter(0.0, -np.inf)],
            [-tolerance, 0.25],
            [np.nextafter(-tolerance, -np.inf), 0.25],
            [1.25, 0.25],
        ]
    )
    if case in {"subnormal", "threshold_underflow"}:
        vertices *= 1e-160
        points *= 1e-160
        points[-2] = [1e-160 / 4, np.nextafter(0.0, -np.inf)]
        if case == "threshold_underflow":
            tolerance = 1e-10
    elif case == "overflow":
        vertices *= 1e150
        points *= 1e150
        points[-2:] = [[-1e308, 1e308], [1e308, -1e308]]
        tolerance = 1e200
    elif case == "skew":
        transform = np.array([[1.0, 0.25], [0.0, 0.5]])
        vertices = vertices @ transform
        points = points @ transform
    elif case == "clockwise":
        vertices = vertices[[0, 2, 1]]
    mesh = TriangleMesh(vertices, np.array([[0, 1, 2]]))
    locator = sampling.TrianglePointLocator(mesh, tolerance=tolerance)
    if engine == "interpreted":
        monkeypatch.setattr(sampling, "containment_filter", containment_filter.py_func)
    owners = np.zeros((len(points), 1), dtype=np.int64)
    expected = exact_membership(vertices, points, tolerance)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        actual = locator._contains(points, owners)[:, 0]
        areas, errors = sampling._orientation_filter(vertices[1], vertices[2], vertices[0])
        function = containment_filter.py_func if engine == "interpreted" else containment_filter
        filtered, pending = function(
            points, owners, vertices[None], np.atleast_1d(areas), np.atleast_1d(errors), tolerance
        )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(filtered[~pending], expected[:, None][~pending])
    assert np.any(pending)


@pytest.mark.parametrize("engine", ["compiled", "interpreted"])
def test_candidate_maps_preserve_literal_bits_and_independent_sides(engine: str) -> None:
    """Read-only affine maps and strided candidates preserve the original dot-product order."""
    mesh = TriangleMesh.unit_square(3)
    mesh = TriangleMesh(mesh.points @ np.array([[1.0, 0.3], [-0.2, 1.1]]), mesh.cells)
    locator = sampling.TrianglePointLocator(mesh, tolerance=0)
    points = np.vstack((mesh.points, mesh.points[mesh.cells].mean(axis=1)))
    candidates = np.broadcast_to(np.arange(len(mesh.cells)), (len(points), len(mesh.cells)))[:, ::2]
    local = np.einsum(
        "qkba,qka->qkb", locator.inverse[candidates], points[:, None] - locator.origins[candidates]
    )
    literal = np.concatenate(((1 - local.sum(axis=2))[..., None], local), axis=2)
    function = candidate_barycentric.py_func if engine == "interpreted" else candidate_barycentric
    actual = function(points, candidates, locator.origins, locator.inverse)
    np.testing.assert_array_equal(actual.view(np.uint8), literal.view(np.uint8))
    assert function(points[:0], candidates[:0], locator.origins, locator.inverse).shape == (
        0,
        candidates.shape[1],
        3,
    )
    assert function(points, candidates[:, :0], locator.origins, locator.inverse).shape == (
        len(points),
        0,
        3,
    )


@pytest.mark.parametrize("engine", ["compiled", "interpreted"])
def test_ambiguous_area_is_deferred_instead_of_deciding_from_overflow(engine: str) -> None:
    """A finite-coordinate determinant beyond binary64 range requires exact resolution."""
    vertices = 1e200 * np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    points = np.array([[1e199, 1e199], [1e200, 1e200]])
    candidates = np.zeros((len(points), 1), dtype=np.int64)
    area, errors = sampling._orientation_filter(vertices[:, 1], vertices[:, 2], vertices[:, 0])
    function = containment_filter.py_func if engine == "interpreted" else containment_filter
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        _, pending = function(points, candidates, vertices, area, errors, 0.0)
    np.testing.assert_array_equal(pending, True)
    np.testing.assert_array_equal(exact_membership(vertices[0], points, 0.0), [True, False])
    assert function(points[:0], candidates[:0], vertices, area, errors, 0.0)[0].shape == (0, 1)
    assert function(points, candidates[:, :0], vertices, area, errors, 0.0)[0].shape == (2, 0)
