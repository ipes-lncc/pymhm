"""Verify cached tensor-face maps against their defining least-squares equations."""

from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits

from pymhm import mapped_rt as candidate

original = candidate


def legacy_transform(permutation):
    """Evaluate the exact previous least-squares rule, including its rejection criterion."""
    canonical = original._UV[np.asarray(permutation)]
    coordinates = np.column_stack((np.ones(4), original._UV))
    transform = np.linalg.lstsq(coordinates, canonical, rcond=None)[0]
    if not np.allclose(coordinates @ transform, canonical, atol=1e-13):
        raise ValueError("shared facets require compatible tensor corner orderings")
    return transform


@pytest.mark.parametrize("permutation", list(permutations(range(4))))
def test_all_corner_permutations(permutation):
    """All eight square symmetries and sixteen incompatible permutations retain semantics."""
    try:
        old = legacy_transform(permutation)
    except ValueError:
        with pytest.raises(ValueError, match="tensor corner orderings"):
            candidate._face_coordinate_transform(permutation)
    else:
        actual = candidate._face_coordinate_transform(permutation)
        assert_array_equal(actual, np.rint(old))
        assert round(np.linalg.det(actual[1:])) == round(np.linalg.det(old[1:]))
        assert not actual.flags.writeable
        assert actual is candidate._face_coordinate_transform(permutation)


def compare(first, second):
    """Require bitwise geometry, incidence, orientation, trace coordinates and refinement."""
    for name in (
        "points",
        "cells",
        "faces",
        "cell_faces",
        "signs",
        "face_transforms",
        "boundary_faces",
    ):
        assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.incidence == second.incidence
    quadrature = original.cube_quadrature(3)[0]
    for a, b in zip(first.geometry(quadrature), second.geometry(quadrature), strict=True):
        assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["cube", "warped", "translated", "rotated", "well"])
def test_complete_mesh_arrays_and_refinement(kind, monkeypatch):
    """Nonaffine and translated meshes preserve all physical coordinates and oriented maps."""
    with threadpool_limits(1):
        base = original.HexMesh.unit_cube(2)
        points, cells = base.points.copy(), base.cells.copy()
        if kind in {"warped", "translated"}:
            points[:, 0] *= 1 + 0.2 * points[:, 1] + 0.1 * points[:, 2]
        if kind == "translated":
            points += (500.0, 2100.0, -1000.0)
        if kind == "rotated":
            permutation = [
                np.flatnonzero(np.all(c[[1, 2, 0]] == original._CORNERS, axis=1))[0]
                for c in original._CORNERS
            ]
            cells[::2] = cells[::2, permutation]
        if kind == "well":
            base = original.HexMesh.annular_prism(np.geomspace(0.2, 50, 5), 10.0, 8)
            points, cells = base.points, base.cells
        subdivisions = 2 if kind == "rotated" else (2, 3, 1)
        with monkeypatch.context() as patch:
            patch.setattr(candidate, "_face_coordinate_transform", legacy_transform)
            first = candidate.HexMesh(points, cells)
            refined_first = first.refined(subdivisions)
        second = candidate.HexMesh(points, cells)
        compare(first, second)
        compare(refined_first, second.refined(subdivisions))
        if kind == "rotated":
            for mesh in (first, second):
                with pytest.raises(ValueError, match="match across"):
                    mesh.refined((2, 3, 1))


@pytest.mark.parametrize("kind", ["repeated", "crossed"])
def test_invalid_full_mesh_semantics(kind, monkeypatch):
    """Caching neither admits crossed face orderings nor duplicate outward orientations."""
    base = original.HexMesh.unit_cube()
    cells = (
        np.tile(base.cells, (2, 1))
        if kind == "repeated"
        else np.vstack((base.cells, base.cells[:, [1, 0, 2, 3, 4, 5, 6, 7]]))
    )
    messages = []
    with pytest.raises(ValueError) as error:
        candidate.HexMesh(base.points, cells)
    messages.append(str(error.value))
    monkeypatch.setattr(candidate, "_face_coordinate_transform", legacy_transform)
    with pytest.raises(ValueError) as error:
        candidate.HexMesh(base.points, cells)
    messages.append(str(error.value))
    assert messages[0] == messages[1]


def test_linear_algebra_count_depends_only_on_permutations(monkeypatch):
    """Two thousand cells reuse the same finite family without repeated least-squares calls."""
    candidate._face_coordinate_transform.cache_clear()
    actual = np.linalg.lstsq
    count = 0

    def counted(*args, **kwargs):
        """Count actual compatibility factorizations without replacing their arithmetic."""
        nonlocal count
        count += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", counted)
    mesh = candidate.HexMesh.annular_prism(np.geomspace(0.2, 50, 5), 10, 8).refined((8, 8, 1))
    assert len(mesh.cells) == 2048
    assert 1 <= count <= 8
    before = count
    candidate.HexMesh(mesh.points, mesh.cells)
    assert count == before
