"""Geometric, orientation, and polynomial trace-space verification."""

from math import factorial

import numpy as np
import pytest

from pymhm.core.validation import positive_int
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("value", [True, False, 1.0, "2", -1, 0, np.nan])
def test_positive_int_rejects_invalid_parameters(value):
    with pytest.raises(ValueError, match="integer"):
        positive_int(value, "resolution")


def test_positive_int_accepts_numpy_integers_and_zero_minimum():
    assert positive_int(np.int32(4), "order") == 4
    assert positive_int(0, "degree", minimum=0) == 0


@pytest.mark.parametrize("nx,ny", [(1, None), (2, 3), (4, 2)])
def test_grid_topology_geometry_and_discrete_divergence(nx, ny):
    mesh = TriangleMesh.unit_square(nx, ny)
    ny = nx if ny is None else ny
    assert len(mesh.points) == (nx + 1) * (ny + 1)
    assert len(mesh.cells) == 2 * nx * ny
    assert len(mesh.faces) == 3 * nx * ny + nx + ny
    assert len(mesh.boundary_faces) == 2 * (nx + ny)
    assert len(mesh.points) - len(mesh.faces) + len(mesh.cells) == 1
    np.testing.assert_allclose(mesh.areas, 1 / (2 * nx * ny))
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1)
    np.testing.assert_allclose(mesh.areas.sum(), 1)
    for cell, faces in enumerate(mesh.cell_faces):
        outward = mesh.signs[cell, :, None] * mesh.normals[faces]
        np.testing.assert_allclose((outward * mesh.lengths[faces, None]).sum(axis=0), 0, atol=1e-15)
        centroid = mesh.points[mesh.cells[cell]].mean(axis=0)
        midpoints = mesh.points[mesh.faces[faces]].mean(axis=1)
        assert np.all(np.einsum("ij,ij->i", midpoints - centroid, outward) > 0)
    for face, adjacent in enumerate(mesh.face_cells):
        signs = [
            mesh.signs[cell, np.flatnonzero(mesh.cell_faces[cell] == face)[0]]
            for cell in adjacent
            if cell >= 0
        ]
        assert signs == ([1] if adjacent[1] < 0 else [1, -1])
    boundary = mesh.boundary_faces
    midpoints = mesh.points[mesh.faces[boundary]].mean(axis=1)
    flux = np.einsum("ij,ij->i", midpoints, mesh.normals[boundary]) * mesh.lengths[boundary]
    np.testing.assert_allclose(flux.sum(), 2)


def test_clockwise_input_is_copied_reoriented_and_frozen():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.array([[0, 2, 1]])
    mesh = TriangleMesh(points, cells)
    np.testing.assert_array_equal(mesh.cells, [[0, 1, 2]])
    np.testing.assert_array_equal(cells, [[0, 2, 1]])
    points[0] = 9
    cells[0] = 0
    np.testing.assert_array_equal(mesh.points[0], [0, 0])
    for name in ("points", "cells", "faces", "cell_faces", "signs", "face_cells"):
        assert not getattr(mesh, name).flags.writeable
        with pytest.raises(ValueError):
            getattr(mesh, name).flat[0] = 0


@pytest.mark.parametrize(
    "points,cells,message",
    [
        ([0.0, 1.0], [[0, 1, 2]], "points"),
        ([[0.0, 1.0, 2.0]], [[0, 1, 2]], "points"),
        ([[0.0, np.inf], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]], "points"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [], "cells"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1]], "cells"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0, 2.0]], "integers"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[-1, 1, 2]], "outside"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 3]], "outside"),
        ([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [[0, 1, 2]], "degenerate"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 0, 2]], "degenerate"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2], [0, 2, 1]], "duplicate"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0, 1, 2], [0, 1, 3]], "nonmanifold"),
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]],
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            "nonmanifold",
        ),
    ],
)
def test_invalid_meshes_fail_explicitly(points, cells, message):
    with pytest.raises(ValueError, match=message):
        TriangleMesh(points, cells)


@pytest.mark.parametrize("cell,n", [(0, 1), (0, 2), (1, 3), (3, 5)])
def test_submesh_partitions_macrocell_without_area_or_orientation_loss(cell, n):
    mesh = TriangleMesh.unit_square(2)
    fine = mesh.submesh(cell, n)
    assert len(fine.cells) == n * n
    assert len(fine.points) == (n + 1) * (n + 2) // 2
    assert len(fine.boundary_faces) == 3 * n
    np.testing.assert_allclose(fine.areas, mesh.areas[cell] / n**2)
    np.testing.assert_allclose(fine.areas.sum(), mesh.areas[cell])
    np.testing.assert_allclose(
        fine.points[fine.cells].mean(axis=(0, 1)), mesh.points[mesh.cells[cell]].mean(axis=0)
    )


@pytest.mark.parametrize("cell,subdivisions", [(-1, 1), (2, 1), (0, 0), (0, 1.5)])
def test_submesh_rejects_invalid_parameters(cell, subdivisions):
    with pytest.raises(ValueError):
        TriangleMesh.unit_square().submesh(cell, subdivisions)


@pytest.mark.parametrize("nx,ny", [(0, None), (True, None), (1, 0), (1, 1.5)])
def test_grid_rejects_invalid_resolutions(nx, ny):
    with pytest.raises(ValueError):
        TriangleMesh.unit_square(nx, ny)


def test_hp_face_space_support_endpoint_conventions_and_orthogonal_mass():
    space = FaceSpace((0.0, 0.2, 0.7, 1.0), (0, 2, 3))
    assert space.size == 8
    np.testing.assert_array_equal(space.evaluate(0), [[1, 0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(space.evaluate(0.2), [[0, 1, -1, 1, 0, 0, 0, 0]])
    np.testing.assert_allclose(space.evaluate(1), [[0, 0, 0, 0, 1, 1, 1, 1]])
    points, weights = space.quadrature(5)
    basis = space.evaluate(points)
    exact = [
        length / (2 * degree + 1)
        for length, maximum in zip(np.diff(space.breaks), space.degrees, strict=True)
        for degree in range(maximum + 1)
    ]
    np.testing.assert_allclose(basis.T @ (weights[:, None] * basis), np.diag(exact), atol=4e-16)
    for degree in range(10):
        np.testing.assert_allclose(weights @ points**degree, 1 / (degree + 1), atol=1e-15)
    assert space.evaluate([]).shape == (0, space.size)


def test_uniform_face_space():
    space = FaceSpace.uniform(2, 4)
    np.testing.assert_allclose(space.breaks, np.arange(5) / 4)
    assert space.degrees == (2, 2, 2, 2)
    assert space.size == 12
    points, weights = FaceSpace().quadrature()
    np.testing.assert_allclose(weights @ points**2, factorial(2) / factorial(3))


@pytest.mark.parametrize(
    "breaks,degrees",
    [
        ((), ()),
        ((0.0, 1.0), ()),
        ((0.0, 0.5, 1.0), (0,)),
        ((0.0, 0.0, 1.0), (0, 0)),
        ((0.1, 1.0), (0,)),
        ((0.0, 0.9), (0,)),
        ((0.0, np.nan, 1.0), (0, 1)),
        ((0.0, 1.0), (-1,)),
        ((0.0, 1.0), (True,)),
    ],
)
def test_face_space_rejects_invalid_partitions(breaks, degrees):
    with pytest.raises(ValueError):
        FaceSpace(breaks, degrees)


@pytest.mark.parametrize("coordinates", [[-0.01], [1.01], [np.nan], [np.inf], [[0, 1]]])
def test_face_evaluation_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError, match="coordinates"):
        FaceSpace().evaluate(coordinates)


def test_face_quadrature_and_uniform_partition_reject_invalid_orders():
    with pytest.raises(ValueError):
        FaceSpace().quadrature(0)
    with pytest.raises(ValueError):
        FaceSpace.uniform(subdivisions=0)


def test_skeleton_numbering_with_independent_hp_and_vector_components():
    mesh = TriangleMesh.unit_square()
    spaces = tuple(FaceSpace.uniform(i % 3, i % 2 + 1) for i in range(len(mesh.faces)))
    skeleton = SkeletonSpace(mesh, spaces, components=2)
    assert skeleton.size == sum(space.size for space in spaces) * 2
    assert not skeleton.offsets.flags.writeable
    np.testing.assert_array_equal(
        np.concatenate([skeleton.dofs(i) for i in range(len(spaces))]), np.arange(skeleton.size)
    )
    for cell in range(len(mesh.cells)):
        expected = np.concatenate([skeleton.dofs(int(face)) for face in mesh.cell_faces[cell]])
        np.testing.assert_array_equal(skeleton.cell_dofs(cell), expected)
    default = SkeletonSpace(mesh)
    assert default.size == len(mesh.faces)
    assert all(space == FaceSpace() for space in default.faces)


@pytest.mark.parametrize("faces,components", [((), 1), ((None,) * 5, 1), (None, 0)])
def test_skeleton_rejects_incompatible_spaces(faces, components):
    with pytest.raises(ValueError):
        SkeletonSpace(TriangleMesh.unit_square(), faces, components)


@pytest.mark.parametrize("face", [-1, 5, True, 1.0])
def test_skeleton_rejects_invalid_face_indices(face):
    with pytest.raises(ValueError):
        SkeletonSpace(TriangleMesh.unit_square()).dofs(face)


@pytest.mark.parametrize("cell", [-1, 2, True, 1.0])
def test_skeleton_rejects_invalid_cell_indices(cell):
    with pytest.raises(ValueError):
        SkeletonSpace(TriangleMesh.unit_square()).cell_dofs(cell)


@pytest.mark.parametrize("order", [1, 2, 5, 8])
def test_segmented_face_gauss_rule_preserves_mass_and_polynomial_moments(order):
    """Every segment integrates its physical length and all Gauss-exact powers."""
    face = FaceSpace((0.0, 0.17, 0.63, 1.0), (2, 2, 2))
    points, weights = face.quadrature(order)
    assert points.dtype == weights.dtype == np.dtype(float)
    assert np.all(weights > 0)
    for segment, (left, right) in enumerate(zip(face.breaks[:-1], face.breaks[1:], strict=True)):
        selected = slice(segment * order, (segment + 1) * order)
        for power in range(2 * order):
            expected = (right ** (power + 1) - left ** (power + 1)) / (power + 1)
            actual = np.sum(
                weights[selected].astype(np.longdouble)
                * points[selected].astype(np.longdouble) ** power
            )
            np.testing.assert_allclose(actual, expected, rtol=8e-14, atol=0)
