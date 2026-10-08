"""Sampled physical compatibility retains relative cancellation without amplitude floors."""

import numpy as np
import pytest

from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.normal import (
    boundary_normal_zero,
    boundary_tangent_2d,
    boundary_tangent_3d,
    normal_component_zero,
    trace_represents,
)
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def test_relative_normal_cancellation_and_invalid_contracts():
    values = np.array([[1.0, 1.0], [1e-16, 0.0]])
    assert normal_component_zero(values, [1, -1])
    assert not normal_component_zero(values, [1, -1], pointwise=True)
    assert not normal_component_zero([[1e-20, 0]], [1, 0])
    assert normal_component_zero(np.empty((0, 2)), [1, 0])
    with pytest.raises(ValueError, match="dimension"):
        normal_component_zero([1, 0], [1, 0])
    with pytest.raises(ValueError, match="dimension"):
        normal_component_zero([[1, 0]], [1])
    for tolerance in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="rtol"):
            normal_component_zero([[1, 0]], [1, 0], rtol=tolerance)


@pytest.mark.parametrize("dimension", [2, 3, "polygon"])
def test_boundary_tangency_resolves_original_faces_and_small_nontangent_data(dimension):
    if dimension == 2:
        space, check = SkeletonSpace(TriangleMesh.unit_square()), boundary_tangent_2d
    elif dimension == 3:
        space, check = TriangularSkeleton(TetraMesh.unit_cube()), boundary_tangent_3d
    else:
        space, check = PolygonalSkeleton3D(PolyhedralMesh.cubes()), boundary_tangent_3d
    dim = space.mesh.points.shape[1]
    assert check(space, np.zeros(dim))
    assert check(space, np.ones(dim), faces=[])
    assert not check(space, np.full(dim, 1e-20))
    assert boundary_normal_zero(space, np.zeros(dim))
    assert not boundary_normal_zero(space, np.ones(dim))
    face = int(space.mesh.boundary_faces[0])
    normal = space.mesh.normals[face]
    vector = np.eye(dim)[np.argmin(abs(normal))]
    vector -= (vector @ normal) * normal
    assert check(space, vector, faces=[face])
    with pytest.raises(TypeError, match="trace mesh"):
        check(object(), 0)
    with pytest.raises(ValueError, match="physical points"):
        boundary_normal_zero(space, np.zeros(dim + 1))


def test_trace_representation_declares_component_and_rank_requirements():
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    assert trace_represents(space, 0, lambda points: 1 + points[:, 0])
    assert not trace_represents(space, 0, lambda points: points[:, 0] ** 3 + points[:, 1] ** 3)
    with pytest.raises(ValueError, match="components"):
        trace_represents(space, 0, [1, 2, 3])

    class RankDeficient:
        mesh = space.mesh

        def trace_quadrature(self, face, *, order):
            return np.arange(2), np.zeros((2, 2)), np.ones(2), np.ones((2, 2, 1))

    assert not trace_represents(RankDeficient(), 0, 1)
