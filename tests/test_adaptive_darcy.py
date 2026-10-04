"""Geometric conformity, bulk marking and estimator-driven Darcy refinement."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.adaptivity.darcy import mark_dorfler, solve_adaptive_darcy
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.refinement import refine_triangles, transfer_skeleton
from pymhm.meshes.triangle import TriangleMesh


def sine(points):
    """Smooth homogeneous-boundary exact pressure."""
    return np.prod(np.sin(np.pi * points), axis=1)


def gradient(points):
    """Analytical gradient, independently defined from the load."""
    a, b = np.pi * points.T
    return np.pi * np.column_stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)))


def assert_conforming(mesh):
    """Check no mesh vertex lies strictly inside another edge, without topology reuse."""
    for start, end in mesh.points[mesh.faces]:
        tangent = end - start
        parameter = (mesh.points - start) @ tangent / (tangent @ tangent)
        interior = (parameter > 1e-10) & (parameter < 1 - 1e-10)
        distance = np.linalg.norm(mesh.points - start - parameter[:, None] * tangent, axis=1)
        assert not np.any(interior & (distance < 1e-12))


def test_red_green_geometry_and_parent_integrals():
    """A single marked triangle creates one red and one green neighbor."""
    mesh = TriangleMesh.unit_square()
    result = refine_triangles(mesh, [True, False])
    assert len(result.mesh.cells) == 6
    assert_allclose(np.bincount(result.parent_cells, weights=result.mesh.areas), mesh.areas)
    assert_conforming(result.mesh)
    for face in mesh.boundary_faces:
        children = np.flatnonzero(result.parent_faces == face)
        assert_allclose(result.mesh.lengths[children].sum(), mesh.lengths[face])
    empty = refine_triangles(mesh, [False, False])
    assert_allclose(empty.mesh.points, mesh.points)
    assert_allclose(empty.parent_faces, np.arange(len(mesh.faces)))
    with pytest.raises(ValueError, match="boolean"):
        refine_triangles(mesh, [1, 0])


def test_closure_multiple_cycles_and_skeleton_restriction():
    """Two-edge closure and reversed global normals retain physical face partitions."""
    mesh = TriangleMesh.unit_square(2)
    spaces = tuple(FaceSpace((0.0, 0.25, 0.6, 1.0), (1, 2, 3), True) for _ in mesh.faces)
    skeleton = SkeletonSpace(mesh, spaces, components=2)
    for cycle in range(3):
        marked = np.arange(len(mesh.cells)) % 3 == cycle % 3
        refined = refine_triangles(mesh, marked)
        transferred = transfer_skeleton(skeleton, refined, new_face=FaceSpace.uniform(1))
        assert_conforming(refined.mesh)
        assert_allclose(refined.mesh.areas.sum(), 1.0)
        assert transferred.components == 2
        for face, parent in enumerate(refined.parent_faces):
            if parent < 0:
                assert transferred.faces[face] == FaceSpace.uniform(1)
                continue
            assert transferred.faces[face].continuous == skeleton.faces[parent].continuous
            start, end = mesh.points[mesh.faces[parent]]
            tangent = end - start
            for a, b, degree in zip(
                transferred.faces[face].breaks[:-1],
                transferred.faces[face].breaks[1:],
                transferred.faces[face].degrees,
                strict=True,
            ):
                x, y = refined.mesh.points[refined.mesh.faces[face]]
                midpoint = x + (a + b) / 2 * (y - x)
                t = (midpoint - start) @ tangent / (tangent @ tangent)
                segment = np.searchsorted(skeleton.faces[parent].breaks, t, side="right") - 1
                assert degree == skeleton.faces[parent].degrees[segment]
        mesh, skeleton = refined.mesh, transferred
    from dataclasses import replace

    with pytest.raises(ValueError, match="ancestry"):
        transfer_skeleton(
            skeleton, replace(refined, parent_faces=np.array([])), new_face=FaceSpace.uniform()
        )


@pytest.mark.parametrize(
    "values", [[1, 4, 2, 3], [1e-300, 4e-300, 2e-300, 3e-300], [1e300, 4e300, 2e300, 3e300]]
)
def test_bulk_marking_scale_and_minimal_cardinality(values):
    """The two largest terms are the minimal set containing 70 percent of the sum."""
    assert_allclose(mark_dorfler(values, 0.7), [False, True, False, True])
    assert mark_dorfler(values, 1).all()
    assert not mark_dorfler([0, 0]).any()
    assert_allclose(mark_dorfler([1, 1, 1], 0.5), [True, True, False])


@pytest.mark.parametrize("values", [[], [[1]], [1j], [-1], [np.inf]])
def test_invalid_marking_data(values):
    """Bulk marking rejects invalid squared indicators."""
    with pytest.raises(ValueError, match="local_squared"):
        mark_dorfler(values)


@pytest.mark.parametrize("theta", [0, 1.1, np.nan, 1j, [0.5]])
def test_invalid_bulk_fraction(theta):
    """Fractions must be real finite scalars in the mathematical bulk interval."""
    with pytest.raises(ValueError, match="theta"):
        mark_dorfler([1], theta)


def test_adaptive_sine_reliability_and_refinement():
    """Five actual solves reduce an independently integrated energy error."""
    result = solve_adaptive_darcy(
        TriangleMesh.unit_square(),
        iterations=5,
        source=lambda x: 2 * np.pi**2 * sine(x),
        local_refinement=2,
        quadrature_order=8,
        estimator_order=8,
    )
    assert len(result.solutions) == 5
    assert len(result.refinements) == 4
    errors = [estimate.energy_error(gradient) for estimate in result.estimators]
    assert errors[-1] < errors[0]
    assert np.all(result.totals >= errors)
    assert len(result.solutions[-1].skeleton.mesh.cells) > 2


def test_mixed_boundaries_and_cellwise_bounds_are_transferred():
    """Boundary classifications and local material certificates survive edge bisection."""
    mesh = TriangleMesh.unit_square()
    natural = {
        int(face): 0.0
        for face in mesh.boundary_faces
        if np.isclose(mesh.points[mesh.faces[face], 1].mean(), 1)
    }
    result = solve_adaptive_darcy(
        mesh,
        iterations=2,
        permeability=np.diag([2.0, 1.0]),
        source=1.0,
        neumann=natural,
        ellipticity_lower_bound=[1.0, 1.0],
        local_refinement=2,
    )
    assert len(result.solutions) == 2
    assert_allclose(result.estimators[-1].ellipticity_lower_bounds, 1.0)
    assert result.totals[-1] < result.totals[0]


def test_stopping_and_input_contracts():
    """Tolerance and cell budget prevent extra solves without discarding a solved state."""
    mesh = TriangleMesh.unit_square()
    zero = solve_adaptive_darcy(mesh, iterations=2, local_refinement=1)
    assert len(zero.solutions) == 1
    fitted = solve_adaptive_darcy(
        mesh, iterations=1, local_mesh_factory=lambda coarse, cell: coarse.submesh(cell, 2)
    )
    assert fitted.totals[0] == 0
    for kwargs in ({"tolerance": 1e6}, {"maximum_cells": 2}):
        stopped = solve_adaptive_darcy(mesh, source=1, iterations=2, local_refinement=2, **kwargs)
        assert len(stopped.solutions) == 1
    for kwargs, message in [
        ({"tolerance": -1}, "tolerance"),
        ({"tolerance": 1j}, "tolerance"),
        ({"formulation": "mixed"}, "primal"),
        ({"skeleton": SkeletonSpace(mesh, components=2)}, "scalar"),
        ({"local_meshes": ()}, "local_mesh_factory"),
    ]:
        with pytest.raises(ValueError, match=message):
            solve_adaptive_darcy(mesh, **kwargs)
