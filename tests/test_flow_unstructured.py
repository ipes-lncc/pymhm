"""Arbitrary conforming local flow meshes and adaptive boundary closure."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.flow.solver import solve_flow
from pymhm._legacy.models.vector import solve_brinkman
from pymhm.adaptivity.flow import adapt_flow
from pymhm.adaptivity.flow_local_mesh import refine_flow_local_meshes
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.longest_edge import refine_longest_edge
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh


def test_nonuniform_local_taylor_hood_patch_and_facade():
    """Quadratic solenoidal velocity and affine pressure are exact on independent local meshes."""
    mesh = TriangleMesh.unit_square()
    local = tuple(mesh.submesh(cell, 2) for cell in range(2))
    local = tuple(
        refine_longest_edge(fine, np.arange(len(fine.cells)) == cell).mesh
        for cell, fine in enumerate(local)
    )
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)

    def velocity(x):
        """Return a quadratic exactly solenoidal field."""
        return x[:, ::-1] ** 2

    def source(x):
        """Apply -Delta u+2u+grad(x-y) analytically."""
        return 2 * velocity(x) + np.array([-1.0, -3.0])

    result = solve_brinkman(
        mesh,
        local_meshes=local,
        skeleton=skeleton,
        drag=2.0,
        source=source,
        dirichlet=velocity,
    )
    assert result.local_meshes == local
    assert result.l2_error(velocity) < 2e-12
    assert result.pressure_l2_error(lambda x: x[:, 0] - x[:, 1]) < 2e-11
    assert result.divergence_l2() < 2e-11
    with pytest.raises(ValueError, match="one mesh"):
        solve_flow(mesh, local_meshes=())
    with pytest.raises(ValueError, match="macro triangle"):
        solve_flow(mesh, local_meshes=(mesh, mesh))


def test_boundary_closure_is_local_conforming_and_angle_preserving():
    """A single eighth-point trace does not require 64 uniform cells in every local mesh."""
    macro = TriangleMesh.unit_square()
    local = tuple(macro.submesh(cell, 1) for cell in range(2))
    faces = [FaceSpace.uniform(0) for _ in macro.faces]
    face = int(macro.boundary_faces[0])
    faces[face] = FaceSpace((0.0, 0.125, 1.0), (0, 0))
    skeleton = SkeletonSpace(macro, tuple(faces), 2)
    fine = refine_flow_local_meshes(skeleton, local, np.zeros(2, dtype=bool), 100)
    assert fine is not None
    expected = macro.points[macro.faces[face]]
    point = 0.875 * expected[0] + 0.125 * expected[1]
    owner = int(macro.face_cells[face, 0])
    assert np.min(np.linalg.norm(fine[owner].points - point, axis=1)) < 1e-14
    assert len(fine[owner].cells) < 64
    assert fine[1 - owner] is local[1 - owner]
    for cell, part in enumerate(fine):
        validate_submesh(macro, cell, part)
        assert_allclose(part.areas.sum(), macro.areas[cell])
    assert refine_flow_local_meshes(skeleton, local, np.zeros(2, dtype=bool), 1) is None
    uniform = SkeletonSpace(macro, components=2)
    refined = refine_flow_local_meshes(uniform, local, np.ones(2, dtype=bool), 4)
    assert refined is not None and all(len(m.cells) == 4 for m in refined)


def test_nonuniform_adaptive_flow_solves_and_memory_cap():
    """The adaptive route retains actual meshes instead of reporting fictitious uniform counts."""
    mesh = TriangleMesh.unit_square()
    options = dict(local_refiner="longest-edge", source=lambda x: 2 * x, drag=1.0)
    result = adapt_flow(mesh, iterations=2, **options)
    assert result.local_refiner == "longest-edge"
    assert not result.refinements
    assert len(result.solutions) == 3
    errors = [
        s.pressure_l2_error(lambda x: np.sum(x * x, axis=1) - 2 / 3) for s in result.solutions
    ]
    assert errors[-1] < errors[0]
    assert max(s.hybrid.residual for s in result.solutions) < 1e-10
    capped = adapt_flow(mesh, max_local_cells=1, **options)
    assert capped.stop_reason == "resolution_limit"
    assert len(capped.solutions) == 1
    with pytest.raises(ValueError, match="initial local mesh"):
        adapt_flow(mesh, local_refinement=2, max_local_cells=1, **options)
    with pytest.raises(ValueError, match="local_refiner"):
        adapt_flow(mesh, local_refiner="unknown")
    with pytest.raises(ValueError, match="own local_meshes"):
        adapt_flow(mesh, local_meshes=())


def test_fine_indicators_partition_local_error_and_selective_refinement():
    """Fine contributions preserve eta2 and concentrate refinement without changing its formula."""
    mesh = TriangleMesh.unit_square()
    options = dict(
        local_refiner="longest-edge",
        local_error_marking="maximum",
        local_refinement=2,
        source=lambda x: 2 * x,
        drag=1.0,
    )
    result = adapt_flow(mesh, iterations=2, **options)
    assert result.local_error_marking == "maximum"
    for estimate in result.estimators:
        assert_allclose(
            [value.sum() for value in estimate.fine_squared],
            estimate.local_squared,
            rtol=8e-15,
        )
        assert all(np.all(value >= 0) for value in estimate.fine_squared)
    assert result.solutions[-1].pressure_l2_error(lambda x: np.sum(x * x, axis=1) - 2 / 3) < (
        result.solutions[0].pressure_l2_error(lambda x: np.sum(x * x, axis=1) - 2 / 3)
    )
    local = tuple(mesh.submesh(cell, 4) for cell in range(2))
    masks = tuple(np.arange(len(fine.cells)) == 0 for fine in local)
    refined = refine_flow_local_meshes(
        SkeletonSpace(mesh, components=2),
        local,
        np.array([True, False]),
        100,
        fine_marked=masks,
    )
    assert refined is not None
    assert 16 < len(refined[0].cells) < 64
    assert refined[1] is local[1]
    validate_submesh(mesh, 0, refined[0])
    with pytest.raises(ValueError, match="local_error_marking"):
        adapt_flow(mesh, local_error_marking="unsupported")
    with pytest.raises(ValueError, match="requires longest-edge"):
        adapt_flow(mesh, local_error_marking="maximum")
