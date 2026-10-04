"""Conforming edge-star refinement and complete adaptive state/boundary transfer."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.adaptivity.darcy_3d import solve_adaptive_darcy_3d
from pymhm.meshes.refinement_3d import refine_tetrahedra
from pymhm.meshes.tetrahedron import TetraMesh


@pytest.fixture(autouse=True)
def one_thread():
    """Use one native thread for lightweight adaptation checks."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("subdivisions", [1, 2])
def test_edge_stars_volume_ancestry_and_boundary_cover(subdivisions):
    """Bisection preserves parent volumes and the complete boundary face ancestry."""
    mesh = TetraMesh.unit_cube(subdivisions)
    selected = np.zeros(len(mesh.cells), dtype=bool)
    selected[0] = True
    result = refine_tetrahedra(mesh, selected)
    assert len(result.mesh.cells) > len(mesh.cells)
    assert_allclose(
        np.bincount(result.cell_parents, weights=result.mesh.volumes), mesh.volumes, atol=2e-16
    )
    for face in result.mesh.boundary_faces:
        parent = result.face_parents[face]
        assert parent in mesh.boundary_faces
        assert_allclose(result.mesh.normals[face], mesh.normals[parent], atol=1e-15)
    covered = np.bincount(
        result.face_parents[result.mesh.boundary_faces],
        weights=result.mesh.areas[result.mesh.boundary_faces],
        minlength=len(mesh.faces),
    )
    assert_allclose(covered[mesh.boundary_faces], mesh.areas[mesh.boundary_faces], atol=3e-16)
    assert np.count_nonzero(result.cell_parents == 0) >= 2
    empty = refine_tetrahedra(mesh, np.zeros(len(mesh.cells), dtype=bool))
    assert_allclose(empty.mesh.points, mesh.points)
    assert_allclose(empty.mesh.cells, mesh.cells)
    with pytest.raises(ValueError, match="boolean"):
        refine_tetrahedra(mesh, [1])


def test_adaptive_states_and_physical_neumann_transfer():
    """Each solved state is saved before refinement and retains inherited flux boundary values."""
    mesh = TetraMesh.unit_cube()
    bottom = {
        int(f): 0.0 for f in mesh.boundary_faces if np.all(mesh.points[mesh.faces[f], 2] == 0)
    }
    seen = []
    result = solve_adaptive_darcy_3d(
        mesh,
        source=1,
        neumann=bottom,
        iterations=3,
        ellipticity_lower_bound=np.ones(len(mesh.cells)),
        on_state=lambda step, solution, estimate: seen.append((step, solution, estimate)),
    )
    assert len(seen) == len(result.solutions) == 3
    assert len(result.refinements) == 2
    assert np.isfinite(result.totals).all()
    assert all(
        len(b.local_meshes) > len(a.local_meshes)
        for a, b in zip(result.solutions, result.solutions[1:], strict=False)
    )
    for solution, estimate in zip(result.solutions, result.estimators, strict=True):
        assert solution.degree >= int(solution.skeleton.degrees.max()) + 3
        assert np.max(abs(solution.conservation_residuals())) < 3e-13
        assert estimate.equilibrium_defect.max() < 3e-13
        coarse = solution.skeleton.mesh
        for face in coarse.boundary_faces:
            if np.all(coarse.points[coarse.faces[face], 2] == 0):
                assert_allclose(solution.hybrid.trace[solution.skeleton.dofs(int(face))], 0, atol=0)


def test_stopping_and_invalid_controls():
    """Tolerance, zero residual and cell caps stop only after a fully solved state."""
    mesh = TetraMesh.unit_cube()
    assert len(solve_adaptive_darcy_3d(mesh).solutions) == 1
    assert len(solve_adaptive_darcy_3d(mesh, source=1, tolerance=1).solutions) == 1
    assert len(solve_adaptive_darcy_3d(mesh, source=1, maximum_cells=6).solutions) == 1
    assert len(solve_adaptive_darcy_3d(mesh, source=1, iterations=2).solutions) == 2
    for options in ({"tolerance": -1}, {"tolerance": 1j}, {"skeleton": None}, {"theta": 0}):
        with pytest.raises(ValueError):
            solve_adaptive_darcy_3d(mesh, **options)
