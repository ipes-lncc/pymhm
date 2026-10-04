"""Geometric and PDE invariants of conforming longest-edge propagation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.meshes.longest_edge import refine_longest_edge
from pymhm.meshes.refinement import transfer_skeleton, validate_submesh


def minimum_angle(mesh):
    """Compute the smallest physical angle independently of the refinement rule."""
    vertices = mesh.points[mesh.cells]
    values = []
    for i in range(3):
        a = vertices[:, (i + 1) % 3] - vertices[:, i]
        b = vertices[:, (i + 2) % 3] - vertices[:, i]
        cosine = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        values.append(np.arccos(np.clip(cosine, -1, 1)))
    return float(np.min(values))


def assert_partition(original, refined):
    """Check exact ancestry, physical area, and absence of hanging boundaries."""
    assert_allclose(
        np.bincount(refined.parent_cells, weights=refined.mesh.areas),
        original.areas,
        rtol=2e-13,
        atol=1e-16,
    )
    for parent in range(len(original.cells)):
        cells = refined.mesh.cells[refined.parent_cells == parent]
        nodes, inverse = np.unique(cells, return_inverse=True)
        local = TriangleMesh(refined.mesh.points[nodes], inverse.reshape(-1, 3))
        validate_submesh(original, parent, local)
    for face in refined.mesh.boundary_faces:
        assert refined.parent_faces[face] in original.boundary_faces


def test_terminal_interior_and_boundary_bisection():
    """A marked square triangle refines its shared longest edge conformingly."""
    original = TriangleMesh.unit_square()
    refined = refine_longest_edge(original, np.array([True, False]))
    assert len(refined.mesh.cells) == 4
    assert len(refined.mesh.points) == 5
    assert_partition(original, refined)
    unchanged = refine_longest_edge(original, np.zeros(2, dtype=bool))
    assert_array_equal(unchanged.mesh.cells, original.cells)
    assert_array_equal(unchanged.parent_cells, [0, 1])
    triangle = TriangleMesh(np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    boundary = refine_longest_edge(triangle, np.ones(1, dtype=bool))
    assert len(boundary.mesh.cells) == 2
    assert_partition(triangle, boundary)


def test_longest_edge_propagates_and_preserves_angles_under_repeated_marking():
    """Nonuniform propagation retains the original angle bound across forty markings."""
    original = TriangleMesh.unit_square(3, 2)
    mesh = TriangleMesh(original.points @ np.array([[1.0, 0.3], [0.0, 1.7]]), original.cells)
    bound = minimum_angle(mesh) / 2
    generator = np.random.default_rng(991)
    for _ in range(40):
        marked = np.zeros(len(mesh.cells), dtype=bool)
        marked[generator.choice(len(mesh.cells), size=3, replace=False)] = True
        refined = refine_longest_edge(mesh, marked)
        assert np.all(np.bincount(refined.parent_cells)[marked] >= 2)
        assert_partition(mesh, refined)
        assert minimum_angle(refined.mesh) >= bound - 3e-13
        mesh = refined.mesh


def test_skeleton_and_neumann_data_follow_exact_ancestry():
    """Pressure gauge and physical boundary flux survive nonuniform propagation."""
    original = TriangleMesh.unit_square(2)
    space = FaceSpace((0.0, 0.3, 1.0), (1, 2), continuous=True)
    skeleton = SkeletonSpace(original, tuple(space for _ in original.faces))
    refined = refine_longest_edge(original, np.arange(len(original.cells)) % 3 == 0)
    transferred = transfer_skeleton(skeleton, refined, new_face=FaceSpace.uniform(1))
    for face, parent in enumerate(refined.parent_faces):
        if parent >= 0:
            assert transferred.faces[face].continuous
            assert set(transferred.faces[face].degrees) <= {1, 2}
    # The affine flux is representable with a constant normal value per face.
    mesh = refined.mesh
    natural = {int(face): -float(mesh.normals[face] @ [2.0, 3.0]) for face in mesh.boundary_faces}
    solution = solve_darcy(mesh, degree=2, neumann=natural, mean_pressure=3.5)
    assert solution.l2_error(lambda x: 1 + x @ [2.0, 3.0]) < 2e-12
    assert solution.flux_l2_error((-2.0, -3.0)) < 2e-12
    assert max(abs(solution.conservation_residuals())) < 2e-13


def test_adaptive_loops_use_the_selected_macro_refiner():
    """The weighted estimator drives either loop through the same explicit geometry operation."""
    from pymhm.adaptivity.darcy import solve_adaptive_darcy
    from pymhm.adaptivity.darcy_balanced import solve_balanced_adaptive_darcy

    mesh = TriangleMesh.unit_square()
    ordinary = solve_adaptive_darcy(
        mesh,
        source=1.0,
        degree=2,
        iterations=2,
        local_refinement=2,
        reconstruction_degree=2,
        estimator_order=5,
        macro_refiner=refine_longest_edge,
    )
    balanced = solve_balanced_adaptive_darcy(
        mesh,
        source=1.0,
        degree=2,
        iterations=2,
        local_refinement=2,
        reconstruction_degree=2,
        estimator_order=5,
        local_error_ratio=1e3,
        macro_refiner=refine_longest_edge,
    )
    assert len(ordinary.solutions[-1].skeleton.mesh.cells) == 4
    assert balanced.decisions == ("macro", "stop")
    for first, second in zip(
        ordinary.solutions[-1].pressure, balanced.result.solutions[-1].pressure, strict=True
    ):
        assert_allclose(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("marked", [[1, 0], [True], np.zeros((2, 1), dtype=bool)])
def test_invalid_selection(marked):
    """Reject integer masks and dimension mismatches before changing topology."""
    with pytest.raises(ValueError, match="boolean array"):
        refine_longest_edge(TriangleMesh.unit_square(), marked)
