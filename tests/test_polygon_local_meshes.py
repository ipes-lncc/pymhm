"""Custom polygonal local partitions preserve geometry and the physical solution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.cut_cells import fit_material_mesh
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.pgmhm import solve_pgmhm
from pymhm.polygon import PolygonMesh, solve_darcy_polygons
from pymhm.refinement import validate_submesh
from pymhm.reservoir import CartesianCellField


def l_macro(shift=(0.0, 0.0)):
    """Return one nonconvex cell whose reentrant corner cannot be convexified."""
    points = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    return PolygonMesh(points + shift, (np.arange(6),))


@pytest.mark.parametrize("shift", [(0.0, 0.0), (500.0, 2100.0)])
def test_fitted_polygon_partition_and_translated_coordinates(shift):
    """Material cuts remain inside an L cell and cover all six oriented edges."""
    macro = l_macro(shift)
    field = CartesianCellField(np.ones((7, 7)), (0.3, 0.3))
    base = fit_material_mesh(l_macro().submesh(0, 3), field)
    fine = TriangleMesh(base.points + shift, base.cells)
    validate_submesh(macro, 0, fine)
    assert_allclose(fine.areas.sum(), 3, atol=1e-12)
    with pytest.raises(ValueError, match="cover its macro polygon"):
        validate_submesh(macro, 0, TriangleMesh(fine.points + [1e-6, 0], fine.cells))


def test_polygon_rejects_convex_bridge_area_loss_and_hanging_internal_edges():
    """Membership, measure and topological boundary checks detect distinct defects."""
    macro = l_macro()
    fine = macro.submesh(0, 3)
    with pytest.raises(ValueError, match="cover its macro polygon"):
        validate_submesh(macro, 0, TriangleMesh(fine.points, fine.cells[:-1]))
    outside = PolygonMesh(
        np.array([[0, 0], [2, 0], [2, 1], [0, 2]], float), (np.arange(4),)
    ).submesh(0, 1)
    assert_allclose(outside.areas.sum(), macro.areas[0])
    with pytest.raises(ValueError, match="cover its macro polygon|oriented macro edge"):
        validate_submesh(macro, 0, outside)
    # All fine triangles are geometrically correct, but independent node copies
    # turn internal faces into unmatched boundaries instead of a conforming mesh.
    disconnected = TriangleMesh(
        fine.points[fine.cells].reshape(-1, 2), np.arange(3 * len(fine.cells)).reshape(-1, 3)
    )
    with pytest.raises(ValueError, match="unmatched interior boundary"):
        validate_submesh(macro, 0, disconnected)
    # Duplicate lower rectangle area replaces the missing upper arm. Its outer
    # boundary multiplicity and reverse interior edges cannot partition the L.
    points = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 1]])
    doubled = TriangleMesh(points, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]))
    with pytest.raises(ValueError, match="oriented macro edge"):
        validate_submesh(macro, 0, doubled)


def test_underresolved_boundary_edge_has_no_ambiguous_macro_assignment():
    """An edge shorter than coordinate uncertainty cannot span two macro faces."""
    points = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [0, 1]], float)
    macro = PolygonMesh(points, (np.arange(5),))
    fine_points = np.array([[0, 0], [1 - 1e-14, 0], [1 + 1e-14, 0], [2, 0], [2, 1], [0, 1]])
    fine = TriangleMesh(fine_points, np.array([[0, 1, 5], [1, 2, 5], [2, 4, 5], [2, 3, 4]]))
    with pytest.raises(ValueError, match="multiple macro edges"):
        validate_submesh(macro, 0, fine)


def test_custom_polygon_meshes_recover_affine_dirichlet_in_both_methods():
    """Both public solvers consume the validated mesh and the same weak trace signs."""
    macro = l_macro()
    fine = macro.submesh(0, 3)
    skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(1) for _ in macro.faces))

    def exact(points):
        """Use a nonzero affine trace on all six sides of the nonconvex polygon."""
        return 1 + points[:, 0] - 2 * points[:, 1]

    options = dict(local_meshes=(fine,), skeleton=skeleton, degree=3, dirichlet=exact)
    for result in (
        solve_darcy_polygons(macro, **options),
        solve_pgmhm(macro, **options, stabilization_parameter=0.1),
    ):
        assert result.l2_error(exact) < 2e-12
        assert result.local_meshes[0] is fine
