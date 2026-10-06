"""Independent DOLFINx/Basix mixed Darcy assembly on the same fine mesh."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.darcy.mixed_bdm import solve_darcy_bdm
from pymhm.backends.fenics import from_ufl
from pymhm.fem.hdiv.bdm import bdm2_evaluate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


def test_bdm2_darcy_matches_independent_global_dolfinx():
    """Compare non-exact quartic pressure and flux with a native global mixed solve."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    coarse = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(coarse, tuple(FaceSpace.uniform(2, 2) for _ in coarse.faces))

    def exact(points):
        """Quartic pressure lies outside the local pressure and flux spaces."""
        x, y = points.T
        return x**4 + y**3 + x * y * y

    def source(points):
        """Negative anisotropic Hessian contraction for K=[[2,.3],[.3,1]]."""
        x, y = points.T
        return -24 * x * x - 7.2 * y - 2 * x

    permeability = np.array([[2.0, 0.3], [0.3, 1.0]])
    native = solve_darcy_bdm(
        coarse, skeleton=skeleton, permeability=permeability, dirichlet=exact, source=source
    )
    all_points = np.vstack([mesh.points for mesh in native.local_meshes])
    points, inverse = np.unique(all_points, axis=0, return_inverse=True)
    cells = []
    offset = 0
    for mesh in native.local_meshes:
        cells.extend(inverse[offset + mesh.cells])
        offset += len(mesh.points)
    coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF, np.array(cells), x=points, e=coordinate_element
    )
    element = basix.ufl.mixed_element(
        [basix.ufl.element("BDM", "triangle", 2), basix.ufl.element("DG", "triangle", 1)]
    )
    space = dolfinx.fem.functionspace(domain, element)
    q, p = ufl.TrialFunctions(space)
    v, w = ufl.TestFunctions(space)
    x = ufl.SpatialCoordinate(domain)
    prescribed = x[0] ** 4 + x[1] ** 3 + x[0] * x[1] ** 2
    force = -24 * x[0] ** 2 - 7.2 * x[1] - 2 * x[0]
    inverse_tensor = ufl.as_matrix(np.linalg.inv(permeability))
    a = (ufl.inner(inverse_tensor * q, v) - p * ufl.div(v) - w * ufl.div(q)) * ufl.dx
    load = -force * w * ufl.dx - prescribed * ufl.dot(v, ufl.FacetNormal(domain)) * ufl.ds
    problem = from_ufl(a, load, [], np.empty(0, dtype=int))
    solution = dolfinx.fem.Function(space)
    solution.x.array[:] = problem.condense().source
    q_ref, p_ref = (field.collapse() for field in solution.split())
    original_to_local = np.argsort(domain.topology.original_cell_index)
    bary, _ = triangle_quadrature(4)
    offset = 0
    for mesh, flux, pressure in zip(native.local_meshes, native.flux, native.pressure, strict=True):
        coordinates = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
        evaluation_points = np.column_stack((coordinates, np.zeros(len(coordinates))))
        indices = np.repeat(original_to_local[offset : offset + len(mesh.cells)], len(bary)).astype(
            np.int32
        )
        values, _ = bdm2_evaluate(mesh, flux, bary)
        assert_allclose(
            values.reshape(-1, 2), q_ref.eval(evaluation_points, indices), atol=3e-11, rtol=0
        )
        assert_allclose(
            (pressure @ bary.T).ravel(),
            p_ref.eval(evaluation_points, indices).ravel(),
            atol=3e-12,
            rtol=0,
        )
        offset += len(mesh.cells)
