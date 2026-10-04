"""Independent Basix/DOLFINx interpolation of all implemented RT orders."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.backends.fenics import from_ufl
from pymhm.fem.hdiv.rt import rt_evaluate, rt_interpolate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_canonical_interpolation_matches_basix_and_native_divergence(degree):
    """Compare physical vectors and divergences using independently constructed bases."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh.unit_square(2)
    coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, mesh.cells, mesh.points, coordinate_element)
    space = dolfinx.fem.functionspace(domain, basix.ufl.element("RT", "triangle", degree + 1))
    field = dolfinx.fem.Function(space)

    def exact(points):
        """A degree-m+1 vector within native interpolation quadrature exactness."""
        x, y = points.T
        return np.column_stack((x ** (degree + 1) + y**degree, x**degree * y + y ** (degree + 1)))

    field.interpolate(lambda x: exact(x[:2].T).T)
    coefficients = rt_interpolate(mesh, exact, degree)
    bary, _ = triangle_quadrature(5)
    values, divergence = rt_evaluate(mesh, coefficients, degree, bary)
    physical = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
    points = np.column_stack((physical, np.zeros(len(physical))))
    original_to_local = np.argsort(domain.topology.original_cell_index)
    cells = np.repeat(original_to_local, len(bary)).astype(np.int32)
    assert_allclose(values.reshape(-1, 2), field.eval(points, cells), atol=5e-12, rtol=0)
    # Cell reference coordinates follow DOLFINx's reordered vertex orientation.
    # Compare the L2 divergence against the exact UFL derivative via interpolation
    # into DG_m, whose point values provide an independent derivative check.
    div_space = dolfinx.fem.functionspace(domain, ("DG", degree))
    div_field = dolfinx.fem.Function(div_space)
    trial, test = ufl.TrialFunction(div_space), ufl.TestFunction(div_space)
    projection = from_ufl(
        trial * test * ufl.dx, ufl.div(field) * test * ufl.dx, [], np.empty(0, dtype=int)
    )
    div_field.x.array[:] = projection.condense().source
    assert_allclose(divergence.ravel(), div_field.eval(points, cells).ravel(), atol=1e-11, rtol=0)
