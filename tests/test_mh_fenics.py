"""Native UFL verification of the modified Robin local bilinear form."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.robin import solve_mh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", [1, 3])
def test_robin_operator_and_local_response_match_independent_ufl(degree):
    """Integrate signed sigma.n on all exterior facets using the native normal."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    coarse = TriangleMesh(np.array([[0.1, 0.2], [1.2, 0.1], [0.3, 0.9]]), np.array([[0, 1, 2]]))
    origin = np.array([-0.2, -0.3])
    result = solve_mh(
        coarse,
        degree=degree,
        local_refinement=2,
        permeability=lambda x: 2 + x[:, 0],
        ellipticity_lower_bound=2,
        source=lambda x: 1 + x[:, 1] ** 2,
        origin=origin,
        robin_parameter=0.2,
        quadrature_order=8,
    )
    fine = result.local_meshes[0]
    coordinate = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, coordinate)
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange", "triangle", degree, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    _, points = nodal_space(fine, degree)
    native_points = space.tabulate_dof_coordinates()[:, :2]
    permutation = np.argmin(np.linalg.norm(points[:, None] - native_points[None], axis=2), axis=1)
    assert_allclose(native_points[permutation], points, atol=1e-14)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
    sigma = 0.1 * (x - ufl.as_vector(origin))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 8})
    form = (2 + x[0]) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.dot(
        sigma, normal
    ) * u * v * ds
    assembled = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
    assembled.scatter_reverse()
    matrix = assembled.to_scipy().toarray()[np.ix_(permutation, permutation)]
    load = dolfinx.fem.assemble_vector(dolfinx.fem.form((1 + x[1] ** 2) * v * dx)).array[
        permutation
    ]
    response = result.system.responses[0]
    assert_allclose(matrix, response.problem.matrix.toarray(), atol=8e-13)
    assert_allclose(load, response.problem.load, atol=4e-15)
    assert_allclose(np.linalg.solve(matrix, load), response.source, atol=2e-11, rtol=3e-12)
