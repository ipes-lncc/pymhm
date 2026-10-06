"""Independent UFL verification of the high-order locals used for trace convergence."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.backends.fenics import from_ufl
from pymhm.fem.scalar.triangle import nodal_space, scalar_operators
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.fem
@pytest.mark.parametrize("degree", [6, 8])
def test_high_order_scalar_local_against_native_ufl(degree):
    """Match the full anisotropic stiffness and a nonconstant polynomial load."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh([[0.1, 0.2], [1.2, 0.3], [0.2, 1.1]], [[0, 1, 2]])
    material = np.array([[2.0, 0.3], [0.3, 1.0]])
    with threadpool_limits(1):
        matrix, _, force = scalar_operators(
            mesh,
            degree,
            diffusion=material,
            source=lambda x: 1 + x[:, 0] + x[:, 1] ** 2,
            order=degree + 2,
        )
        _, nodes = nodal_space(mesh, degree)
        domain = dolfinx.mesh.create_mesh(
            MPI.COMM_SELF,
            mesh.cells,
            x=mesh.points,
            e=ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
        )
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange", "triangle", degree, lagrange_variant=basix.LagrangeVariant.equispaced
            ),
        )
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 2})
        native = from_ufl(
            ufl.inner(ufl.as_matrix(material) * ufl.grad(u), ufl.grad(v)) * dx,
            (1 + x[0] + x[1] ** 2) * v * dx,
            [],
            np.empty(0, dtype=int),
        )
    points = space.tabulate_dof_coordinates()[:, :2]
    distances = np.linalg.norm(nodes[:, None] - points[None], axis=-1)
    permutation = distances.argmin(axis=1)
    assert len(np.unique(permutation)) == len(nodes)
    assert distances.min(axis=1).max() < 5e-15
    expected = native.matrix.toarray()[np.ix_(permutation, permutation)]
    assert np.linalg.norm(matrix.toarray() - expected) / np.linalg.norm(expected) < 3e-12
    assert_allclose(force, native.load[permutation], rtol=2e-11, atol=1e-13)
