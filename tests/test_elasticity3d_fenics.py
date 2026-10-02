"""Independent UFL three-dimensional anisotropic symmetric-strain operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.darcy3d import TriangularSkeleton
from pymhm.elasticity3d import _local
from pymhm.fenics import from_ufl
from pymhm.tetrahedral import TetraMesh, tetra_nodal_space


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_general_3d_tensor_operator_against_ufl(degree: int) -> None:
    """Check Kelvin shear ordering, physical derivatives and every local matrix/load entry."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TetraMesh(
        [[0.1, 0.2, -0.1], [1.2, 0.3, 0.0], [0.2, 1.1, 0.2], [0.0, 0.1, 1.2]], [[0, 1, 2, 3]]
    )
    stiffness = np.diag([5.0, 6, 7, 2, 3, 4]) + 0.1 * np.ones((6, 6))
    assembly = _local(
        0,
        mesh=mesh,
        skeleton=TriangularSkeleton(mesh, degree=1),
        degree=degree,
        refinement=1,
        constitutive=lambda x: (2 + x[:, 0] + x[:, 1] ** 2 + x[:, 2])[:, None, None] * stiffness,
        lame_lambda=1.0,
        lame_mu=1.0,
        source=lambda x: np.column_stack((1 + x[:, 0] ** 2, 2 - x[:, 1], x[:, 0] * x[:, 2])),
        order=degree + 3,
    )
    fine = assembly.metadata[0]
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            "tetrahedron",
            degree,
            shape=(3,),
            lagrange_variant=basix.LagrangeVariant.equispaced,
        ),
    )
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    eu, ev = ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))
    ku = ufl.as_vector(
        (
            eu[0, 0],
            eu[1, 1],
            eu[2, 2],
            ufl.sqrt(2) * eu[1, 2],
            ufl.sqrt(2) * eu[0, 2],
            ufl.sqrt(2) * eu[0, 1],
        )
    )
    kv = ufl.as_vector(
        (
            ev[0, 0],
            ev[1, 1],
            ev[2, 2],
            ufl.sqrt(2) * ev[1, 2],
            ufl.sqrt(2) * ev[0, 2],
            ufl.sqrt(2) * ev[0, 1],
        )
    )
    x = ufl.SpatialCoordinate(domain)
    c = (2 + x[0] + x[1] ** 2 + x[2]) * ufl.as_matrix(stiffness)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 5})
    independent = from_ufl(
        ufl.inner(c * ku, kv) * dx,
        ufl.inner(ufl.as_vector((1 + x[0] ** 2, 2 - x[1], x[0] * x[2])), v) * dx,
        [],
        np.empty(0, dtype=int),
    )
    nodes = tetra_nodal_space(fine, degree)[1]
    distances = np.linalg.norm(space.tabulate_dof_coordinates()[:, None] - nodes[None], axis=2)
    ids = np.argmin(distances, axis=1)
    assert distances[np.arange(len(ids)), ids].max() < 2e-14
    permutation = (3 * ids[:, None] + np.arange(3)).ravel()
    assert_allclose(
        independent.matrix.toarray(),
        assembly.problem.matrix.toarray()[permutation][:, permutation],
        atol=2e-11,
        rtol=2e-11,
    )
    assert_allclose(independent.load, assembly.problem.load[permutation], atol=3e-13, rtol=3e-13)
