"""Independent UFL inertial elasticity operators on oblique physical simplexes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.darcy3d import TriangularSkeleton
from pymhm.elastodynamics import _make_local
from pymhm.fenics import from_ufl
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.tetrahedral import TetraMesh


@pytest.mark.fem
@pytest.mark.parametrize("dimension", [2, 3])
def test_inertial_effective_operator_source_and_mass_projection(dimension):
    """Compare every Newmark matrix entry and variable-density projection to UFL."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    if dimension == 2:
        mesh = TriangleMesh([[0.1, 0.2], [1.3, 0.3], [0.2, 1.4]], [[0, 1, 2]])
        skeleton = SkeletonSpace(mesh, components=2)
        kind = "triangle"
    else:
        mesh = TetraMesh(
            [[0.1, 0.2, -0.1], [1.3, 0.3, 0], [0.2, 1.4, 0.2], [0.1, 0.2, 1.2]], [[0, 1, 2, 3]]
        )
        skeleton = TriangularSkeleton(mesh, degree=1)
        kind = "tetrahedron"
    dt = 0.13
    local = _make_local(0, mesh, skeleton, 2, 1, 6, lambda p: 2 + p[:, 0] ** 2, None, 1.3, 0.7)
    fine = local.mesh
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", kind, 1, shape=(dimension,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            kind,
            2,
            shape=(dimension,),
            lagrange_variant=basix.LagrangeVariant.equispaced,
        ),
    )
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    rho = 2 + x[0] ** 2
    field = ufl.as_vector([1 + (j + 1) * x[j] for j in range(dimension)])
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    operator = rho * ufl.inner(u, v) + dt**2 / 4 * (
        1.4 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) + 1.3 * ufl.div(u) * ufl.div(v)
    )
    form = from_ufl(operator * dx, rho * ufl.inner(field, v) * dx, [], np.empty(0, dtype=int))
    distances = np.linalg.norm(
        space.tabulate_dof_coordinates()[:, :dimension, None].transpose(0, 2, 1)
        - local.nodes[None],
        axis=2,
    )
    ids = np.argmin(distances, axis=1)
    assert distances[np.arange(len(ids)), ids].max() < 2e-14
    permutation = (dimension * ids[:, None] + np.arange(dimension)).ravel()
    assert_allclose(
        form.matrix.toarray(),
        (local.mass + dt**2 / 4 * local.stiffness).toarray()[permutation][:, permutation],
        atol=2e-13,
        rtol=2e-13,
    )
    expected = local.load(lambda p: 1 + p * np.arange(1, dimension + 1), density_weighted=True)
    assert_allclose(form.load, expected[permutation], atol=2e-13, rtol=2e-13)
