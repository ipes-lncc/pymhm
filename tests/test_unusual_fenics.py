"""Independent native UFL assembly of the scalar UNUSUAL operator and load."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import SkeletonSpace, TriangleMesh
from pymhm._legacy.models.transport.rad import _rad_local
from pymhm._legacy.models.transport.stabilization import UnusualParameters
from pymhm.backends.fenics import from_ufl
from pymhm.fem.scalar.triangle import nodal_space


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_complete_unusual_operator_matches_independent_ufl(degree, variable):
    """Differentiate the coefficient in UFL and check negative residuals on both sides."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh.unit_square()
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])

    def coefficient(points):
        """The sampled tensor corresponds exactly to the independently written UFL field."""
        return (1 + points[:, 0])[:, None, None] * tensor

    inverse, lower, upper = 1e-4, 0.8, 4.0
    assembly = _rad_local(
        0,
        mesh=mesh,
        skeleton=SkeletonSpace(mesh),
        degree=degree,
        refinement=2,
        diffusion=coefficient if variable else tensor,
        diffusion_divergence=tensor[:, 0] if variable else (0.0, 0.0),
        velocity=(0.0, 0.0),
        velocity_divergence=0.0,
        reaction=lambda x: 3 + x[:, 1],
        source=lambda x: 1 + x[:, 0] ** 2 + 2 * x[:, 1],
        stabilization="unusual",
        unusual_parameters=UnusualParameters(lower, upper, inverse),
        order=6,
    )
    fine = assembly.metadata[0]
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        x=fine.points,
        e=ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange", "triangle", degree, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    x = ufl.SpatialCoordinate(domain)
    material = (1 + x[0] if variable else 1) * ufl.as_matrix(tensor)
    reaction = 3 + x[1]
    force = 1 + x[0] ** 2 + 2 * x[1]
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    trial = reaction * u - ufl.div(material * ufl.grad(u))
    test = reaction * v - ufl.div(material * ufl.grad(v))
    h = ufl.CellDiameter(domain)
    tau = inverse * h**2 / (ufl.max_value(upper * inverse * h**2, 2 * lower) + 2 * lower)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 4})
    a = (
        ufl.inner(material * ufl.grad(u), ufl.grad(v)) + reaction * u * v - tau * trial * test
    ) * dx
    load = (force * v - tau * force * test) * dx
    independent = from_ufl(a, load, [], np.empty(0, dtype=int))
    _, nodes = nodal_space(fine, degree)
    coordinates = space.tabulate_dof_coordinates()[:, :2]
    distance = np.linalg.norm(coordinates[:, None] - nodes[None], axis=2)
    permutation = np.argmin(distance, axis=1)
    assert distance[np.arange(len(permutation)), permutation].max() < 3e-14
    assert_allclose(
        independent.matrix.toarray(),
        assembly.problem.matrix.toarray()[permutation][:, permutation],
        rtol=5e-13,
        atol=2e-12,
    )
    assert_allclose(independent.load, assembly.problem.load[permutation], atol=3e-14, rtol=3e-13)
