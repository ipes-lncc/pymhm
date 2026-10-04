"""Native UFL checks the distinct Robin and physical-diffusive boundary forms."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.transport.rad import _rad_local
from pymhm.backends.fenics import from_ufl
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.fem
@pytest.mark.parametrize("physical_diffusion", [False, True])
def test_mixed_boundary_operator_and_nonzero_load_match_ufl(physical_diffusion):
    """Nonzero beta.n exposes an omitted half-advection mass or a flux-sign error."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    macro = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), [[0, 1, 2]])
    fine = macro.submesh(0, 3)
    skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(1) for _ in macro.faces))
    face = int(np.flatnonzero(np.all(macro.points[macro.faces, 1] == 0, axis=1))[0])
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])
    beta, reaction = np.array([0.7, 0.4]), 2.0
    assembled = _rad_local(
        0,
        mesh=macro,
        skeleton=skeleton,
        degree=2,
        refinement=3,
        diffusion=tensor,
        diffusion_divergence=(0, 0),
        velocity=beta,
        velocity_divergence=0,
        reaction=reaction,
        source=lambda x: 1 + x[:, 0],
        stabilization="galerkin",
        order=6,
        diffusive_faces=(face,) if physical_diffusion else (),
    )
    _, fixed = boundary_data(skeleton, 0, {face: lambda x: 2 + x[:, 0]}, order=6)
    coefficients = np.zeros(skeleton.size)
    for index, value in fixed.items():
        coefficients[index] = value
    actual_load = assembled.problem.load - assembled.problem.coupling @ coefficients

    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange", "triangle", 2, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    facets = dolfinx.mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0))
    facets.sort()
    tags = dolfinx.mesh.meshtags(domain, 1, facets, np.ones(len(facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 6})
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    velocity = ufl.as_vector(beta)
    operator = (
        ufl.inner(ufl.as_matrix(tensor) * ufl.grad(u), ufl.grad(v))
        + (ufl.dot(velocity, ufl.grad(u)) * v - u * ufl.dot(velocity, ufl.grad(v))) / 2
        + reaction * u * v
    ) * dx
    if physical_diffusion:
        operator += ufl.dot(velocity, normal) * u * v / 2 * ds(1)
    native = from_ufl(operator, (1 + x[0]) * v * dx - (2 + x[0]) * v * ds(1), [], [])
    nodes = nodal_space(fine, 2)[1]
    coordinates = space.tabulate_dof_coordinates()[:, :2]
    distance = np.linalg.norm(nodes[:, None] - coordinates[None], axis=2)
    permutation = np.argmin(distance, axis=1)
    assert len(np.unique(permutation)) == len(nodes)
    assert distance[np.arange(len(nodes)), permutation].max() < 3e-14
    assert_allclose(
        assembled.problem.matrix.toarray(),
        native.matrix.toarray()[np.ix_(permutation, permutation)],
        atol=4e-13,
        rtol=5e-13,
    )
    assert_allclose(actual_load, native.load[permutation], atol=4e-14, rtol=5e-13)
