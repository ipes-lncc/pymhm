"""Independent UFL symmetric-strain energy for general-tensor primal elasticity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.elasticity_primal import _local_primal
from pymhm.fenics import from_ufl
from pymhm.lagrange import nodal_space


@pytest.mark.fem
@pytest.mark.parametrize(
    "degree,minimal", [(1, False), (2, False), (3, False), (4, False), (2, True), (4, True)]
)
def test_general_tensor_operator_against_independent_ufl(degree: int, minimal: bool) -> None:
    """Compare all energy/load entries, including Kelvin shear and the reduced enrichment."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh([[0.2, -0.1], [1.3, 0.2], [0.1, 0.8]], [[0, 1, 2]])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree - 1) for _ in mesh.faces), 2)
    stiffness = np.array([[5.0, 1.0, 0.4], [1.0, 4.0, 0.3], [0.4, 0.3, 2.0]])
    assembly = _local_primal(
        0,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        minimal_enrichment=minimal,
        refinement=1,
        constitutive=lambda x: (2 + x[:, 0] + x[:, 1] ** 2)[:, None, None] * stiffness,
        lame_lambda=1.0,
        lame_mu=1.0,
        source=lambda x: np.column_stack((1 + x[:, 0] ** 2, 2 - x[:, 1])),
        order=degree + 3,
    )
    fine = assembly.metadata[0]
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    full_degree = degree + int(minimal)
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            "triangle",
            full_degree,
            shape=(2,),
            lagrange_variant=basix.LagrangeVariant.equispaced,
        ),
    )
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    strain, test_strain = ufl.sym(ufl.grad(trial)), ufl.sym(ufl.grad(test))
    kelvin = ufl.as_vector((strain[0, 0], strain[1, 1], ufl.sqrt(2) * strain[0, 1]))
    test_kelvin = ufl.as_vector(
        (test_strain[0, 0], test_strain[1, 1], ufl.sqrt(2) * test_strain[0, 1])
    )
    tensor = (2 + x[0] + x[1] ** 2) * ufl.as_matrix(stiffness)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * full_degree + 4})
    independent = from_ufl(
        ufl.inner(tensor * kelvin, test_kelvin) * dx,
        ufl.inner(ufl.as_vector((1 + x[0] ** 2, 2 - x[1])), test) * dx,
        [],
        np.empty(0, dtype=int),
    )
    nodes = nodal_space(fine, full_degree)[1]
    coordinates = space.tabulate_dof_coordinates()[:, :2]
    distances = np.linalg.norm(coordinates[:, None] - nodes[None], axis=2)
    ids = np.argmin(distances, axis=1)
    assert distances[np.arange(len(ids)), ids].max() < 2e-14
    permutation = (2 * ids[:, None] + np.arange(2)).ravel()
    embedding = assembly.metadata[2]
    transform = np.eye(2 * len(nodes)) if embedding is None else embedding
    transform = transform[permutation]
    assert_allclose(
        transform.T @ independent.matrix @ transform,
        assembly.problem.matrix.toarray(),
        atol=2e-12,
        rtol=2e-12,
    )
    assert_allclose(transform.T @ independent.load, assembly.problem.load, atol=3e-13, rtol=3e-13)
