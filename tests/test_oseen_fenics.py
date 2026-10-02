"""Independent UFL assembly of the complete published stabilized Oseen operator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_scientific_review import _independent_inverse_constants

from pymhm import SkeletonSpace, TriangleMesh
from pymhm.fenics import from_ufl
from pymhm.flow import _flow_local
from pymhm.lagrange import nodal_space


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_oseen_operator_and_load_against_ufl(degree: int, variable: bool) -> None:
    """Check variable transport, Hessians, grad-div, opposite adjoint signs and load."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, components=2)

    def beta(points: np.ndarray) -> np.ndarray:
        """Return a variable affine field with nonzero divergence or a constant."""
        return np.column_stack((0.2 + points[:, 0], 0.3 - points[:, 1] / 2))

    def force(points: np.ndarray) -> np.ndarray:
        """Define a polynomial load independently of the strong-residual assembly."""
        return np.column_stack((1 + points[:, 0] ** 2, -2 + points[:, 0] * points[:, 1]))

    viscosity, gamma, bound = 0.7, 1.3, 2.0
    assembly = _flow_local(
        0,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        formulation="oseen",
        refinement=2,
        viscosity=viscosity,
        drag=gamma,
        beta=beta if variable else np.array([0.2, 0.3]),
        beta_divergence=0.5 if variable else 0.0,
        beta_bound=bound,
        source=force,
        order=5,
    )
    fine = assembly.metadata[0]
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    variant = basix.LagrangeVariant.equispaced
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.mixed_element(
            [
                basix.ufl.element(
                    "Lagrange", "triangle", degree, shape=(2,), lagrange_variant=variant
                ),
                basix.ufl.element("Lagrange", "triangle", degree, lagrange_variant=variant),
            ]
        ),
    )
    constants = _independent_inverse_constants(fine, degree)
    h = fine.lengths[fine.cell_faces].max(axis=1)
    peclet_reaction = 4 * viscosity / (gamma * h**2 * constants)
    peclet_advection = constants * bound * h / (4 * viscosity)
    delta_values = h**2 / (
        gamma * h**2 * np.maximum(1, peclet_reaction)
        + 4 * viscosity / constants * np.maximum(1, peclet_advection)
    )
    kappa_values = bound * h * np.minimum(1, peclet_advection)
    cell_space = dolfinx.fem.functionspace(domain, ("DG", 0))
    delta, kappa = dolfinx.fem.Function(cell_space), dolfinx.fem.Function(cell_space)
    for cell, original in enumerate(domain.topology.original_cell_index):
        delta.x.array[cell_space.dofmap.cell_dofs(cell)] = delta_values[original]
        kappa.x.array[cell_space.dofmap.cell_dofs(cell)] = kappa_values[original]
    x = ufl.SpatialCoordinate(domain)
    transport = (
        ufl.as_vector((0.2 + x[0], 0.3 - x[1] / 2)) if variable else ufl.as_vector((0.2, 0.3))
    )
    force_ufl = ufl.as_vector((1 + x[0] ** 2, -2 + x[0] * x[1]))
    u, p = ufl.TrialFunctions(space)
    v, q = ufl.TestFunctions(space)
    trial = -viscosity * ufl.div(ufl.grad(u)) + ufl.grad(u) * transport + gamma * u + ufl.grad(p)
    test = -viscosity * ufl.div(ufl.grad(v)) - ufl.grad(v) * transport + gamma * v + ufl.grad(q)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 4})
    a = (
        viscosity * ufl.inner(ufl.grad(u), ufl.grad(v))
        + (ufl.inner(ufl.grad(u) * transport, v) - ufl.inner(u, ufl.grad(v) * transport)) / 2
        + (gamma - (0.25 if variable else 0)) * ufl.inner(u, v)
        - p * ufl.div(v)
        - q * ufl.div(u)
        + kappa * ufl.div(u) * ufl.div(v)
        - delta * ufl.inner(trial, test)
    ) * dx
    load = (ufl.inner(force_ufl, v) - delta * ufl.inner(force_ufl, test)) * dx
    independent = from_ufl(a, load, [], np.empty(0, dtype=int))
    permutation = np.empty(independent.matrix.shape[0], dtype=int)
    _, nodes = nodal_space(fine, degree)
    for component in (0, 1):
        collapsed, mapping = space.sub(component).collapse()
        coordinates = collapsed.tabulate_dof_coordinates()[:, :2]
        distance = np.linalg.norm(coordinates[:, None] - nodes[None], axis=2)
        ids = np.argmin(distance, axis=1)
        assert distance[np.arange(len(ids)), ids].max() < 2e-14
        permutation[mapping] = (
            (2 * ids[:, None] + np.arange(2)).ravel() if component == 0 else 2 * len(nodes) + ids
        )
    native = assembly.problem
    assert_allclose(
        independent.matrix.toarray(),
        native.matrix.toarray()[permutation][:, permutation],
        atol=4e-13,
        rtol=4e-13,
    )
    assert_allclose(independent.load, native.load[permutation], atol=4e-14, rtol=4e-13)
