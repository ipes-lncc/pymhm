"""Native UFL verification of current and previous-state forms on a custom mesh."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.fenics import from_ufl
from pymhm.lagrange import nodal_space
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.rad import _rad_local
from pymhm.scalar_transient import _load_map, _StepReaction


@pytest.mark.fem
@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_custom_mesh_time_residual_and_previous_mass_match_native_ufl(stabilization):
    """Independently integrate the complete backward-Euler operator and old-state load."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    macro = TriangleMesh(np.array([[0.0, 0.0], [1.2, 0.1], [0.1, 0.9]]), np.array([[0, 1, 2]]))
    vertices = macro.points
    points = np.vstack((vertices, 0.2 * vertices[0] + 0.3 * vertices[1] + 0.5 * vertices[2]))
    fine = TriangleMesh(points, np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]]))
    degree, order, dt, rho, c = 2, 6, 0.125, 1.3, 0.7
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])
    beta = np.array([0.7, 0.2])
    step_reaction = _StepReaction(c, rho, dt)
    assembled = _rad_local(
        0,
        mesh=macro,
        skeleton=SkeletonSpace(macro),
        degree=degree,
        refinement=2,
        local_meshes=(fine,),
        diffusion=tensor,
        diffusion_divergence=(0, 0),
        velocity=beta,
        velocity_divergence=0,
        reaction=step_reaction,
        source=0,
        stabilization=stabilization,
        order=order,
    )
    load_map = _load_map(
        fine,
        degree,
        order,
        tensor,
        beta,
        0,
        step_reaction,
        rho,
        stabilization,
        np.empty(0, dtype=np.int64),
    )
    _, nodes = nodal_space(fine, degree)
    old = 1 + nodes[:, 0] ** 2 + 2 * nodes[:, 1]
    actual_load = load_map.load(lambda x: 1 + x[:, 0] + x[:, 1] ** 2, old, dt, 0)
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            "triangle",
            degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        ),
    )
    x = ufl.SpatialCoordinate(domain)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    native_beta, native_tensor = ufl.as_vector(beta), ufl.as_matrix(tensor)
    adv_u, adv_v = ufl.dot(native_beta, ufl.grad(u)), ufl.dot(native_beta, ufl.grad(v))
    reaction = c + rho / dt
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    operator = ufl.inner(native_tensor * ufl.grad(u), ufl.grad(v))
    operator += (adv_u * v - u * adv_v) / 2 + reaction * u * v
    test = v
    if stabilization == "supg":
        h = ufl.CellDiameter(domain)
        largest = float(np.linalg.eigvalsh(tensor)[-1])
        tau = 1 / ufl.sqrt(
            (2 * np.linalg.norm(beta) / h) ** 2 + (4 * largest / h**2) ** 2 + reaction**2
        )
        operator += tau * (-ufl.div(native_tensor * ufl.grad(u)) + adv_u + reaction * u) * adv_v
        test += tau * adv_v
    force = 1 + x[0] + x[1] ** 2 + rho * (1 + x[0] ** 2 + 2 * x[1]) / dt
    native = from_ufl(operator * dx, force * test * dx, [], np.empty(0, dtype=int))
    coordinates = space.tabulate_dof_coordinates()[:, :2]
    distance = np.linalg.norm(nodes[:, None] - coordinates[None], axis=2)
    permutation = np.argmin(distance, axis=1)
    assert len(np.unique(permutation)) == len(nodes)
    assert distance[np.arange(len(nodes)), permutation].max() < 3e-14
    assert assembled.metadata[0] is fine
    assert_allclose(
        native.matrix.toarray()[np.ix_(permutation, permutation)],
        assembled.problem.matrix.toarray(),
        atol=3e-13,
        rtol=5e-13,
    )
    assert_allclose(native.load[permutation], actual_load, atol=4e-14, rtol=5e-13)
