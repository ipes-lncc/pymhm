"""Native UFL restriction onto the unchanged unfitted P2 PGMHM space."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.triangle import nodal_space
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.petrov_galerkin import solve_pgmhm

pytestmark = pytest.mark.fem


@pytest.fixture(autouse=True)
def one_thread():
    """Avoid native oversubscription in the small independent variational system."""
    with threadpool_limits(1):
        yield


def _monomials(points):
    """Evaluate complete total-degree-two physical monomials."""
    x, y = points.T
    return np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y))


def _boundary(points):
    """Continuous affine Dirichlet data, unused on lateral Neumann sides."""
    return 1 - points[:, 1]


def test_native_unfitted_high_contrast_operator_and_mixed_boundary_fields():
    """Compare the entire uncondensed system and both independent pressure fields."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), np.array([[0, 1, 3], [1, 2, 3]])
    )
    material = CartesianCellField(np.array([[3.636e-7, 1394.25], [0.003, 4.0]]), (0.5, 0.5))
    neumann = {int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) > 0.5}
    alpha = 0.1
    lower = material.values.min()
    result = solve_pgmhm(
        mesh,
        stabilization_parameter=alpha,
        permeability=material,
        dirichlet=_boundary,
        neumann=neumann,
        local_refinement_precision="extended",
        quadrature_order=8,
    )
    matrices = []
    means = []
    coefficient_maps = []
    for cell in range(2):
        coarse = mesh.submesh(cell, 1)
        fine = mesh.submesh(cell, 2)
        coordinate = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, coordinate)
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange", "triangle", 2, lagrange_variant=basix.LagrangeVariant.equispaced
            ),
        )
        dg = dolfinx.fem.functionspace(domain, ("DG", 0))
        coefficient = dolfinx.fem.Function(dg)
        coefficient.interpolate(lambda x: material(x[:2].T))
        p, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
        native = dolfinx.fem.assemble_matrix(
            dolfinx.fem.form(coefficient * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx)
        )
        native.scatter_reverse()
        nodes = nodal_space(coarse, 2)[1]
        nodal_map = np.linalg.inv(_monomials(nodes))
        prolongation = _monomials(space.tabulate_dof_coordinates()[:, :2]) @ nodal_map
        matrices.append(prolongation.T @ native.to_scipy() @ prolongation)
        means.append(prolongation.T @ dolfinx.fem.assemble_vector(dolfinx.fem.form(v * dx)).array)
        coefficient_maps.append(nodal_map)
    a = block_diag(*matrices)
    c = block_diag(*(mean[:, None] for mean in means))
    z = block_diag(np.ones((6, 1)), np.ones((6, 1)))
    inverse = np.linalg.inv(np.block([[a, c], [c.T, np.zeros((2, 2))]]))[:12, :12]
    b = np.zeros((12, 5))
    d = np.zeros((12, 12))
    e = np.zeros(12)
    g = np.zeros(5)
    gauss, w = leggauss(8)
    parameter = (gauss + 1) / 2
    for face, vertices in enumerate(mesh.faces):
        start, end = mesh.points[vertices]
        points = start + parameter[:, None] * (end - start)
        measure = w / 2 * np.linalg.norm(end - start)
        jump = np.zeros((8, 12))
        for cell in mesh.face_cells[face]:
            if cell < 0:
                continue
            evaluation = _monomials(points) @ coefficient_maps[cell]
            side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
            sign = mesh.signs[cell, side]
            ids = np.arange(6 * cell, 6 * cell + 6)
            jump[:, ids] += sign * evaluation
            b[ids, face] += sign * evaluation.T @ measure
        if face in neumann:
            continue
        tau = alpha * lower / (2 * np.linalg.norm(end - start))
        d += tau * jump.T @ (measure[:, None] * jump)
        if mesh.face_cells[face, 1] < 0:
            prescribed = _boundary(points)
            e += tau * jump.T @ (measure * prescribed)
            g[face] = measure @ prescribed
    matrix = np.block(
        [
            [a, b, c],
            [b.T + b.T @ inverse @ d, np.zeros((5, 7))],
            [-z.T @ d, z.T @ b, np.zeros((2, 2))],
        ]
    )
    rhs = np.r_[np.zeros(12), g + b.T @ inverse @ e, -z.T @ e]
    free = np.setdiff1d(np.arange(19), 12 + np.array(list(neumann)))
    sol = np.zeros(19)
    sol[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs[free])
    enriched = sol[:12] + inverse @ (d @ sol[:12] - e)
    ours = np.concatenate(result.pressure).astype(float)
    ours_extra = np.concatenate(result.enriched_pressure).astype(float)
    operator = np.concatenate(
        [response.problem.matrix.toarray().ravel() for response in result.system.responses]
    )
    ufl_operator = np.concatenate([value.ravel() for value in matrices])
    assert np.linalg.norm(operator - ufl_operator) / np.linalg.norm(ufl_operator) < 2e-13
    assert np.linalg.norm(sol[:12] - ours) / np.linalg.norm(ours) < 2e-11
    assert np.linalg.norm(enriched - ours_extra) / np.linalg.norm(ours_extra) < 2e-11
    assert (
        np.linalg.norm(sol[12:17] - result.hybrid.trace) / np.linalg.norm(result.hybrid.trace)
        < 2e-11
    )
    assert np.linalg.norm((matrix @ sol - rhs)[free]) / np.linalg.norm(rhs[free]) < 1e-10
    assert max(abs(result.conservation_residuals())) < 2e-12
    # The independently assembled non-monotone coarse discretization need not
    # obey the continuous maximum principle, even when its algebraic solve passes.
    assert min(sol[:12]) < 0 < 1 < max(sol[:12])
