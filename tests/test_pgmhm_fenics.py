"""Native UFL operators in an independent uncondensed Petrov-Galerkin system."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander
from numpy.testing import assert_allclose
from scipy.linalg import block_diag

from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.petrov_galerkin import solve_pgmhm

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", [2, 3])
def test_native_uncondensed_petrov_galerkin_and_enrichment(degree):
    """Verify volume equations, face residuals and both pressures without condensation."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh.unit_square()
    ell = degree - 2
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(ell) for _ in mesh.faces))
    result = solve_pgmhm(
        mesh,
        degree=degree,
        skeleton=skeleton,
        stabilization_parameter=0.17,
        permeability=lambda x: 2 + x[:, 0],
        ellipticity_lower_bound=2,
        source=lambda x: 1 + x[:, 1] ** 3,
        dirichlet=lambda x: np.exp(x[:, 0] + x[:, 1]),
        quadrature_order=10,
    )
    matrices, loads, means, evaluators = [], [], [], []
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 1)
        coordinate = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, coordinate)
        element = basix.ufl.element(
            "Lagrange", "triangle", degree, lagrange_variant=basix.LagrangeVariant.equispaced
        )
        space = dolfinx.fem.functionspace(domain, element)
        _, points = nodal_space(fine, degree)
        native_points = space.tabulate_dof_coordinates()[:, :2]
        permutation = np.argmin(
            np.linalg.norm(points[:, None] - native_points[None], axis=2), axis=1
        )
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})
        form = dolfinx.fem.form((2 + x[0]) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
        native = dolfinx.fem.assemble_matrix(form)
        native.scatter_reverse()
        matrices.append(native.to_scipy().toarray()[np.ix_(permutation, permutation)])
        loads.append(
            dolfinx.fem.assemble_vector(dolfinx.fem.form((1 + x[1] ** 3) * v * dx)).array[
                permutation
            ]
        )
        means.append(dolfinx.fem.assemble_vector(dolfinx.fem.form(v * dx)).array[permutation])
        evaluators.append((dolfinx.fem.Function(space), permutation))
    size = len(loads[0])
    volume_size = size * len(mesh.cells)
    a = block_diag(*matrices)
    f = np.concatenate(loads)
    c = block_diag(*(weights[:, None] for weights in means))
    z = block_diag(*(np.ones((size, 1)) for _ in mesh.cells))
    inverse = np.linalg.inv(np.block([[a, c], [c.T, np.zeros((2, 2))]]))[:volume_size, :volume_size]
    b = np.zeros((volume_size, skeleton.size))
    d = np.zeros((volume_size, volume_size))
    e = np.zeros(volume_size)
    g = np.zeros(skeleton.size)
    nodes, weights = leggauss(10)
    t = (nodes + 1) / 2
    trace_basis = legvander(nodes, ell)
    for face, vertices in enumerate(mesh.faces):
        start, end = mesh.points[vertices]
        points = start + t[:, None] * (end - start)
        xyz = np.column_stack((points, np.zeros(len(points))))
        measure = weights / 2 * mesh.lengths[face]
        jump = np.zeros((len(t), volume_size))
        for cell in mesh.face_cells[face]:
            if cell < 0:
                continue
            function, permutation = evaluators[cell]
            evaluation = np.empty((len(t), size))
            for index, native_index in enumerate(permutation):
                function.x.array[:] = 0
                function.x.array[native_index] = 1
                evaluation[:, index] = function.eval(xyz, np.zeros(len(t), dtype=np.int32)).ravel()
            side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
            sign = mesh.signs[cell, side]
            ids = np.arange(cell * size, (cell + 1) * size)
            jump[:, ids] += sign * evaluation
            b[np.ix_(ids, skeleton.dofs(face))] += (
                sign * evaluation.T @ (measure[:, None] * trace_basis)
            )
        tau = 0.17 * 2 / (2 * mesh.lengths[face])
        d += tau * jump.T @ (measure[:, None] * jump)
        if mesh.face_cells[face, 1] < 0:
            prescribed = np.exp(points.sum(axis=1))
            e += tau * jump.T @ (measure * prescribed)
            g[skeleton.dofs(face)] = trace_basis.T @ (measure * prescribed)
    # Unknowns are all local nodal pressures, the physical flux trace, and
    # local equation multipliers; no condensed global matrix is reused.
    matrix = np.block(
        [
            [a, b, c],
            [b.T + b.T @ inverse @ d, np.zeros((skeleton.size, skeleton.size + 2))],
            [-z.T @ d, z.T @ b, np.zeros((2, 2))],
        ]
    )
    rhs = np.r_[f, g + b.T @ inverse @ e, z.T @ (f - e)]
    solution = np.linalg.solve(matrix, rhs)
    pressure = solution[:volume_size]
    enriched = pressure + inverse @ (d @ pressure - e)
    assert_allclose(pressure, np.concatenate(result.pressure), atol=2e-12, rtol=2e-12)
    assert_allclose(enriched, np.concatenate(result.enriched_pressure), atol=2e-12, rtol=2e-12)
    assert_allclose(
        solution[volume_size : volume_size + skeleton.size],
        result.hybrid.trace,
        atol=2e-11,
        rtol=2e-12,
    )
