"""Independent UFL assembly of every block of a three-dimensional MH²M system."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.three_field_3d import solve_mh2m_3d

pytestmark = pytest.mark.fem


def test_native_full_three_field_neumann_system():
    """Assemble stiffness, both trace pairings, source and physical volume gauge."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TetraMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.1, 0.2, 1]]), np.array([[0, 1, 2, 3]])
    )
    value = float(mesh.volumes.sum() / mesh.areas.sum())
    with threadpool_limits(1):
        result = solve_mh2m_3d(
            mesh,
            permeability=lambda x: 2 + x[:, 0],
            source=1,
            neumann={int(f): value for f in mesh.boundary_faces},
            mean_pressure=1.75,
        )
        fine = result.local_meshes[0]
        coordinate = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
        domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, x=fine.points, e=coordinate)
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange", "tetrahedron", 2, lagrange_variant=basix.LagrangeVariant.equispaced
            ),
        )
        _, points = tetra_nodal_space(fine, 2)
        native = space.tabulate_dof_coordinates()
        permutation = np.argmin(np.linalg.norm(points[:, None] - native[None], axis=2), axis=1)
        assert_allclose(native[permutation], points, atol=2e-15)
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
        ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 8})
        assembled = dolfinx.fem.assemble_matrix(
            dolfinx.fem.form((2 + x[0]) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
        )
        assembled.scatter_reverse()
        a = assembled.to_scipy().toarray()[np.ix_(permutation, permutation)]
        f = dolfinx.fem.assemble_vector(dolfinx.fem.form(v * dx)).array[permutation]
        inverse = np.linalg.inv((mesh.points[1:] - mesh.points[0]).T)
        local = ufl.dot(ufl.as_matrix(inverse), x - ufl.as_vector(mesh.points[0]))
        bary = [1 - local[0] - local[1] - local[2], *[local[i] for i in range(3)]]
        gamma_order = np.argmin(
            np.linalg.norm(result.trace_space.nodes[:, None] - mesh.points[None], axis=2), axis=1
        )
        b, d = np.zeros((len(f), 4)), np.zeros((4, 4))
        h = np.zeros(4)
        for face in mesh.boundary_faces:
            point = mesh.points[mesh.faces[face, 0]]
            mask = ufl.conditional(
                ufl.lt(
                    abs(ufl.dot(x - ufl.as_vector(point), ufl.as_vector(mesh.normals[face]))), 1e-12
                ),
                1.0,
                0.0,
            )
            column = int(np.flatnonzero(mesh.cell_faces[0] == face)[0])
            b[:, column] = dolfinx.fem.assemble_vector(dolfinx.fem.form(mask * v * ds)).array[
                permutation
            ]
            for j, node in enumerate(gamma_order):
                d[column, j] = dolfinx.fem.assemble_scalar(dolfinx.fem.form(mask * bary[node] * ds))
                h[j] -= value * d[column, j]
        full = np.block(
            [
                [a, -b, np.zeros((len(f), 4))],
                [-b.T, np.zeros((4, 4)), d],
                [np.zeros((4, len(f))), d.T, np.zeros((4, 4))],
            ]
        )
        gauge = np.r_[f, np.zeros(8)]
        full = np.block([[full, gauge[:, None]], [gauge[None], np.zeros((1, 1))]])
        rhs = np.r_[f, np.zeros(4), h, 1.75 * mesh.volumes.sum()]
        answer = np.linalg.solve(full, rhs)
        assert_allclose(a, result.local[0].stiffness.toarray(), atol=4e-13)
        assert_allclose(b, result.local[0].boundary_coupling, atol=3e-15)
        assert_allclose(d, result.local[0].trace_pairing, atol=3e-15)
        assert_allclose(answer[: len(f)], result.pressure[0], atol=2e-12)
        assert_allclose(answer[len(f) : len(f) + 4], result.conormal[0], atol=2e-12)
        assert_allclose(answer[-5:-1], result.trace, atol=2e-12)
