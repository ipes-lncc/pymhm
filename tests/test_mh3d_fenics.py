"""Independent UFL full tetrahedral Robin systems with physical Neumann data."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.robin_3d import solve_mh_3d

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("pure", [False, True])
def test_native_three_dimensional_uncondensed_boundary_system(pure):
    """Compare every volume, coupling, mass and gauge block assembled directly in UFL."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TetraMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.1, 0.2, 1]]), np.array([[0, 1, 2, 3]])
    )
    selected = list(mesh.boundary_faces if pure else mesh.boundary_faces[:1])
    value = float(mesh.volumes.sum() / mesh.areas.sum())
    natural = {int(face): value for face in selected}
    with threadpool_limits(1):
        result = solve_mh_3d(
            mesh,
            permeability=lambda x: 2 + x[:, 0],
            ellipticity_lower_bound=2,
            source=1,
            dirichlet=lambda x: np.exp(x.sum(axis=1)),
            neumann=natural,
            mean_pressure=1.75 if pure else 0,
            degree=2,
            local_refinement=2,
            skeleton=TriangularSkeleton(mesh, degree=1),
            robin_parameter=0.2,
            quadrature_order=6,
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
        x, n = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})
        ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 10})
        sigma = 0.2 * x / 3
        assembled = dolfinx.fem.assemble_matrix(
            dolfinx.fem.form(
                (2 + x[0]) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
                + ufl.dot(sigma, n) * u * v * ds
            )
        )
        assembled.scatter_reverse()
        a = assembled.to_scipy().toarray()[np.ix_(permutation, permutation)]
        f = dolfinx.fem.assemble_vector(dolfinx.fem.form(v * dx)).array[permutation]
        b = np.zeros((len(f), result.skeleton.size))
        j = np.zeros((result.skeleton.size, 3 * len(selected)))
        r = np.zeros((3 * len(selected),) * 2)
        g, h = np.zeros(result.skeleton.size), np.zeros(3 * len(selected))
        for face in mesh.boundary_faces:
            vertices = mesh.points[mesh.faces[face]]
            inverse = np.linalg.pinv((vertices[1:] - vertices[0]).T)
            local = ufl.dot(ufl.as_matrix(inverse), x - ufl.as_vector(vertices[0]))
            basis = (1 - local[0] - local[1], local[0], local[1])
            mask = ufl.conditional(
                ufl.lt(
                    abs(ufl.dot(x - ufl.as_vector(vertices[0]), ufl.as_vector(mesh.normals[face]))),
                    1e-12,
                ),
                1.0,
                0.0,
            )
            ids = result.skeleton.dofs(int(face))
            for k, phi in enumerate(basis):
                b[:, ids[k]] = dolfinx.fem.assemble_vector(
                    dolfinx.fem.form(mask * phi * v * ds)
                ).array[permutation]
                if face not in selected:
                    g[ids[k]] = dolfinx.fem.assemble_scalar(
                        dolfinx.fem.form(mask * ufl.exp(sum(x[i] for i in range(3))) * phi * ds)
                    )
                else:
                    offset = 3 * selected.index(face)
                    h[offset + k] = dolfinx.fem.assemble_scalar(
                        dolfinx.fem.form(mask * value * phi * ds)
                    )
                    for column, psi in enumerate(basis):
                        j[ids[k], offset + column] = dolfinx.fem.assemble_scalar(
                            dolfinx.fem.form(mask * phi * psi * ds)
                        )
                        r[offset + k, offset + column] = dolfinx.fem.assemble_scalar(
                            dolfinx.fem.form(mask * ufl.dot(sigma, n) * phi * psi * ds)
                        )
        nv, nl, nr = len(f), len(g), len(h)
        full = np.block(
            [
                [-a, -b, np.zeros((nv, nr))],
                [-b.T, np.zeros((nl, nl)), j],
                [np.zeros((nr, nv)), j.T, r],
            ]
        )
        rhs = np.r_[-f, -g, h]
        if pure:
            gauge = np.r_[f, np.zeros(nl + nr)]
            full = np.block([[full, gauge[:, None]], [gauge[None], np.zeros((1, 1))]])
            rhs = np.r_[rhs, 1.75 * mesh.volumes.sum()]
        coefficients = np.linalg.solve(full, rhs)
        assert_allclose(coefficients[:nv], result.pressure[0], atol=2e-11, rtol=2e-11)
        assert_allclose(
            coefficients[nv : nv + nl + nr], result.global_coefficients, atol=2e-11, rtol=2e-11
        )
        assert_allclose(a, result.system.responses[0].problem.matrix.toarray(), atol=3e-13)
        assert_allclose(b, result.system.responses[0].problem.coupling, atol=3e-15)
