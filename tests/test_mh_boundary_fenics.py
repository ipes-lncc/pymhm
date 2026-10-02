"""Independent native UFL full systems for physical MH boundary conditions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

from pymhm.lagrange import nodal_space
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.mh import solve_mh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("boundary", ["mixed", "neumann"])
def test_native_full_robin_system_with_physical_neumann_data(boundary):
    """Assemble every block independently before local/global elimination."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    natural_faces = sorted(
        mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:1]
    )
    natural = {int(f): lambda x: 1 / 3 + x[:, 0] - 0.5 for f in natural_faces}
    with threadpool_limits(1):
        result = solve_mh(
            mesh,
            degree=2,
            local_refinement=2,
            quadrature_order=8,
            permeability=lambda x: 2 + x[:, 0],
            ellipticity_lower_bound=2,
            source=lambda x: 1 + x[:, 0] ** 2,
            dirichlet=lambda x: np.exp(x[:, 0] + x[:, 1]),
            neumann=natural,
            skeleton=skeleton,
            robin_parameter=0.25,
            mean_pressure=1.75 if boundary == "neumann" else 0,
        )
        sizes = [len(p) for p in result.pressure]
        offsets = np.r_[0, np.cumsum(sizes)]
        volume = int(offsets[-1])
        count = 2 * len(natural_faces)
        coupling = np.zeros((volume, skeleton.size))
        pair = np.zeros((skeleton.size, count))
        robin = np.zeros((count, count))
        g, h = np.zeros(skeleton.size), np.zeros(count)
        operators, loads, means = [], [], []
        for cell, fine in enumerate(result.local_meshes):
            coordinate = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
            domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, coordinate)
            element = basix.ufl.element(
                "Lagrange", "triangle", 2, lagrange_variant=basix.LagrangeVariant.equispaced
            )
            space = dolfinx.fem.functionspace(domain, element)
            _, nodes = nodal_space(fine, 2)
            native_nodes = space.tabulate_dof_coordinates()[:, :2]
            permutation = np.argmin(
                np.linalg.norm(nodes[:, None] - native_nodes[None], axis=2), axis=1
            )
            assert_allclose(native_nodes[permutation], nodes, atol=1e-14)
            u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
            x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
            sigma = 0.125 * x
            dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
            ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 8})
            form = (2 + x[0]) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.dot(
                sigma, normal
            ) * u * v * ds
            assembled = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
            assembled.scatter_reverse()
            operators.append(assembled.to_scipy().toarray()[np.ix_(permutation, permutation)])
            loads.append(
                dolfinx.fem.assemble_vector(dolfinx.fem.form((1 + x[0] ** 2) * v * dx)).array[
                    permutation
                ]
            )
            means.append(dolfinx.fem.assemble_vector(dolfinx.fem.form(v * dx)).array[permutation])
            for face in mesh.cell_faces[cell]:
                a, b = mesh.points[mesh.faces[face]]
                tangent = b - a
                n = ufl.as_vector(mesh.normals[face])
                selected = ufl.conditional(
                    ufl.lt(abs(ufl.dot(x - ufl.as_vector(a), n)), 1e-12), 1.0, 0.0
                )
                t = ufl.dot(x - ufl.as_vector(a), ufl.as_vector(tangent)) / (tangent @ tangent)
                basis = (1 + 0 * t, 2 * t - 1)
                dofs = skeleton.dofs(int(face))
                for j, psi in enumerate(basis):
                    coupling[offsets[cell] : offsets[cell + 1], dofs[j]] = (
                        dolfinx.fem.assemble_vector(
                            dolfinx.fem.form(selected * ufl.dot(n, normal) * psi * v * ds)
                        ).array[permutation]
                    )
                if face not in mesh.boundary_faces:
                    continue
                if face not in natural:
                    for j, psi in enumerate(basis):
                        g[dofs[j]] = dolfinx.fem.assemble_scalar(
                            dolfinx.fem.form(selected * ufl.exp(x[0] + x[1]) * psi * ds)
                        )
                    continue
                start = 2 * natural_faces.index(face)
                for i, phi in enumerate(basis):
                    h[start + i] = dolfinx.fem.assemble_scalar(
                        dolfinx.fem.form(selected * (1 / 3 + x[0] - 0.5) * phi * ds)
                    )
                    for j, psi in enumerate(basis):
                        pair[dofs[i], start + j] = dolfinx.fem.assemble_scalar(
                            dolfinx.fem.form(selected * phi * psi * ds)
                        )
                        robin[start + i, start + j] = dolfinx.fem.assemble_scalar(
                            dolfinx.fem.form(selected * ufl.dot(sigma, normal) * phi * psi * ds)
                        )
        # Symmetric uncondensed equations in (p, lambda, rho_N).
        a = block_diag(*operators)
        full = np.block(
            [
                [-a, -coupling, np.zeros((volume, count))],
                [-coupling.T, np.zeros((skeleton.size, skeleton.size)), pair],
                [np.zeros((count, volume)), pair.T, robin],
            ]
        )
        rhs = np.r_[-np.concatenate(loads), -g, h]
        if boundary == "neumann":
            weights = np.r_[np.concatenate(means), np.zeros(skeleton.size + count)]
            full = np.block([[full, weights[:, None]], [weights[None], np.zeros((1, 1))]])
            rhs = np.r_[rhs, 1.75]
        expected = np.linalg.solve(full, rhs)
        actual = np.r_[np.concatenate(result.pressure), result.global_coefficients]
        assert_allclose(actual, expected[: len(actual)], rtol=4e-11, atol=4e-11)
        assert np.linalg.norm(full @ expected - rhs) / np.linalg.norm(rhs) < 3e-13
