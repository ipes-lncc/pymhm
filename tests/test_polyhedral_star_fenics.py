"""Independent UFL full-saddle verification on nonconvex polyhedral macroelements."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm import PolygonMesh, PolyhedralMesh
from pymhm._legacy.models.transport.polyhedral import (
    PolygonalSkeleton3D,
    polygonal_trace_coupling,
    solve_polyhedral_rad,
)
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_operators

pytestmark = pytest.mark.fem


def test_native_nonconvex_full_saddle():
    """Match independent local forms, original faces and the uncondensed solution."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix")
    pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    polygon = PolygonMesh(
        np.array([[0.0, 0.0], [1, 0], [1, 0.4], [0.4, 0.4], [0.4, 1], [0, 1]]),
        (np.arange(6),),
    )
    macro = PolyhedralMesh.extrude(polygon, 2)
    skeleton = PolygonalSkeleton3D(macro)
    degree = 3

    def source(points):
        """Use a polynomial source that yields a nonpolynomial discrete solution."""
        return 1 + points[:, 0] - 2 * points[:, 1] + points[:, 2] ** 2

    def boundary(points):
        """Prescribe nonhomogeneous pressure on the entire reentrant exterior."""
        return 0.25 + points[:, 0] - points[:, 1]

    with threadpool_limits(1):
        solution = solve_polyhedral_rad(
            macro,
            degree=degree,
            source=source,
            dirichlet=boundary,
            quadrature_order=6,
        )
        blocks, loads, rows, columns, entries = [], [], [], [], []
        boundary_load = np.zeros(skeleton.size)
        offset = 0
        for cell, fine in enumerate(solution.local_meshes):
            domain = dolfinx.mesh.create_mesh(
                mpi.COMM_SELF,
                fine.cells,
                fine.points,
                ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
            )
            space = dolfinx.fem.functionspace(
                domain,
                basix.ufl.element(
                    "Lagrange",
                    "tetrahedron",
                    degree,
                    lagrange_variant=basix.LagrangeVariant.equispaced,
                ),
            )
            u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
            x = ufl.SpatialCoordinate(domain)
            dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
            matrix = dolfinx.fem.assemble_matrix(
                dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
            )
            matrix.scatter_reverse()
            load = dolfinx.fem.assemble_vector(
                dolfinx.fem.form((1 + x[0] - 2 * x[1] + x[2] ** 2) * v * dx)
            ).array.copy()
            local_columns, local_boundary = [], []
            mode = dolfinx.fem.Function(space)
            for face in macro.cell_faces[cell]:
                vertices = macro.points[macro.faces[face]]
                lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                # Axis-aligned face boxes independently select the original L-prism walls.
                facets = dolfinx.mesh.locate_entities_boundary(
                    domain,
                    2,
                    lambda points, lower=lower, upper=upper: (
                        np.all(points >= lower[:, None] - 2e-13, axis=0)
                        & np.all(points <= upper[:, None] + 2e-13, axis=0)
                    ),
                )
                tags = dolfinx.mesh.meshtags(
                    domain, 2, np.sort(facets), np.ones(len(facets), dtype=np.int32)
                )
                ds = ufl.Measure(
                    "ds", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 8}
                )
                canonical = dolfinx.fem.Constant(domain, macro.normals[face].copy())
                form = dolfinx.fem.form(
                    ufl.dot(canonical, ufl.FacetNormal(domain)) * mode * v * ds(1)
                )
                boundary_form = dolfinx.fem.form(mode * (0.25 + x[0] - x[1]) * ds(1))
                for component in range(3):

                    def physical_mode(points, component=component, face=face):
                        """Evaluate the original planar affine trace in physical coordinates."""
                        if component == 0:
                            return np.ones(points.shape[1])
                        return (
                            macro.face_tangents[face, component - 1]
                            @ (points - macro.face_origins[face, :, None])
                            / np.sqrt(macro.areas[face])
                        )

                    mode.interpolate(physical_mode)
                    local_columns.append(dolfinx.fem.assemble_vector(form).array.copy())
                    local_boundary.append(
                        dolfinx.fem.assemble_scalar(boundary_form)
                        if macro.face_cells[face, 1] < 0
                        else 0.0
                    )
            distances, permutation = cKDTree(space.tabulate_dof_coordinates()).query(
                tetra_nodal_space(fine, degree)[1]
            )
            assert distances.max() < 3e-13
            a = matrix.to_scipy().tocsc()[permutation][:, permutation]
            b, f = np.column_stack(local_columns)[permutation], load[permutation]
            native_a, _, native_f = tetra_operators(fine, degree, source=source, order=6)
            native_b = polygonal_trace_coupling(macro, cell, fine, skeleton, degree)
            assert_allclose(a.toarray(), native_a.toarray(), atol=4e-13, rtol=3e-12)
            assert_allclose(b, native_b, atol=4e-15, rtol=3e-12)
            assert_allclose(f, native_f, atol=3e-15, rtol=3e-12)
            blocks.append(a)
            loads.append(f)
            rows.append(np.repeat(np.arange(len(f)) + offset, b.shape[1]))
            columns.append(np.tile(skeleton.cell_dofs(cell), len(f)))
            entries.append(b.ravel())
            boundary_load[skeleton.cell_dofs(cell)] += local_boundary
            offset += len(f)
        coupling = sparse.coo_matrix(
            (np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
            shape=(offset, skeleton.size),
        ).tocsc()
        matrix = sparse.bmat(
            [[sparse.block_diag(blocks, format="csc"), coupling], [coupling.T, None]], format="csc"
        )
        rhs = np.r_[np.concatenate(loads), boundary_load]
        independent = splu(matrix).solve(rhs)
        assert np.linalg.norm(matrix @ independent - rhs) / np.linalg.norm(rhs) < 2e-11
        assert_allclose(
            np.concatenate(solution.values), independent[:offset], atol=3e-12, rtol=3e-12
        )
        assert_allclose(solution.hybrid.trace, independent[offset:], atol=2e-11, rtol=2e-11)
