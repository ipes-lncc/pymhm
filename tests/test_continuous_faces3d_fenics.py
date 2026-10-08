"""Native UFL pairing and an original MHM saddle for C0 piecewise macroface traces."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_operators
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.meshes.tetrahedron import TetraMesh


def _source(points: np.ndarray) -> np.ndarray:
    """Specify the same polynomial volume data for two independent assemblers."""
    return 1 + points[:, 0] - 2 * points[:, 1] + points[:, 2] ** 2


def _boundary(points: np.ndarray) -> np.ndarray:
    """Return nonhomogeneous pressure prescribed weakly on exterior macrofaces."""
    return 0.25 + points[:, 0] - points[:, 1]


def _native_local(
    macro: TetraMesh,
    cell: int,
    fine: TetraMesh,
    skeleton: TriangularSkeleton,
    nonzero_boundary: bool,
    modules: tuple[Any, Any, Any, Any],
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Use native CG1 vertex hats for C0 traces and native CG2 pressure gradients.

    Trace vertices are obtained from the declared geometric numbering. Facet
    normals, pressure bases, volume forms and boundary integration come from
    DOLFINx/UFL, without the package's trace tabulator or local elimination.
    """
    dolfinx, ufl, basix, mpi = modules
    domain = dolfinx.mesh.create_mesh(
        mpi.COMM_SELF,
        fine.cells,
        x=fine.points,
        e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange", "tetrahedron", 2, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    trace_space = dolfinx.fem.functionspace(domain, basix.ufl.element("Lagrange", "tetrahedron", 1))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    stiffness = dolfinx.fem.assemble_matrix(
        dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
    )
    stiffness.scatter_reverse()
    load = dolfinx.fem.assemble_vector(
        dolfinx.fem.form((1 + x[0] - 2 * x[1] + x[2] ** 2) * v * dx)
    ).array.copy()
    columns, boundary = [], []
    trace_function = dolfinx.fem.Function(trace_space)
    trace_coordinates = cKDTree(trace_space.tabulate_dof_coordinates())
    normal_constant = dolfinx.fem.Constant(domain, np.zeros(3))
    for face in macro.cell_faces[cell]:
        face = int(face)
        vertices = macro.points[macro.faces[face]]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)
        owner = int(macro.face_cells[face, 0])
        if normal @ (vertices.mean(0) - macro.points[macro.cells[owner]].mean(0)) < 0:
            normal *= -1
        normal_constant.value[:] = normal
        facets = dolfinx.mesh.locate_entities_boundary(
            domain,
            2,
            lambda points, normal=normal, origin=vertices[0]: (
                abs(normal @ (points - origin[:, None])) < 1e-12
            ),
        )
        tags = dolfinx.mesh.meshtags(
            domain, 2, np.sort(facets), np.ones(len(facets), dtype=np.int32)
        )
        ds = ufl.Measure(
            "ds", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 6}
        )
        sign = ufl.dot(normal_constant, ufl.FacetNormal(domain))
        trace_form = dolfinx.fem.form(sign * trace_function * v * ds(1))
        datum = (0.25 + x[0] - x[1]) if nonzero_boundary else 0.0
        datum_form = dolfinx.fem.form(trace_function * datum * ds(1))
        locations: dict[int, np.ndarray] = {}
        for segment, triangle in enumerate(skeleton.face_partition(face)):
            for coefficient, point in zip(
                skeleton.subtriangle_dofs(face, segment), triangle @ vertices, strict=True
            ):
                if int(coefficient) in locations:
                    assert_allclose(locations[int(coefficient)], point, atol=2e-15)
                locations[int(coefficient)] = point
        for coefficient in skeleton.dofs(face):
            distance, node = trace_coordinates.query(locations[int(coefficient)])
            assert distance < 2e-13
            trace_function.x.array[:] = 0.0
            trace_function.x.array[int(node)] = 1.0
            trace_function.x.scatter_forward()
            columns.append(dolfinx.fem.assemble_vector(trace_form).array.copy())
            boundary.append(
                dolfinx.fem.assemble_scalar(datum_form) if macro.face_cells[face, 1] < 0 else 0.0
            )
    distance, permutation = cKDTree(space.tabulate_dof_coordinates()).query(
        tetra_nodal_space(fine, 2)[1]
    )
    assert distance.max() < 2e-13
    return (
        stiffness.to_scipy().tocsc()[permutation][:, permutation],
        np.column_stack(columns)[permutation],
        load[permutation],
        np.asarray(boundary),
    )


@pytest.mark.fem
@pytest.mark.parametrize("nonzero_boundary", [False, True])
def test_native_c0_uncondensed_darcy_saddle(nonzero_boundary: bool) -> None:
    """Match an independently assembled original mixed problem across an interior face."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix")
    pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    mesh = TetraMesh(
        np.vstack((np.zeros(3), np.eye(3), [0, 0, -1])),
        np.array([[0, 1, 2, 3], [0, 2, 1, 4]]),
    )
    skeleton = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    with threadpool_limits(1):
        solution = solve_darcy_3d(
            mesh,
            skeleton=skeleton,
            degree=2,
            local_refinement=2,
            source=_source,
            dirichlet=_boundary if nonzero_boundary else 0.0,
            quadrature_order=6,
        )
        blocks, loads, entries, rows, columns = [], [], [], [], []
        boundary = np.zeros(skeleton.size)
        offset = 0
        for cell, fine in enumerate(solution.local_meshes):
            a, b, f, g = _native_local(
                mesh, cell, fine, skeleton, nonzero_boundary, (dolfinx, ufl, basix, mpi)
            )
            expected_a, _, expected_f = tetra_operators(fine, 2, source=_source, order=6)
            expected_b = tetra_trace_coupling(mesh, cell, fine, skeleton, 2)
            assert_allclose(a.toarray(), expected_a.toarray(), atol=2e-14, rtol=2e-12)
            assert_allclose(b, expected_b, atol=3e-15, rtol=2e-12)
            assert_allclose(f, expected_f, atol=2e-15, rtol=2e-12)
            blocks.append(a)
            loads.append(f)
            rows.append(np.repeat(np.arange(len(f)) + offset, b.shape[1]))
            columns.append(np.tile(skeleton.cell_dofs(cell), len(f)))
            entries.append(b.ravel())
            boundary[skeleton.cell_dofs(cell)] += g
            offset += len(f)
        coupling = sparse.coo_matrix(
            (np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
            shape=(offset, skeleton.size),
        ).tocsc()
        matrix = sparse.bmat(
            [[sparse.block_diag(blocks, format="csc"), coupling], [coupling.T, None]],
            format="csc",
        )
        rhs = np.r_[np.concatenate(loads), boundary]
        independent = splu(matrix).solve(rhs)
        assert np.linalg.norm(matrix @ independent - rhs) / np.linalg.norm(rhs) < 2e-11
        assert_allclose(
            np.concatenate(solution.pressure), independent[:offset], atol=3e-12, rtol=3e-12
        )
        assert_allclose(solution.hybrid.trace, independent[offset:], atol=3e-11, rtol=3e-11)
