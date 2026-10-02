"""Independent native UFL volume/trace forms and an uncondensed 3D Darcy saddle."""

from math import factorial
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton, solve_darcy_3d, tetra_trace_coupling
from pymhm.tetrahedral import TetraMesh, tetra_nodal_space, tetra_operators


def _source(points: np.ndarray) -> np.ndarray:
    """Return a polynomial volume load integrated exactly by both assemblers."""
    return 1 + points[:, 0] - 2 * points[:, 1] + points[:, 2] ** 2


def _boundary(points: np.ndarray) -> np.ndarray:
    """Prescribe nonhomogeneous weak pressure data on every exterior face."""
    return 0.25 + points[:, 0] - points[:, 1]


def _local_ufl(
    macro: TetraMesh,
    cell: int,
    fine: TetraMesh,
    skeleton: TriangularSkeleton,
    degree: int,
    trace_degree: int,
    modules: tuple[Any, Any, Any, Any],
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Build operators using UFL gradients, native facet normals and physical Bernstein modes."""
    dolfinx, ufl, basix, mpi = modules
    domain = dolfinx.mesh.create_mesh(
        mpi.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange", "tetrahedron", degree, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 2})
    stiffness = dolfinx.fem.assemble_matrix(
        dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
    )
    stiffness.scatter_reverse()
    load = dolfinx.fem.assemble_vector(
        dolfinx.fem.form((1 + x[0] - 2 * x[1] + x[2] ** 2) * v * dx)
    ).array.copy()
    columns, boundary = [], []
    trace_function = dolfinx.fem.Function(space)
    normal_constant = dolfinx.fem.Constant(domain, np.zeros(3))
    for face in macro.cell_faces[cell]:
        vertices = macro.points[macro.faces[face]]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)
        owner = int(macro.face_cells[face, 0])
        if normal @ (vertices.mean(axis=0) - macro.points[macro.cells[owner]].mean(axis=0)) < 0:
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
            "ds",
            domain=domain,
            subdomain_data=tags,
            metadata={"quadrature_degree": 2 * degree + 2},
        )
        sign = ufl.dot(normal_constant, ufl.FacetNormal(domain))
        trace_form = dolfinx.fem.form(sign * trace_function * v * ds(1))
        datum_form = dolfinx.fem.form(trace_function * (0.25 + x[0] - x[1]) * ds(1))
        inverse = np.linalg.pinv((vertices[1:] - vertices[0]).T)
        for i in range(trace_degree, -1, -1):
            for j in range(trace_degree - i, -1, -1):
                k = trace_degree - i - j
                factor = factorial(trace_degree) / (factorial(i) * factorial(j) * factorial(k))

                def mode(
                    points: np.ndarray,
                    powers: tuple[int, int, int] = (i, j, k),
                    transform: np.ndarray = inverse,
                    origin: np.ndarray = vertices[0],
                    multiplier: float = factor,
                ) -> np.ndarray:
                    """Evaluate one physical polynomial, independent of the skeleton tabulator."""
                    coordinates = transform @ (points - origin[:, None])
                    bary = np.vstack((1 - coordinates.sum(axis=0), coordinates))
                    return multiplier * np.prod(bary ** np.asarray(powers)[:, None], axis=0)

                trace_function.interpolate(mode)
                columns.append(dolfinx.fem.assemble_vector(trace_form).array.copy())
                boundary.append(
                    dolfinx.fem.assemble_scalar(datum_form)
                    if macro.face_cells[face, 1] < 0
                    else 0.0
                )
    distance, permutation = cKDTree(space.tabulate_dof_coordinates()).query(
        tetra_nodal_space(fine, degree)[1]
    )
    assert distance.max() < 2e-13
    return (
        stiffness.to_scipy().tocsc()[permutation][:, permutation],
        np.column_stack(columns)[permutation],
        load[permutation],
        np.asarray(boundary),
    )


@pytest.mark.fem
@pytest.mark.parametrize("degree,trace_degree", [(2, 0), (3, 0), (4, 2)])
def test_native_uncondensed_darcy_saddle(degree: int, trace_degree: int) -> None:
    """Match complete weak-Dirichlet fields without reusing MHM condensation or trace forms."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix")
    pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    mesh = TetraMesh.unit_cube()
    skeleton = TriangularSkeleton(mesh, degree=trace_degree)
    with threadpool_limits(1):
        solution = solve_darcy_3d(
            mesh,
            skeleton=skeleton,
            degree=degree,
            local_refinement=2,
            source=_source,
            dirichlet=_boundary,
            quadrature_order=degree + 2,
        )
        blocks, loads, entries, rows, columns = [], [], [], [], []
        boundary = np.zeros(skeleton.size)
        offset = 0
        for cell, fine in enumerate(solution.local_meshes):
            a, b, f, g = _local_ufl(
                mesh, cell, fine, skeleton, degree, trace_degree, (dolfinx, ufl, basix, mpi)
            )
            expected_a, _, expected_f = tetra_operators(
                fine, degree, source=_source, order=degree + 2
            )
            expected_b = tetra_trace_coupling(mesh, cell, fine, skeleton, degree)
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
