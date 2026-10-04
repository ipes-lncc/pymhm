"""Tetrahedral geometry, polynomial integration and original 3D MHM verification."""

from math import factorial

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from simplex_native_bounds import gamma, reference_roundoff_bounds

from pymhm.darcy3d import TriangularSkeleton, _DarcyFactory, solve_darcy_3d, tetra_trace_coupling
from pymhm.tetrahedral import (
    TetraMesh,
    scalar_values_3d,
    tensor_values_3d,
    tetra_basis,
    tetra_nodal_space,
    tetra_operators,
    tetra_tabulate,
    tetrahedron_quadrature,
)


def affine(points):
    return 1 + points @ np.array([1.0, 2.0, 3.0])


def quadratic(points):
    return np.sum(points**2, axis=1)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cube_orientation_divergence_and_red_refinement(n):
    mesh = TetraMesh.unit_cube(n)
    assert len(mesh.cells) == 6 * n**3
    assert len(mesh.boundary_faces) == 12 * n * n
    np.testing.assert_allclose(mesh.volumes.sum(), 1, atol=1e-14)
    for cell, faces in enumerate(mesh.cell_faces):
        normal = mesh.signs[cell, :, None] * mesh.normals[faces]
        np.testing.assert_allclose(np.sum(normal * mesh.areas[faces, None], axis=0), 0, atol=1e-15)
    face = mesh.boundary_faces
    surface = np.sum(
        mesh.areas[face]
        * np.einsum("ij,ij->i", mesh.points[mesh.faces[face]].mean(1), mesh.normals[face])
    )
    np.testing.assert_allclose(surface, 3)
    for r in (1, 2, 4):
        fine = mesh.submesh(0, r)
        assert len(fine.cells) == r**3
        np.testing.assert_allclose(fine.volumes.sum(), mesh.volumes[0])
        assert len(fine.boundary_faces) == 4 * r * r
    assert not mesh.points.flags.writeable


def test_tetrahedron_reorientation_copy_and_validators():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mesh = TetraMesh(points, [[0, 2, 1, 3]])
    assert np.linalg.det(mesh.points[mesh.cells[0, 1:]] - mesh.points[0]) > 0
    points[0] = 100
    np.testing.assert_allclose(mesh.points[0], 0)
    with pytest.raises(ValueError, match="cell outside"):
        mesh.submesh(1)
    with pytest.raises(ValueError, match="power of two"):
        mesh.submesh(0, 3)


@pytest.mark.parametrize(
    "points,cells,message",
    [
        ([[0, 0]], [[0, 1, 2, 3]], "points"),
        ([[0, 0, 0]] * 4, [], "cells"),
        ([[0, 0, 0]] * 4, [[0.0, 1, 2, 3]], "integers"),
        ([[0, 0, 0]] * 4, [[0, 1, 2, 4]], "outside"),
        ([[0, 0, 0]] * 4, [[0, 1, 2, 3]], "degenerate"),
        (np.eye(4, 3), [[0, 1, 2, 3], [0, 3, 2, 1]], "duplicate"),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]],
            [[0, 1, 2, 3], [0, 1, 2, 4]],
            "overlap",
        ),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2], [0, 0, -1]],
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]],
            "nonmanifold",
        ),
        ([[0, 0, 1j]] * 4, [[0, 1, 2, 3]], "finite and real"),
    ],
)
def test_invalid_tetrahedra(points, cells, message):
    with pytest.raises(ValueError, match=message):
        TetraMesh(points, cells)


def test_positive_quadrature_and_cardinal_polynomials():
    bary, w = tetrahedron_quadrature(5)
    assert np.min(w) > 0
    np.testing.assert_allclose(w.sum(), 1)
    for powers in ((0, 0, 0, 0), (1, 0, 0, 0), (2, 1, 1, 0), (1, 1, 2, 2), (0, 0, 0, 6)):
        exact = 6 * np.prod([factorial(k) for k in powers]) / factorial(sum(powers) + 3)
        np.testing.assert_allclose(w @ np.prod(bary**powers, axis=1), exact, rtol=2e-13)
    mesh = TetraMesh(np.eye(4, 3), [[0, 1, 2, 3]])
    for degree in (1, 2):
        dofs, points = tetra_nodal_space(mesh, degree)
        nodes = np.column_stack((points, 1 - points.sum(axis=1)))
        values, grad = tetra_basis(degree, nodes)
        bound, _, _ = reference_roundoff_bounds("tetrahedron", degree, nodes, nodes)
        assert np.all(abs(values - np.eye(len(points))) <= bound)
        basis = tetra_basis(degree, bary)[0]
        np.testing.assert_allclose(basis.sum(axis=1), 1)
        assert len(dofs[0]) == len(points)
    with pytest.raises(ValueError, match="degree"):
        tetra_basis(0, bary)
    with pytest.raises(ValueError, match="shape"):
        tetra_basis(1, np.ones((2, 3)))
    with pytest.raises(ValueError, match="degree"):
        tetra_nodal_space(mesh, True)


@pytest.mark.parametrize("degree", [1, 2])
def test_tetra_operators_against_polynomial_integrals(degree):
    mesh = TetraMesh.unit_cube(1).submesh(0, 2)
    A, M, f = tetra_operators(mesh, degree, diffusion=np.diag([2.0, 3.0, 4.0]), source=3.0)
    dofs, nodes = tetra_nodal_space(mesh, degree)
    # Native dualized polynomial evaluation and the assembled row summation
    # have a finite cancellation error; test backward error in the original
    # matrix rather than exact zeros from a specific cardinal factor formula.
    row_terms = int(np.max(np.diff(A.tocsr().indptr)))
    assert np.all(abs(A @ np.ones(len(nodes))) <= gamma(row_terms) * (abs(A) @ np.ones(len(nodes))))
    np.testing.assert_allclose(M @ np.ones(len(nodes)) * 3, f, atol=1e-16)
    np.testing.assert_allclose(M.sum(), mesh.volumes.sum())
    p = affine(nodes)
    np.testing.assert_allclose(p @ A @ p, mesh.volumes.sum() * (2 + 12 + 36), rtol=2e-14)
    bary, w = tetrahedron_quadrature(4)
    _, _, v, g = tetra_tabulate(mesh, degree, bary)
    np.testing.assert_allclose(
        np.einsum("ti,tqij->tqj", p[dofs], g),
        np.broadcast_to([1.0, 2.0, 3.0], (len(mesh.cells), len(w), 3)),
        atol=8e-15,
    )
    np.testing.assert_allclose(
        p[dofs] @ v.T,
        affine(np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 3)).reshape(
            len(mesh.cells), -1
        ),
    )


def test_coefficient_contracts():
    points = np.zeros((5, 3))
    np.testing.assert_allclose(scalar_values_3d(lambda x: np.ones(len(x)), points), 1)
    np.testing.assert_allclose(
        tensor_values_3d(lambda x: np.ones(len(x)) * 2, points),
        np.broadcast_to(2 * np.eye(3), (5, 3, 3)),
    )
    for bad in ([1, 2], np.eye(2), -1, np.diag([1, 1, 0]), [[1, 1, 0], [0, 1, 0], [0, 0, 1]]):
        with pytest.raises(ValueError):
            tensor_values_3d(bad, points)
    with pytest.raises(ValueError, match="one value"):
        scalar_values_3d([1, 2], points)


@pytest.mark.parametrize("degree,r,s", [(1, 2, 1), (2, 2, 2), (2, 4, 2)])
def test_affine_3d_patch_anisotropic_diffusion(degree, r, s):
    mesh = TetraMesh.unit_cube(1)
    K = np.array([[2.0, 0.2, 0], [0.2, 1.0, 0.1], [0, 0.1, 3.0]])
    result = solve_darcy_3d(
        mesh,
        degree=degree,
        local_refinement=r,
        skeleton=TriangularSkeleton(mesh, s),
        permeability=K,
        dirichlet=affine,
    )
    assert result.l2_error(affine) < 2e-13
    assert result.flux_l2_error(-K @ np.array([1.0, 2.0, 3.0])) < 2e-12
    assert np.max(abs(result.conservation_residuals())) < 2e-13


def test_quadratic_p2_patch_mixed_and_pure_neumann():
    mesh = TetraMesh.unit_cube(1)
    flux = {
        int(face): float(
            -2 * np.dot(mesh.points[mesh.faces[face]].mean(axis=0), mesh.normals[face])
        )
        for face in mesh.boundary_faces
    }
    for boundary in ({next(iter(flux)): next(iter(flux.values()))}, flux):
        result = solve_darcy_3d(
            mesh, source=-6.0, dirichlet=quadratic, neumann=boundary, mean_pressure=1.0
        )
        assert result.l2_error(quadratic) < 2e-13
        assert result.flux_l2_error(lambda x: -2 * x) < 2e-12
        assert max(abs(result.conservation_residuals())) < 1e-13
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy_3d(mesh, source=1, neumann=dict.fromkeys(map(int, mesh.boundary_faces), 0.0))


def test_independent_uncondensed_3d_algebra():
    mesh = TetraMesh.unit_cube(1)
    skeleton = TriangularSkeleton(mesh)
    factory = _DarcyFactory(mesh, skeleton, 2, 2, 1.0, 1.0, 5)
    locals = [factory(i).problem for i in range(len(mesh.cells))]
    A = sparse.block_diag([p.matrix for p in locals], format="csc")
    B = np.zeros((A.shape[0], skeleton.size))
    offset = 0
    for p in locals:
        B[offset : offset + len(p.load), p.trace_dofs] = p.coupling
        offset += len(p.load)
    system = sparse.bmat([[A, sparse.csc_matrix(B)], [sparse.csc_matrix(B.T), None]], format="csc")
    values = spsolve(
        system, np.r_[np.concatenate([p.load for p in locals]), np.zeros(skeleton.size)]
    )
    result = solve_darcy_3d(mesh, source=1.0)
    np.testing.assert_allclose(np.concatenate(result.pressure), values[: A.shape[0]], atol=2e-13)
    np.testing.assert_allclose(result.hybrid.trace, values[A.shape[0] :], atol=2e-13)


def test_3d_trace_and_solver_contracts():
    mesh = TetraMesh.unit_cube(1)
    skel = TriangularSkeleton(mesh)
    with pytest.raises(ValueError, match="one integer"):
        TriangularSkeleton(mesh, [1, 2])
    for method, index in (
        (skel.dofs, -1),
        (skel.dofs, len(mesh.faces)),
        (skel.cell_dofs, len(mesh.cells)),
    ):
        with pytest.raises(ValueError):
            method(index)
    with pytest.raises(ValueError, match="supplied"):
        solve_darcy_3d(mesh, skeleton=TriangularSkeleton(TetraMesh.unit_cube(1)))
    with pytest.raises(ValueError, match="resolve"):
        solve_darcy_3d(mesh, skeleton=TriangularSkeleton(mesh, 4), local_refinement=2)
    with pytest.raises(ValueError, match="finite"):
        solve_darcy_3d(mesh, mean_pressure=np.nan)
    with pytest.raises(ValueError, match="external"):
        solve_darcy_3d(mesh, neumann={int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]): 0})
    with pytest.raises(ValueError, match="contained|align"):
        tetra_trace_coupling(mesh, 0, mesh.submesh(0, 1), TriangularSkeleton(mesh, 2), 2)
    with pytest.raises(ValueError, match="contained|align"):
        tetra_trace_coupling(mesh, 0, mesh.submesh(0, 1), TriangularSkeleton(mesh, 4), 2)
    result = solve_darcy_3d(mesh)
    with pytest.raises(ValueError, match="cell outside"):
        result.evaluate(len(mesh.cells), np.ones((1, 4)) / 4)
    with pytest.raises(ValueError, match="three components"):
        result.flux_l2_error([1, 2])


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_3d_factory_parallel_equality(backend):
    mesh = TetraMesh.unit_cube(1)
    reference = solve_darcy_3d(mesh, dirichlet=affine)
    actual = solve_darcy_3d(mesh, dirichlet=affine, backend=backend, workers=2)
    np.testing.assert_allclose(actual.pressure, reference.pressure, atol=1e-13)
    np.testing.assert_allclose(actual.hybrid.trace, reference.hybrid.trace, atol=1e-13)


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2])
def test_native_dolfinx_tetrahedral_operator(degree):
    """Compare independent UFL volume assembly and nodal ordering on distorted tetrahedra."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    mesh = TetraMesh(
        [[0.1, 0.2, 0.3], [1.3, 0.1, 0.2], [0.2, 1.1, 0.4], [0.3, 0.1, 1.4]], [[0, 1, 2, 3]]
    ).submesh(0, 2)
    geometry = ufl.Mesh(basix.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    domain = dolfinx.mesh.create_mesh(mpi.COMM_SELF, mesh.cells, mesh.points, geometry)
    space = dolfinx.fem.functionspace(domain, ("Lagrange", degree))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    K = ufl.as_matrix([[2 + x[0], 0.1, 0], [0.1, 3 + x[1], 0.2], [0, 0.2, 4 + x[2]]])
    forms = [ufl.inner(K * ufl.grad(u), ufl.grad(v)) * ufl.dx, u * v * ufl.dx]
    matrices = []
    for form in forms:
        assembled = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
        assembled.scatter_reverse()
        matrices.append(assembled.to_scipy().toarray())
    load = dolfinx.fem.assemble_vector(dolfinx.fem.form((1 + x[0] + 2 * x[1]) * v * ufl.dx)).array

    def tensor(points):
        values = np.broadcast_to(
            [[2.0, 0.1, 0], [0.1, 3, 0.2], [0, 0.2, 4]], (len(points), 3, 3)
        ).copy()
        values[:, np.arange(3), np.arange(3)] += points
        return values

    A, M, f = tetra_operators(
        mesh, degree, diffusion=tensor, source=lambda pts: 1 + pts[:, 0] + 2 * pts[:, 1]
    )
    nodes = tetra_nodal_space(mesh, degree)[1]
    from scipy.spatial import cKDTree

    distance, order = cKDTree(space.tabulate_dof_coordinates()).query(nodes)
    assert distance.max() < 1e-12
    np.testing.assert_allclose(
        A.toarray(), matrices[0][np.ix_(order, order)], atol=3e-14, rtol=1e-12
    )
    np.testing.assert_allclose(M.toarray(), matrices[1][np.ix_(order, order)], atol=1e-15)
    np.testing.assert_allclose(f, load[order], atol=1e-15)


def test_trace_coupling_rejects_external_fine_face():
    coarse = TetraMesh([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2, 3]])
    exterior = TetraMesh(2 * coarse.points, coarse.cells)
    with pytest.raises(ValueError, match="not contained"):
        tetra_trace_coupling(coarse, 0, exterior, TriangularSkeleton(coarse), 2)


def test_tetrahedral_translation_and_degree_contract():
    original = TetraMesh.unit_cube(1)
    shift = np.array([1e6, -3e6, 2e6])
    translated = TetraMesh(original.points + shift, original.cells)
    skel = TriangularSkeleton(original, 2)
    shifted_skel = TriangularSkeleton(translated, 2)
    first = tetra_trace_coupling(original, 0, original.submesh(0, 2), skel, 2)
    shifted = tetra_trace_coupling(translated, 0, translated.submesh(0, 2), shifted_skel, 2)
    np.testing.assert_allclose(first, shifted, atol=1e-16)
    for degree in (1.0, True):
        with pytest.raises(ValueError, match="integer"):
            tetra_nodal_space(original, degree)


def test_invalid_tetrahedral_nodal_degree():
    with pytest.raises(ValueError, match="degree"):
        tetra_nodal_space(TetraMesh.unit_cube(1), 0)
