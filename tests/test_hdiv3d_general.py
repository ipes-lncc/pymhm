"""General polynomial degrees, trace restriction and enriched mixed 3D invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy_hdiv3d import solve_darcy_hdiv3d
from pymhm.hdiv3d_family import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    reference_faces,
    reference_vertices,
)
from pymhm.hdiv3d_general import coefficients
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs


@pytest.fixture(autouse=True)
def one_thread():
    """Limit CI's small dense polynomial constructions to one native thread."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize(
    "kind,p,k,size",
    [
        ("tetrahedron", 0, 0, 4),
        ("tetrahedron", 0, 1, 12),
        ("tetrahedron", 1, 2, 30),
        ("tetrahedron", 2, 2, 44),
        ("tetrahedron", 3, 1, 57),
        ("tetrahedron", 3, 3, 85),
        ("prism", 0, 0, 5),
        ("prism", 2, 1, 54),
        ("prism", 2, 2, 75),
        ("prism", 3, 2, 129),
    ],
)
def test_divergence_normal_space_and_bubble_moments(kind, p, k, size):
    """Divergence is onto the pressure space and the actual normal polynomial has degree k."""
    family = HDiv3DFamily(kind, p, k)
    points, w = cell_quadrature(kind, p + 4)
    values, div, pressure = family.tabulate(points)
    assert family.local_size == size
    projected = pressure @ np.linalg.lstsq(pressure, div, rcond=None)[0]
    assert_allclose(projected, div, atol=2e-9, rtol=2e-11)
    assert np.linalg.matrix_rank(pressure.T @ (w[:, None] * div)) == family.pressure_size
    vertices = reference_vertices(kind)
    rows = []
    for face in reference_faces(kind):
        nodes = vertices[list(face)]
        uv, weights = face_quadrature(len(face), p + 4)
        normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        normal *= np.sign(normal @ (nodes.mean(axis=0) - vertices.mean(axis=0)))
        trace = family.tabulate(face_shape(uv, len(face)) @ nodes)[0] @ normal
        tests = face_polynomials(uv, len(face), k)
        assert_allclose(tests @ np.linalg.lstsq(tests, trace, rcond=None)[0], trace, atol=2e-9)
        rows.append(tests.T @ (weights[:, None] * trace))
    moments = np.vstack(rows)
    assert_allclose(moments, np.eye(size)[: len(moments)], atol=3e-11)
    gram = np.einsum("q,qia,qja->ij", w, values, values)
    boundary = sum(family.face_sizes)
    assert_allclose(gram[boundary:, boundary:], np.eye(family.interior_size), atol=3e-11)
    assert_allclose(gram[:boundary, boundary:], 0, atol=3e-11)
    with pytest.raises(ValueError, match="Nedelec"):
        _ = family.interior_moment_seeds


@pytest.mark.parametrize("corners", [3, 4])
@pytest.mark.parametrize("degree", [2, 3, 5])
def test_high_face_moments_are_orthogonal_and_vertex_regular(corners, degree):
    """A normalized face basis avoids ill-conditioned monomial moment coordinates."""
    points, weights = face_quadrature(corners, degree + 3)
    tests = face_polynomials(points, corners, degree)
    assert_allclose(
        tests.T @ (weights[:, None] * tests) / weights.sum(), np.eye(tests.shape[1]), atol=2e-13
    )
    assert np.isfinite(
        face_polynomials(np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]), corners, degree)
    ).all()


@pytest.mark.parametrize(
    "kind,p,k", [("tetrahedron", 2, 2), ("tetrahedron", 3, 3), ("prism", 2, 2)]
)
@pytest.mark.parametrize("natural", [False, True])
def test_quadratic_darcy_patch_and_physical_equilibrium(kind, p, k, natural):
    """General face orders recover nonhomogeneous data and all pressure-tested balances."""
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    bc = (
        {int(f): lambda x, n=mesh.normals[f]: -2 * x @ n for f in mesh.boundary_faces}
        if natural
        else None
    )
    solution = solve_darcy_hdiv3d(
        mesh,
        pressure_degree=p,
        normal_degree=k,
        trace_degree=k,
        local_refinement=1,
        source=-6,
        dirichlet=lambda x: np.sum(x * x, axis=1),
        neumann=bc,
        mean_pressure=1 if natural else 0,
    )
    assert max(solution.errors(lambda x: np.sum(x * x, axis=1), lambda x: -2 * x).values()) < 3e-11
    assert max(np.max(abs(r)) for r in solution.equilibrium_residuals()) < 3e-12
    assert np.max(solution.physical_residuals) < 1e-11


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_face_permutations_and_archived_basis_replay(kind):
    """Canonical normal coordinates agree on both owners and replay the executed basis."""
    family = HDiv3DFamily(kind, 2, 2)
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    dofs = hdiv3d_dofs(mesh, family)
    field = np.random.default_rng(31).normal(size=dofs.max() + 1)
    saved = family.coefficients.copy()
    coefficients.cache_clear()
    with threadpool_limits(4):
        assert_allclose(family.coefficients, saved, rtol=2e-12, atol=3e-11)
    for face, owners in enumerate(mesh.incidence):
        if len(owners) != 2:
            continue
        uv, _ = face_quadrature(len(mesh.faces[face]), 5)
        physical = face_shape(uv, len(mesh.faces[face])) @ mesh.points[mesh.faces[face]]
        traces = []
        for cell, _ in owners:
            reference = (physical - mesh.points[mesh.cells[cell, 0]]) @ mesh.inverse[cell].T
            basis = hdiv3d_basis(mesh, family, reference, coefficients=saved)[0][cell]
            traces.append(np.einsum("qia,i,a->q", basis, field[dofs[cell]], mesh.normals[face]))
        assert_allclose(*traces, atol=3e-11)
