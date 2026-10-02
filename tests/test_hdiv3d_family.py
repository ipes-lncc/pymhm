"""Polynomial-space, moment-duality, orientation and Piola invariants in 3D."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.hdiv3d_family import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    reference_faces,
    reference_vertices,
)
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs, reference_submesh


@pytest.fixture(autouse=True)
def one_native_thread():
    """Avoid native-thread oversubscription in the small polynomial identities."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize(
    "kind,degree,size,psize",
    [("tetrahedron", 1, 18, 4), ("tetrahedron", 2, 32, 10), ("prism", 1, 27, 6)],
)
def test_dimensions_divergence_and_exact_moments(kind, degree, size, psize):
    """The full divergence equals pressure, and canonical face moments are Kronecker dual."""
    family = HDiv3DFamily(kind, degree)
    points, w = cell_quadrature(kind, 5)
    value, div, p = family.tabulate(points)
    assert family.local_size == size and family.pressure_size == psize
    assert np.linalg.matrix_rank(p.T @ (w[:, None] * div)) == psize
    assert_allclose(p @ np.linalg.lstsq(p, div, rcond=None)[0], div, atol=2e-11)
    vertices = reference_vertices(kind)
    rows = []
    for nodes in reference_faces(kind):
        x = vertices[list(nodes)]
        uv, weights = face_quadrature(len(nodes), 5)
        normal = np.cross(x[1] - x[0], x[2] - x[0])
        normal *= np.sign(normal @ (x.mean(axis=0) - vertices.mean(axis=0)))
        q = family.tabulate(face_shape(uv, len(nodes)) @ x)[0] @ normal
        rows.append(face_polynomials(uv, len(nodes)).T @ (weights[:, None] * q))
    moments = np.vstack(rows)
    assert_allclose(moments, np.eye(size)[: len(moments)], atol=4e-12)
    assert_allclose(np.sum([row[0] for row in rows], axis=0), w @ div, atol=4e-12)
    eps = 1e-6
    derivative = sum(
        (
            family.tabulate(points + eps * np.eye(3)[a])[0][..., a]
            - family.tabulate(points - eps * np.eye(3)[a])[0][..., a]
        )
        / (2 * eps)
        for a in range(3)
    )
    assert_allclose(derivative, div, atol=1e-7)
    assert np.linalg.eigvalsh(np.einsum("q,qia,qja->ij", w, value, value)).min() > 0


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_piola_and_shared_normal_continuity(kind):
    """A skew affine map preserves integral moments, volume and both incident normal traces."""
    unit = AffineMixedMesh.unit_cube(kind=kind)
    jac = np.array([[2.0, 0.2, 0.1], [0.1, 1.5, 0.3], [0.0, 0.2, 0.8]])
    mesh = AffineMixedMesh(unit.points @ jac.T + [2, -3, 1], unit.cells, kind)
    family = HDiv3DFamily(kind)
    q, w = cell_quadrature(kind, 4)
    basis, div, _ = hdiv3d_basis(mesh, family, q)
    ub, ud, _ = hdiv3d_basis(unit, family, q)
    assert_allclose(basis, np.einsum("ab,tqib->tqia", jac, ub) / np.linalg.det(jac), atol=2e-11)
    assert_allclose(div, ud / np.linalg.det(jac), atol=2e-11)
    assert_allclose(mesh.volumes.sum(), np.linalg.det(jac), rtol=1e-14)
    coefficients = np.random.default_rng(13).normal(size=int(hdiv3d_dofs(mesh, family).max()) + 1)
    for face, owners in enumerate(mesh.incidence):
        if len(owners) != 2:
            continue
        uv, _ = face_quadrature(len(mesh.faces[face]), 4)
        physical = face_shape(uv, len(mesh.faces[face])) @ mesh.points[mesh.faces[face]]
        values = []
        for cell, _ in owners:
            reference = (physical - mesh.points[mesh.cells[cell, 0]]) @ mesh.inverse[cell].T
            local = hdiv3d_basis(mesh, family, reference)[0][cell]
            values.append(
                np.einsum(
                    "qia,i,a->q",
                    local,
                    coefficients[hdiv3d_dofs(mesh, family)[cell]],
                    mesh.normals[face],
                )
            )
        assert_allclose(values[0], values[1], atol=2e-11)
    assert np.isfinite(basis).all()


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_refinement_and_vertex_reorientation(kind):
    """Local r=2 refinement preserves physical volume and normalization of reversed vertices."""
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    cells = mesh.cells.copy()
    cells[:, [1, 2]] = cells[:, [2, 1]]
    if kind == "prism":
        cells[:, [4, 5]] = cells[:, [5, 4]]
    reversed_mesh = AffineMixedMesh(mesh.points, cells, kind)
    assert_allclose(reversed_mesh.jacobian, mesh.jacobian)
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 2)
        assert len(fine.cells) == 8
        assert_allclose(fine.volumes.sum(), mesh.volumes[cell], atol=2e-15)
    reference_submesh(kind, 1)


@pytest.mark.parametrize(
    "call",
    [
        lambda: HDiv3DFamily("invalid"),
        lambda: HDiv3DFamily("prism", 1, 2),
        lambda: HDiv3DFamily("tetrahedron", 1, 3),
        lambda: HDiv3DFamily().tabulate([[1j, 0, 0]]),
        lambda: HDiv3DFamily().tabulate([[0, 0]]),
        lambda: face_shape(np.zeros((1, 2)), 5),
        lambda: face_quadrature(5, 2),
        lambda: face_polynomials(np.zeros((1, 2)), 5),
        lambda: face_polynomials(np.zeros((1, 2)), 3, -1),
        lambda: reference_submesh("tetrahedron", 3),
        lambda: AffineMixedMesh.unit_cube().submesh(6, 1),
        lambda: hdiv3d_dofs(AffineMixedMesh.unit_cube(), HDiv3DFamily("prism")),
    ],
)
def test_family_contracts(call):
    """Unsupported topology, degrees and point contracts are rejected explicitly."""
    with pytest.raises(ValueError):
        call()


def test_mesh_contracts_and_zero_degree_tests():
    """Malformed, degenerate, duplicate, overlapping and nonaffine cells fail validation."""
    mesh = AffineMixedMesh.unit_cube()
    for points, cells in [
        (mesh.points.astype(complex) + 1j, mesh.cells),
        (mesh.points, [[0, 1, 2, 2]]),
        (mesh.points, [[0, 1, 2, 3]]),
        (mesh.points, np.vstack((mesh.cells, mesh.cells[:1]))),
        (mesh.points, [[-1, 1, 2, 3]]),
        (mesh.points, mesh.cells.astype(float)),
    ]:
        with pytest.raises(ValueError):
            AffineMixedMesh(points, cells)
    prism = AffineMixedMesh.unit_cube(kind="prism")
    points = prism.points.copy()
    points[-1, 0] += 0.2
    with pytest.raises(ValueError, match="affine"):
        AffineMixedMesh(points, prism.cells, "prism")
    assert_allclose(face_polynomials(np.zeros((2, 2)), 3, 0), 1)


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_octagonal_well_partition_preserves_boundary_and_volume(kind):
    """Tetra/prism subdivisions preserve the exact faceted hexahedral reservoir domain."""
    from pymhm.hdiv3d_mesh import hdiv3d_transform
    from pymhm.mapped_rt import HexMesh, cube_quadrature

    hexa = HexMesh.annular_prism(np.geomspace(0.2, 50, 5), 10, 8)
    mesh = AffineMixedMesh.from_extruded_hexahedra(hexa.points, hexa.cells, kind)
    points, w = cube_quadrature(4)
    assert_allclose(mesh.volumes.sum(), np.sum(hexa.geometry(points)[2] * w), rtol=3e-15)
    refined = mesh.refined(2)
    assert_allclose(refined.volumes.sum(), mesh.volumes.sum(), rtol=3e-15)
    assert len(refined.cells) == 8 * len(mesh.cells)
    for face in refined.boundary_faces:
        x = refined.points[refined.faces[face]]
        assert any(
            np.max(abs((x - mesh.points[mesh.faces[parent][0]]) @ mesh.normals[parent])) < 1e-11
            for parent in mesh.boundary_faces
        )
    family = HDiv3DFamily(kind)
    unit = AffineMixedMesh.unit_cube(kind=kind)
    reference, _ = cell_quadrature(kind, 2)
    transform = hdiv3d_transform(unit, family)
    assert_allclose(
        hdiv3d_basis(unit, family, reference, transform=transform)[0],
        hdiv3d_basis(unit, family, reference)[0],
    )
    with pytest.raises(ValueError):
        AffineMixedMesh.from_extruded_hexahedra(hexa.points, hexa.cells[:, :7], kind)


def test_topological_nonmanifold_and_overlap_rejection():
    """A shared facet cannot have three owners or owners on its same physical side."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.1, 0.1, 2.0],
        ]
    )
    with pytest.raises(ValueError, match="nonmanifold"):
        AffineMixedMesh(points, np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]]))
    with pytest.raises(ValueError, match="overlap"):
        AffineMixedMesh(points, np.array([[0, 1, 2, 3], [0, 1, 2, 5]]))


def test_polynomial_dimension_diagnostic(monkeypatch):
    """A failed constraint rank is rejected before a malformed family can be used."""
    import pymhm.hdiv3d_family as module

    module._coefficients.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(module, "null_space", lambda matrix, **kwargs: np.zeros((matrix.shape[1], 0)))
        with pytest.raises(ArithmeticError, match="dimension"):
            _ = HDiv3DFamily().local_size
    module._coefficients.cache_clear()


@pytest.mark.parametrize("kind,degree", [("tetrahedron", 1), ("tetrahedron", 2), ("prism", 1)])
def test_canonical_interior_moments_and_orthogonality(kind, degree):
    """Fixed seed moments determine positive-Cholesky bubbles and minimum-norm face lifts."""
    family = HDiv3DFamily(kind, degree)
    points, weights = cell_quadrature(kind, degree + 4)
    values = family.tabulate(points)[0]
    boundary = sum(family.face_sizes)
    gram = np.einsum("q,qia,qja->ij", weights, values, values)
    assert_allclose(gram[boundary:, boundary:], np.eye(family.interior_size), atol=3e-11)
    assert_allclose(gram[:boundary, boundary:], 0, atol=3e-11)
    moments = np.stack(
        [
            np.einsum(
                "q,qi,q->i", weights, values[:, boundary:, axis], np.prod(points**exponent, axis=1)
            )
            for axis, exponent in family.interior_moment_seeds
        ]
    )
    assert_allclose(np.tril(moments, -1), 0, atol=2e-12)
    assert np.all(np.diag(moments) > 0)
    assert not family.coefficients.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        family.coefficients[0, 0] = 0


@pytest.mark.parametrize("kind,degree", [("tetrahedron", 1), ("tetrahedron", 2), ("prism", 1)])
def test_basis_coordinates_ignore_nullspace_rotations_and_blas_threads(kind, degree, monkeypatch):
    """Interior coordinates remain fixed when singular-vector frames or BLAS threads change."""
    import pymhm.hdiv3d_family as module

    module._coefficients.cache_clear()
    family = HDiv3DFamily(kind, degree)
    expected = family.coefficients.copy()
    original = module.null_space
    generator = np.random.default_rng(190)

    def rotated(matrix, **kwargs):
        """Return exactly the same nullspace in a different orthonormal coordinate frame."""
        span = original(matrix, **kwargs)
        frame, _ = np.linalg.qr(generator.normal(size=(span.shape[1], span.shape[1])))
        return span @ frame

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, "null_space", rotated)
            module._coefficients.cache_clear()
            rotated_coefficients = family.coefficients.copy()
        with threadpool_limits(4):
            module._coefficients.cache_clear()
            threaded_coefficients = family.coefficients.copy()
        for actual in (rotated_coefficients, threaded_coefficients):
            assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-11
            points, _ = cell_quadrature(kind, 3)
            expected_values = family.tabulate(points, coefficients=expected)[0]
            actual_values = family.tabulate(points, coefficients=actual)[0]
            assert (
                np.linalg.norm(actual_values - expected_values) / np.linalg.norm(expected_values)
                < 2e-11
            )
    finally:
        module._coefficients.cache_clear()


@pytest.mark.parametrize("kind,degree", [("tetrahedron", 1), ("tetrahedron", 2), ("prism", 1)])
def test_archived_basis_replays_the_executed_field(kind, degree, tmp_path):
    """An archive's explicit basis preserves fields even when its bubble coordinates differ."""
    from pymhm.hdiv3d_mesh import hdiv3d_transform

    family = HDiv3DFamily(kind, degree)
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    points, _ = cell_quadrature(kind, 3)
    dofs = hdiv3d_dofs(mesh, family)
    generator = np.random.default_rng(224)
    vector = generator.normal(size=int(dofs.max()) + 1)
    expected = np.einsum("tqia,ti->tqa", hdiv3d_basis(mesh, family, points)[0], vector[dofs])
    rotation, _ = np.linalg.qr(generator.normal(size=(family.interior_size, family.interior_size)))
    coordinates = np.eye(family.local_size)
    boundary = sum(family.face_sizes)
    coordinates[boundary:, boundary:] = rotation
    stored_basis = family.coefficients @ coordinates
    stored_vector = vector.copy()
    stored_vector[dofs[:, boundary:]] = vector[dofs[:, boundary:]] @ rotation
    filename = tmp_path / "executed-field.npz"
    np.savez(filename, coefficients=stored_basis, flux=stored_vector)
    with np.load(filename) as archive:
        basis, divergence, _ = hdiv3d_basis(
            mesh, family, points, coefficients=archive["coefficients"]
        )
        actual = np.einsum("tqia,ti->tqa", basis, archive["flux"][dofs])
        assert_allclose(actual, expected, atol=3e-11)
        expected_div = np.einsum("tqi,ti->tq", hdiv3d_basis(mesh, family, points)[1], vector[dofs])
        assert_allclose(
            np.einsum("tqi,ti->tq", divergence, archive["flux"][dofs]), expected_div, atol=3e-11
        )
        wrong = np.einsum(
            "tqia,ti->tqa", hdiv3d_basis(mesh, family, points)[0], archive["flux"][dofs]
        )
        assert np.linalg.norm(wrong - expected) > 0.1 * np.linalg.norm(expected)
        custom = hdiv3d_transform(mesh, family, coefficients=archive["coefficients"])
        assert custom is not hdiv3d_transform(mesh, family)
        assert not custom.flags.writeable


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_archived_basis_contract(kind):
    """Replay rejects nonfinite, complex and incorrectly shaped coefficient matrices."""
    family = HDiv3DFamily(kind)
    for matrix in (
        family.coefficients[:-1],
        family.coefficients.astype(complex),
        family.coefficients * np.nan,
    ):
        with pytest.raises(ValueError, match="archived basis"):
            family.tabulate(np.zeros((1, 3)), coefficients=matrix)
