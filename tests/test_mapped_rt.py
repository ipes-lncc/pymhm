"""Mapped hexahedral Piola identities, normal moments and mixed-MHM physical patches."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.mapped_rt import (
    _CORNERS,
    HexMesh,
    _modal,
    _trace_map,
    cube_quadrature,
    mapped_rt_basis,
    mapped_rt_dofs,
    solve_darcy_mapped_rt,
)


@pytest.fixture(autouse=True)
def one_thread():
    """Keep the small sparse/dense contract systems deterministic and inexpensive."""
    with threadpool_limits(1):
        yield


def warped_mesh(n=1):
    """Create an injective nonaffine trilinear distortion of a conforming cube grid."""
    mesh = HexMesh.unit_cube(n)
    p = mesh.points.copy()
    p[:, 0] *= 1 + 0.2 * p[:, 1] + 0.1 * p[:, 2]
    return HexMesh(p, mesh.cells)


def exact(points):
    """Affine physical pressure used independently of the reference interpolation."""
    return 1 + points[:, 0] + 2 * points[:, 1] - points[:, 2]


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_piola_normal_moments_and_divergence_theorem(degree):
    """Every oriented flux basis has its canonical facet moment and matching volume balance."""
    mesh = warped_mesh()
    uv, weights = cube_quadrature(degree + 4, 2)
    count = (degree + 1) ** 2
    width = 3 * (degree + 2) * count
    moments = np.zeros((6 * count, width))
    for side in range(6):
        axis, end = divmod(side, 2)
        points = np.empty((len(uv), 3))
        points[:, axis] = end
        points[:, np.arange(3) != axis] = uv
        values, _, _ = mapped_rt_basis(mesh, degree, points)
        _, J, det = mesh.geometry(points)
        normal_measure = (
            (2 * end - 1) * det[0, :, None] * np.linalg.inv(J[0]).transpose(0, 2, 1)[:, :, axis]
        )
        normal = np.einsum("qia,qa->qi", values[0], normal_measure)
        moments[side * count : (side + 1) * count] = _modal(degree, uv).T @ (
            weights[:, None] * normal
        )
    assert_allclose(moments, np.eye(width)[: 6 * count], atol=3e-14)
    points, weights = cube_quadrature(degree + 4)
    _, div, _ = mapped_rt_basis(mesh, degree, points)
    volume = np.einsum("q,qi,q->i", weights, div[0], mesh.geometry(points)[2][0])
    assert_allclose(volume, moments[::count].sum(axis=0), atol=2e-14)


def test_piola_pointwise_divergence_by_physical_finite_difference():
    """Check the Piola divergence against Cartesian differentiation on a warped cell."""
    mesh = warped_mesh()
    points = np.array([[0.2, 0.3, 0.4], [0.65, 0.2, 0.7]])
    _, J, _ = mesh.geometry(points)
    derivative = np.zeros((len(points), 36, 3, 3))
    step = 2e-6
    for axis in range(3):
        shift = np.eye(3)[axis] * step
        plus = mapped_rt_basis(mesh, 1, points + shift)[0][0]
        minus = mapped_rt_basis(mesh, 1, points - shift)[0][0]
        derivative[:, :, :, axis] = (plus - minus) / (2 * step)
    physical_gradient = np.einsum("qiaj,qjb->qiab", derivative, np.linalg.inv(J[0]))
    assert_allclose(
        np.trace(physical_gradient, axis1=2, axis2=3),
        mapped_rt_basis(mesh, 1, points)[1][0],
        rtol=2e-8,
        atol=8e-10,
    )


def test_shared_face_permutations_preserve_normal_continuity():
    """Rotate one cell's reference axes without changing its positive physical orientation."""
    mesh = HexMesh.unit_cube(2)
    cells = mesh.cells.copy()
    permutation = [np.flatnonzero(np.all(c[[1, 2, 0]] == _CORNERS, axis=1))[0] for c in _CORNERS]
    cells[1] = cells[1, permutation]
    mesh = HexMesh(mesh.points, cells)
    dofs = mapped_rt_dofs(mesh, 1)
    uv, _ = cube_quadrature(3, 2)
    for face, incident in enumerate(mesh.incidence):
        if len(incident) == 1:
            continue
        normals = []
        for cell, side in incident:
            axis, end = divmod(side, 2)
            transform = mesh.face_transforms[cell, side]
            local = (uv - transform[0]) @ np.linalg.inv(transform[1:])
            points = np.empty((len(uv), 3))
            points[:, axis] = end
            points[:, np.arange(3) != axis] = local
            basis = mapped_rt_basis(mesh, 1, points)[0][cell]
            _, J, det = mesh.geometry(points)
            area_normal = (
                (2 * end - 1)
                * det[cell, :, None]
                * np.linalg.inv(J[cell]).transpose(0, 2, 1)[:, :, axis]
            )
            indices = np.flatnonzero(np.isin(dofs[cell], 4 * face + np.arange(4)))
            normals.append(np.einsum("qia,qa->qi", basis[:, indices], area_normal))
        assert_allclose(normals[0] + normals[1], 0, atol=1e-14)


@pytest.mark.parametrize("refinement,segments", [(1, 1), (2, 1), (2, 2)])
def test_nonaffine_affine_pressure_tensor_permeability(refinement, segments):
    """The complete physical affine patch survives nonaffine maps and trace subdivision."""
    mesh = warped_mesh()
    K = np.array([[2.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 4.0]])
    solution = solve_darcy_mapped_rt(
        mesh,
        degree=1,
        local_refinement=refinement,
        subdivisions=segments,
        permeability=K,
        dirichlet=exact,
    )
    errors = solution.errors(exact, -K @ np.array([1.0, 2.0, -1.0]))
    assert errors["pressure_l2"] < 2e-13
    assert errors["flux_l2"] < 2e-12
    assert max(abs(r).max() for r in solution.equilibrium_residuals()) < 1e-13
    fine, reference = mesh.submesh(0, refinement)
    mapping = _trace_map(mesh, 0, fine, reference, solution.skeleton, 1)
    boundary = (4 * fine.boundary_faces[:, None] + np.arange(4)).ravel()
    assert_allclose(solution.flux[0][boundary], mapping @ solution.hybrid.trace, atol=2e-13)


@pytest.mark.parametrize("resolution", [1, 2])
def test_source_mixed_boundary_and_pure_neumann_gauge(resolution):
    """A quadratic pressure gives exact linear flux and tested source on affine cells."""
    mesh = HexMesh.unit_cube(resolution)
    boundary = {}
    for face in mesh.boundary_faces:
        side = mesh.incidence[face][0][1]
        boundary[int(face)] = 0.0 if side % 2 == 0 else -2.0
    solution = solve_darcy_mapped_rt(
        mesh, degree=1, local_refinement=1, source=-6.0, neumann=boundary, mean_pressure=1.0
    )
    assert solution.errors(lambda p: (p * p).sum(axis=1), lambda p: -2 * p)["flux_l2"] < 1e-12
    points, weights = cube_quadrature(5)
    integral = sum(
        np.sum(m.geometry(points)[2] * weights * solution.evaluate(c, points)[0])
        for c, m in enumerate(solution.local_meshes)
    )
    assert_allclose(integral, 1.0, atol=2e-14)
    boundary.pop(int(mesh.boundary_faces[-1]))
    solution = solve_darcy_mapped_rt(
        mesh,
        degree=1,
        local_refinement=1,
        source=-6.0,
        neumann=boundary,
        dirichlet=lambda p: (p * p).sum(axis=1),
    )
    assert solution.errors(lambda p: (p * p).sum(axis=1), lambda p: -2 * p)["flux_l2"] < 1e-12


@pytest.mark.parametrize("degree", [0, 1])
def test_full_trace_equals_global_conforming_partition(degree):
    """Changing macro grouping leaves the fully resolved conforming mixed solution invariant."""
    macro = HexMesh.unit_cube()

    def force(points):
        """A nonconstant source shared by the two algebraically equivalent partitions."""
        return 1 + points[:, 0] * points[:, 2]

    grouped = solve_darcy_mapped_rt(
        macro, degree=degree, local_refinement=2, subdivisions=2, source=force
    )
    fine = solve_darcy_mapped_rt(
        HexMesh.unit_cube(2), degree=degree, local_refinement=1, source=force
    )
    points = cube_quadrature(3)[0]
    gp, gq, gd = grouped.evaluate(0, points)
    for cell in range(8):
        p, q, d = fine.evaluate(cell, points)
        assert_allclose(gp[cell], p[0], atol=2e-13)
        assert_allclose(gq[cell], q[0], atol=2e-13)
        assert_allclose(gd[cell], d[0], atol=2e-13)


def test_input_contracts_and_nonmanifold_mesh():
    """Reject malformed geometry, unsupported reference data and incompatible skeleton choices."""
    mesh = HexMesh.unit_cube()
    for points, cells in [
        (np.zeros((3, 2)), mesh.cells),
        (mesh.points.astype(complex), mesh.cells),
        (mesh.points, np.zeros((1, 8), dtype=int)),
        (mesh.points, mesh.cells.astype(float)),
        (mesh.points, np.full((1, 8), 10, dtype=int)),
    ]:
        with pytest.raises(ValueError):
            HexMesh(points, cells)
    with pytest.raises(ValueError, match="positive Jacobian"):
        HexMesh(mesh.points * [-1, 1, 1], mesh.cells)
    with pytest.raises(ValueError, match="nonmanifold"):
        HexMesh(
            mesh.points,
            np.vstack((mesh.cells, mesh.cells[:, [4, 5, 6, 7, 0, 1, 2, 3]], mesh.cells)),
        )
    for points in [
        np.array([[1j, 0, 0]]),
        np.zeros((2, 2)),
        np.array([[np.nan, 0, 0]]),
        np.array([[1.1, 0, 0]]),
    ]:
        with pytest.raises(ValueError):
            mapped_rt_basis(mesh, 1, points)
    for options in [
        dict(trace_degree=2),
        dict(subdivisions=3),
        dict(neumann={99: 0}),
        dict(neumann={int(f): 0 for f in mesh.boundary_faces}, mean_pressure=np.nan),
    ]:
        with pytest.raises(ValueError):
            solve_darcy_mapped_rt(mesh, **options)
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy_mapped_rt(
            mesh,
            degree=0,
            local_refinement=1,
            source=1,
            neumann={int(f): 0 for f in mesh.boundary_faces},
        )


def test_annular_geometry_and_topological_refinement():
    """The polygonal reservoir volume and shared refinement follow exact geometric identities."""
    mesh = HexMesh.annular_prism(np.geomspace(0.2, 50, 5), 10)
    points, weights = cube_quadrature(4)
    expected = 4 * np.sin(np.pi / 4) * (50**2 - 0.2**2) * 10
    assert len(mesh.cells) == 32
    assert_allclose(np.sum(mesh.geometry(points)[2] * weights), expected, rtol=3e-15)
    fine = mesh.refined(2)
    assert len(fine.cells) == 256
    assert_allclose(np.sum(fine.geometry(points)[2] * weights), expected, rtol=3e-15)
    assert_allclose(np.sort(fine.points, axis=0), np.sort(np.unique(fine.points, axis=0), axis=0))
    for radii, height in [
        ([0, 1], 1),
        ([1, 1], 1),
        ([1, 2], 0),
        ([np.nan, 2], 1),
        ([1 + 1j, 2], 1),
    ]:
        with pytest.raises(ValueError):
            HexMesh.annular_prism(np.array(radii), height)


def test_physical_unit_congruence_keeps_pressure_flux_and_mean():
    """Very small permeability changes physical pressure, not the normalized mixed solution."""
    mesh = warped_mesh()
    reference = solve_darcy_mapped_rt(mesh, degree=1, local_refinement=1, dirichlet=exact)
    scaled = solve_darcy_mapped_rt(
        mesh, degree=1, local_refinement=1, permeability=1e-10, dirichlet=lambda x: 1e10 * exact(x)
    )
    points = cube_quadrature(3)[0]
    p, q, d = reference.evaluate(0, points)
    ps, qs, ds = scaled.evaluate(0, points)
    assert_allclose(ps / 1e10, p, atol=4e-14, rtol=4e-14)
    assert_allclose(qs, q, atol=2e-13, rtol=2e-13)
    assert_allclose(ds, d, atol=2e-13)
    assert max(abs(r).max() for r in scaled.equilibrium_residuals()) < 1e-13


def test_invalid_shared_face_topology_is_rejected():
    """Repeated outward facets and crossed tensor corner identifications are invalid meshes."""
    mesh = HexMesh.unit_cube()
    with pytest.raises(ValueError, match="opposite outward"):
        HexMesh(mesh.points, np.tile(mesh.cells, (2, 1)))
    with pytest.raises(ValueError, match="tensor corner"):
        HexMesh(mesh.points, np.vstack((mesh.cells, mesh.cells[:, [1, 0, 2, 3, 4, 5, 6, 7]])))


def test_physical_block_gate_rejects_corrupted_reconstruction(monkeypatch):
    """Deliberately corrupt a returned flux coefficient to verify the physical residual gate."""
    from dataclasses import replace

    from pymhm.hybrid import HybridSystem

    original = HybridSystem.solve

    def corrupted(self, **kwargs):
        """Return a native solve with one deliberately corrupted local coefficient."""
        result = original(self, **kwargs)
        fields = list(result.fields)
        fields[0] = fields[0].copy()
        fields[0][0] += 0.1
        return replace(result, fields=tuple(fields))

    monkeypatch.setattr(HybridSystem, "solve", corrupted)
    with pytest.raises(ValueError, match="physical mixed block"):
        solve_darcy_mapped_rt(HexMesh.unit_cube(), degree=0, local_refinement=1, dirichlet=1.0)


def test_anisotropic_quadrature_and_streamed_mixed_operators(monkeypatch):
    """Streaming preserves a complete independent anisotropic tensor-Gauss assembly."""
    import pymhm.mapped_rt as module

    mesh = warped_mesh()
    points, weights = cube_quadrature((5, 6, 7))
    assert_allclose(
        np.sum(weights * points[:, 0] ** 8 * points[:, 1] ** 10 * points[:, 2] ** 12),
        1 / (9 * 11 * 13),
        rtol=2e-14,
    )
    material = np.array([[2.0, 0.2, 0.1], [0.2, 3.0, -0.1], [0.1, -0.1, 1.0]])
    physical, _, det = mesh.geometry(points)
    basis, div, pressure = mapped_rt_basis(mesh, 1, points)
    expected_mass = np.einsum(
        "tq,q,tqia,ab,tqjb->tij", det, weights, basis, np.linalg.inv(material), basis
    )[0]
    expected_div = np.einsum("tq,q,qi,tqj->tij", det, weights, pressure, div)[0]
    expected_load = np.einsum("tq,q,qi,tq->ti", det, weights, pressure, physical[..., 0])[0]
    expected_mean = np.einsum("tq,q,qi->ti", det, weights, pressure)[0]
    monkeypatch.setattr(module, "_QUADRATURE_BATCH_ENTRIES", 36 * 3 * 23)
    mass, divergence, load, mean = module._operators(
        mesh, 1, material, lambda p: p[:, 0], (5, 6, 7)
    )
    ids = mapped_rt_dofs(mesh, 1)[0]
    assert_allclose(mass.toarray()[np.ix_(ids, ids)], expected_mass, atol=3e-14)
    assert_allclose(divergence.toarray()[:, ids], expected_div, atol=3e-14)
    assert_allclose(load, expected_load, atol=3e-14)
    assert_allclose(mean, expected_mean, atol=3e-14)
    result = solve_darcy_mapped_rt(
        mesh, local_refinement=1, dirichlet=exact, quadrature_order=(4, 5, 6)
    )
    assert result.quadrature_order == (4, 5, 6)
    assert result.errors(exact, np.array([-1.0, -2.0, 1.0]), (5, 6, 7))["flux_l2"] < 1e-12
    nonzero = result.errors(lambda p: exact(p) + 1, np.array([0.0, -2.0, 1.0]), (5, 6, 7))
    assert_allclose(list(nonzero.values()), np.sqrt(1.15), atol=1e-12)
    assert max(abs(v).max() for v in result.equilibrium_residuals()) < 1e-12
    with pytest.raises(ValueError, match="dimension"):
        cube_quadrature((2, 3))
    with pytest.raises(ValueError, match="three axis"):
        solve_darcy_mapped_rt(mesh, quadrature_order=(3, 4))


def test_anisotropic_refinement_geometry_and_oriented_face_contract():
    """Axiswise refinement preserves geometry and rejects mismatched tangential resolutions."""
    base = warped_mesh(2)
    fine = base.refined((2, 3, 1))
    assert len(fine.cells) == 48
    p, w = cube_quadrature(4)
    assert_allclose(np.sum(base.geometry(p)[2] * w), np.sum(fine.geometry(p)[2] * w), atol=3e-14)
    with pytest.raises(ValueError, match="three axis"):
        base.refined((2, 3))
    cells = base.cells.copy()
    permutation = [np.flatnonzero(np.all(c[[1, 2, 0]] == _CORNERS, axis=1))[0] for c in _CORNERS]
    cells[1] = cells[1, permutation]
    rotated = HexMesh(base.points, cells)
    with pytest.raises(ValueError, match="match across"):
        rotated.refined((2, 3, 1))
    assert len(rotated.refined(2).cells) == 64


def test_extruded_diagonal_problem_is_invariant_under_vertical_refinement():
    """A z-invariant physical solution is represented identically in one and two vertical layers."""
    base = HexMesh.annular_prism(np.array([0.5, 2.0]), 2.0, 3)
    one = base.refined((2, 2, 1))
    two = base.refined(2)

    def tensor(points):
        """Positive block-diagonal tensor with a separate nonconstant vertical component."""
        x, y, z = points.T
        return np.column_stack((1 + 0.3 * np.sin(x), 2 + 0.2 * np.cos(y), 1 + 0.2 * z * z))[
            :, :, None
        ] * np.eye(3)

    def pressure(points):
        """Nonaffine and z-invariant physical boundary data."""
        return 1 + points[:, 0] * points[:, 1]

    results = []
    for mesh in (one, two):
        zero = {
            int(f): 0.0
            for f in mesh.boundary_faces
            if np.ptp(mesh.points[mesh.faces[f], 2]) < 1e-13
        }
        results.append(
            solve_darcy_mapped_rt(
                mesh,
                local_refinement=1,
                permeability=tensor,
                dirichlet=pressure,
                neumann=zero,
                quadrature_order=7,
            )
        )
    points = cube_quadrature(3)[0]
    for cell in range(len(two.cells)):
        coarse_points = points.copy()
        coarse_points[:, 2] = (cell % 2 + coarse_points[:, 2]) / 2
        p, q, _ = results[1].evaluate(cell, points)
        cp, cq, _ = results[0].evaluate(cell // 2, coarse_points)
        assert_allclose(p, cp, atol=3e-12)
        assert_allclose(q, cq, atol=3e-12)
        assert_allclose(q[..., 2], 0.0, atol=3e-12)


@pytest.mark.parametrize("scale", [1.0, 1e-12])
@pytest.mark.parametrize("pure_neumann", [False, True])
def test_strict_global_refinement_preserves_physical_units_and_gauge(scale, pure_neumann):
    """Explicit global accuracy retains physical flux, pressure moments and mixed boundaries."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("the requested extended mode requires a wider long-double type")
    mesh = HexMesh.unit_cube()
    boundary = {
        int(face): (-2.0 * scale if side % 2 else 0.0)
        for face in mesh.boundary_faces
        for _, side in [mesh.incidence[face][0]]
    }
    if not pure_neumann:
        boundary.pop(int(mesh.boundary_faces[-1]))
    options = dict(
        degree=1,
        local_refinement=2,
        permeability=scale,
        source=-6.0 * scale,
        neumann=boundary,
        dirichlet=lambda points: (points * points).sum(axis=1),
        mean_pressure=1.0,
    )
    reference = solve_darcy_mapped_rt(mesh, **options)
    strict = solve_darcy_mapped_rt(
        mesh, **options, global_rtol=1e-17, global_refinement_precision="extended"
    )
    assert strict.hybrid.trace.dtype == np.dtype(np.longdouble)
    assert strict.physical_residuals.max() < 1e-10
    points = cube_quadrature(3)[0]
    rp, rq, _ = reference.evaluate(0, points)
    p, q, _ = strict.evaluate(0, points)
    assert_allclose(p, rp, atol=3e-12)
    assert_allclose(q / scale, rq / scale, atol=3e-11)
    assert max(abs(row).max() for row in strict.equilibrium_residuals()) < 1e-12 * scale
    pressure = strict.evaluate(0, cube_quadrature(4)[0])[0]
    weights = strict.local_meshes[0].geometry(cube_quadrature(4)[0])[2] * cube_quadrature(4)[1]
    assert_allclose(np.sum(weights * pressure), 1.0, atol=2e-13)


def test_global_refinement_options_fail_before_local_assembly():
    """Unsupported precision and nonpositive residual thresholds fail explicitly."""
    mesh = HexMesh.unit_cube()
    for value in (0.0, -1.0, np.nan, np.inf, 1j):
        with pytest.raises(ValueError, match="global_rtol"):
            solve_darcy_mapped_rt(mesh, global_rtol=value)
    with pytest.raises(ValueError, match="global_refinement_precision"):
        solve_darcy_mapped_rt(mesh, global_refinement_precision="arbitrary")
