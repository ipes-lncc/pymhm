"""Literal RT face/volume reconstruction and its precise continuous-test balance."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.recovery.moments import reconstruct_darcy_moments, reconstruct_flux_moments


def exact(points):
    """A smooth pressure with nontrivial fine-cell conservation defects."""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def source(points):
    """Negative Laplacian of the analytical pressure."""
    return 2 * np.pi**2 * exact(points)


def flux(points):
    """Analytical physical flux."""
    x, y = np.pi * points.T
    return -np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def solve(degree, refinement=2, resolution=2):
    """Use the stable primal k=m+2 space and aligned degree-m macro traces."""
    mesh = TriangleMesh.unit_square(resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree) for _ in mesh.faces))
    return solve_darcy(
        mesh,
        degree=degree + 2,
        skeleton=skeleton,
        source=source,
        local_refinement=refinement,
        quadrature_order=8,
    )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_cut_face_batches_match_independent_one_sided_moments(degree):
    """Cut quadrature and both material sides survive batched callback evaluation."""
    from numpy.polynomial.legendre import legvander

    from pymhm.fem.quadrature.material import cartesian_edge_quadrature, cartesian_trace_values
    from pymhm.materials.cartesian import CartesianCellField
    from pymhm.recovery.moments import _cut_face_moments

    mesh = TriangleMesh.unit_square(5)
    field = CartesianCellField(np.arange(12).reshape(3, 4) + 1.0, (1 / 3, 0.25))
    interior = np.flatnonzero(mesh.face_cells[:, 1] >= 0)

    def raw(points, owners):
        """Use unequal side values and a cubic physical field, independently of FE bases."""
        coefficient = cartesian_trace_values(field, points, mesh.points[mesh.cells[owners]].mean(1))
        return coefficient[:, None] * np.column_stack((points[:, 0] ** 3, points[:, 1] ** 2))

    actual = _cut_face_moments(mesh, interior, raw, field, degree, 5)
    expected = np.zeros_like(actual)
    for index, face in enumerate(interior):
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start
        for owner in mesh.face_cells[face]:
            points, weights, _ = cartesian_edge_quadrature(
                start, end, field, 5, interior_point=mesh.points[mesh.cells[owner]].mean(0)
            )
            parameter = (points - start) @ tangent / (tangent @ tangent)
            flux = raw(points, np.full(len(points), owner))
            expected[index] += (
                mesh.lengths[face]
                * (legvander(2 * parameter - 1, degree).T @ (weights * (flux @ mesh.normals[face])))
                / 2
            )
    assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)


def test_raw_flux_tabulation_memory_is_bounded():
    """No callback receives more than the declared 4096-point working batch."""
    from pymhm.recovery.moments import _batched_flux_values

    points = np.column_stack((np.arange(8200) / 8200, np.ones(8200)))
    owners = np.arange(8200) % 3
    calls = []

    def raw(part, indices):
        """Record each actual tabulation size while retaining incident-cell information."""
        calls.append(len(part))
        return part * (1 + indices[:, None])

    assert_allclose(_batched_flux_values(raw, points, owners), points * (1 + owners[:, None]))
    assert calls == [4096, 4096, 8]


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_continuous_moments_hold_without_fictitious_fine_cell_equilibrium(degree):
    """Proposition 4.3 concerns C0 P_m, whereas fine DG0 defects can remain nonzero."""
    solution = solve(degree)
    result = reconstruct_darcy_moments(solution, degree=degree)
    for moments in result.continuous_moment_residuals():
        assert_allclose(moments, 0, atol=2e-12, rtol=0)
    for moments in result.normal_flux_residuals():
        assert_allclose(moments, 0, atol=1e-14, rtol=0)
    assert_allclose(result.conservation_residuals(), 0, atol=2e-12, rtol=0)
    assert max(np.max(np.abs(v)) for v in result.fine_conservation_residuals()) > 1e-5
    assert result.flux_l2_error(flux) > 0
    assert result.divergence_l2_error(source) > 0
    assert result.projected_divergence_l2_error(source) > 0


@pytest.mark.parametrize("degree", [1, 2])
def test_volume_moments_match_raw_flux_independently(degree):
    """RT volume moments equal the original Pk flux for every polynomial test."""
    from pymhm.fem.scalar.triangle import tabulate

    solution = solve(degree)
    result = reconstruct_darcy_moments(solution, degree=degree)
    bary, weights = triangle_quadrature(7)
    for mesh, coefficients, pressure in zip(
        result.local_meshes, result.flux, solution.pressure, strict=True
    ):
        values, _ = rt_evaluate(mesh, coefficients, degree, bary)
        dofs, _, _, gradients, _ = tabulate(mesh, solution.degree, bary)
        raw = -np.einsum("ti,tqia->tqa", pressure[dofs], gradients)
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        for i in range(degree):
            for j in range(degree - i):
                moment = np.einsum(
                    "q,tq,tqa,t->ta",
                    weights,
                    points[..., 0] ** i * points[..., 1] ** j,
                    values - raw,
                    mesh.areas,
                )
                assert_allclose(moment, 0, atol=2e-13, rtol=0)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_normal_component_is_single_valued_across_macrofaces(degree):
    """Evaluate both physical traces, independently of boundary coefficient checks."""
    result = reconstruct_darcy_moments(solve(degree, resolution=1), degree=degree)
    traces = {}
    for mesh, coefficients in zip(result.local_meshes, result.flux, strict=True):
        for face in mesh.boundary_faces:
            edge = mesh.points[mesh.faces[face]]
            edge = edge[np.lexsort((edge[:, 1], edge[:, 0]))]
            key = tuple(edge.ravel())
            parameter = np.array([0.17, 0.43, 0.81])
            points = edge[0] + parameter[:, None] * (edge[1] - edge[0])
            cell = mesh.face_cells[face, 0]
            vertices = mesh.points[mesh.cells[cell]]
            reference = np.linalg.solve((vertices[1:] - vertices[0]).T, (points - vertices[0]).T).T
            bary = np.column_stack((1 - reference.sum(axis=1), reference))
            values, _ = rt_evaluate(mesh, coefficients, degree, bary)
            tangent = edge[1] - edge[0]
            normal = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
            traces.setdefault(key, []).append(values[cell] @ normal)
    shared = [values for values in traces.values() if len(values) == 2]
    assert len(shared) == 2
    for left, right in shared:
        assert_allclose(left, right, atol=4e-12, rtol=0)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_single_triangle_macro_has_cellwise_divergence_projection(degree):
    """With one fine triangle per macro, continuous and discontinuous tests coincide."""
    result = reconstruct_darcy_moments(solve(degree, refinement=1), degree=degree)
    assert_allclose(result.conservation_residuals(), 0, atol=2e-12)
    assert_allclose(
        result.divergence_l2_error(source),
        result.projected_divergence_l2_error(source),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_projected_divergence_converges_at_degree_plus_one(degree):
    """The continuous projection gains the expected order on three mesh levels."""
    errors = [
        reconstruct_darcy_moments(
            solve(degree, resolution=n), degree=degree
        ).projected_divergence_l2_error(source, 8)
        for n in (2, 4, 8)
    ]
    assert np.min(np.log2(np.array(errors[:-1]) / errors[1:])) > degree + 0.7


def test_one_sided_cell_evaluator_and_arithmetic_interior_average():
    """A deliberately discontinuous raw field verifies averaging and volume constraints."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh)
    local_meshes = tuple(mesh.submesh(cell, 2) for cell in range(2))

    def raw(points, cells):
        """Different constants in adjacent fine cells; interface values are explicit."""
        return np.column_stack((1 + cells, -2 * cells))

    result = reconstruct_flux_moments(
        skeleton, np.zeros(skeleton.size), local_meshes, (raw, raw), degree=2, source=0.0
    )
    for fine, coefficients in zip(local_meshes, result.flux, strict=True):
        face = np.flatnonzero(fine.face_cells[:, 1] >= 0)[0]
        first, second = fine.face_cells[face]
        expected = 0.5 * np.array([2 + first + second, -2 * (first + second)]) @ fine.normals[face]
        assert_allclose(coefficients[3 * face], expected * fine.lengths[face], atol=2e-14)
    primal = solve(1)
    overridden = reconstruct_darcy_moments(
        primal, degree=1, raw_fluxes=(raw,) * len(primal.local_meshes), quadrature_order=7
    )
    assert overridden.quadrature_order == 7


def test_generic_validation_and_nonmatching_geometry():
    """Reject under-resolved or misaligned traces and malformed macro-local inputs."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh)
    meshes = tuple(mesh.submesh(c, 2) for c in range(2))

    def raw(points, cells):
        """A harmless field for input-validation paths."""
        return np.zeros_like(points)

    trace = np.zeros(skeleton.size)
    for local, fields in (
        (meshes[:1], (raw,)),
        (meshes, (raw,)),
        (meshes, (raw, 0)),
        ((0, 0), (raw, raw)),
    ):
        with pytest.raises((ValueError, TypeError), match="local|macrocell"):
            reconstruct_flux_moments(skeleton, trace, local, fields)
    vector = SkeletonSpace(mesh, components=2)
    with pytest.raises(ValueError, match="scalar"):
        reconstruct_flux_moments(vector, np.zeros(vector.size), meshes, (raw, raw))
    for bad in (np.zeros(1), np.full(skeleton.size, np.nan), np.ones(skeleton.size, dtype=complex)):
        with pytest.raises(ValueError, match="trace"):
            reconstruct_flux_moments(skeleton, bad, meshes, (raw, raw))
    for face in (FaceSpace.uniform(2), FaceSpace((0.0, 0.500004, 1.0), (1, 1))):
        enriched = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
        with pytest.raises(ValueError, match="degree|align"):
            reconstruct_flux_moments(
                enriched, np.zeros(enriched.size), meshes, (raw, raw), degree=1
            )
    shifted = TriangleMesh(meshes[0].points + [0.001, 0.001], meshes[0].cells)
    with pytest.raises(ValueError, match="cover"):
        reconstruct_flux_moments(skeleton, trace, (shifted, meshes[1]), (raw, raw))
    holed = TriangleMesh(meshes[0].points, meshes[0].cells[[0, 2, 3]])
    with pytest.raises(ValueError, match="exactly one"):
        reconstruct_flux_moments(skeleton, trace, (holed, meshes[1]), (raw, raw))


def test_primal_wrapper_checks_formulation_and_test_space():
    """An RT order beyond primal degree cannot claim the continuous-test identity."""
    with pytest.raises(ValueError, match="primal Darcy"):
        reconstruct_darcy_moments(solve_darcy(TriangleMesh.unit_square(), formulation="mixed"))
    with pytest.raises(ValueError, match="primal degree"):
        reconstruct_darcy_moments(solve_darcy(TriangleMesh.unit_square()), degree=2)
