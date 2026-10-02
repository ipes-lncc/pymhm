"""Independent checks of Oswald averaging and unit-diffusion error estimation."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.estimator import estimate_darcy_error, recover_potential
from pymhm.lagrange import nodal_space


def exact(points):
    return np.sin(2 * np.pi * points[:, 0]) * np.sin(2 * np.pi * points[:, 1])


def gradient(points):
    x, y = 2 * np.pi * points.T
    return 2 * np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def source(points):
    return 8 * np.pi**2 * exact(points)


def solve(resolution=2, degree=2, trace_degree=0, refinement=2, order=10):
    mesh = TriangleMesh.unit_square(resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(trace_degree) for _ in mesh.faces))
    return solve_darcy(
        mesh,
        degree=degree,
        skeleton=skeleton,
        source=source,
        local_refinement=refinement,
        quadrature_order=order,
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_measured_bound_and_components(order):
    solution = solve()
    result = estimate_darcy_error(
        solution, degree=order, homogeneous_dirichlet=True, quadrature_order=10
    )
    error = result.energy_error(gradient, order=12)
    assert 1 < result.total / error < 10
    assert_allclose(result.total**2, result.local_squared.sum())
    assert max(result.equilibrium_defect) < 2e-11
    assert np.linalg.norm(result.divergence_defect) > 0.1
    assert np.linalg.norm(result.nonconformity) > 0.1
    assert np.linalg.norm(result.oscillation) > 1e-5
    assert result.potential.l2_error(exact, order=12) > 0
    refined_diagnostics = estimate_darcy_error(
        solution, degree=order, homogeneous_dirichlet=True, quadrature_order=12
    )
    assert_allclose(result.total, refined_diagnostics.total, rtol=1e-8)


def test_estimator_and_energy_error_decrease_with_refinement():
    results = [
        estimate_darcy_error(
            solve(n, 3, 1), degree=2, homogeneous_dirichlet=True, quadrature_order=10
        )
        for n in (1, 2, 4)
    ]
    errors = [result.energy_error(gradient, order=12) for result in results]
    assert np.all(np.diff(errors) < 0)
    assert np.all(np.diff([result.total for result in results]) < 0)
    assert np.all([result.total >= error for result, error in zip(results, errors, strict=True)])


def test_oswald_uses_fine_triangle_incidence_and_zero_boundary():
    mesh = TriangleMesh.unit_square()
    solution = solve_darcy(mesh, degree=2, local_refinement=2)
    broken = replace(
        solution,
        pressure=tuple(np.full_like(v, 1 + 4 * i) for i, v in enumerate(solution.pressure)),
    )
    recovered = recover_potential(broken, homogeneous_dirichlet=True)
    dofs, points = nodal_space(recovered.mesh, recovered.degree)
    expected = np.zeros(len(points))
    count = np.zeros(len(points))
    start = 0
    for local, value in zip(broken.local_meshes, (1.0, 5.0), strict=True):
        indices = dofs[start : start + len(local.cells)].ravel()
        np.add.at(expected, indices, value)
        np.add.at(count, indices, 1)
        start += len(local.cells)
    expected /= count
    boundary = np.any((points == 0) | (points == 1), axis=1)
    expected[boundary] = 0
    assert_allclose(recovered.values, expected, atol=0, rtol=0)
    center = np.flatnonzero(np.all(points == 0.5, axis=1))
    assert_allclose(recovered.values[center], 3.0)
    for local, values in zip(broken.local_meshes, recovered.local_values, strict=True):
        _, nodes = nodal_space(local, recovered.degree)
        for node, value in zip(nodes, values, strict=True):
            index = np.flatnonzero(np.linalg.norm(points - node, axis=1) < 1e-14)
            assert len(index) == 1
            assert value == recovered.values[index[0]]


def test_zero_and_one_element_submeshes():
    mesh = TriangleMesh.unit_square()
    zero = solve_darcy(mesh, degree=2, local_refinement=1)
    estimate = estimate_darcy_error(zero, degree=2, homogeneous_dirichlet=True)
    assert estimate.total == 0
    assert estimate.energy_error((0.0, 0.0)) == 0
    result = estimate_darcy_error(
        solve(refinement=1), degree=2, homogeneous_dirichlet=True, quadrature_order=10
    )
    assert_allclose(result.divergence_defect, 0, atol=2e-13)


def test_oswald_preserves_extended_coefficients_without_double_rounding():
    """Averaging keeps a representable sub-double perturbation and its dtype."""
    solution = solve_darcy(TriangleMesh.unit_square(), degree=2, local_refinement=2)
    value = np.longdouble(1) + 8 * np.finfo(np.longdouble).eps
    broken = replace(
        solution,
        pressure=tuple(np.full(v.shape, value, dtype=np.longdouble) for v in solution.pressure),
    )
    recovered = recover_potential(broken, homogeneous_dirichlet=True)
    _, points = nodal_space(recovered.mesh, recovered.degree)
    interior = np.all((points > 0) & (points < 1), axis=1)
    assert recovered.values.dtype == np.dtype(np.longdouble)
    assert np.all(recovered.values[interior] == value)
    assert all(v.dtype == np.dtype(np.longdouble) for v in recovered.local_values)


@pytest.mark.parametrize("material", [2.0, [[1.0, 0.0], [0.0, 2.0]], 1j, lambda x: np.ones(len(x))])
def test_no_unproved_general_material_bound(material):
    solution = replace(solve(), permeability=material)
    with pytest.raises(ValueError, match="identity diffusion"):
        estimate_darcy_error(solution, homogeneous_dirichlet=True)
    identity = replace(solution, permeability=np.eye(2))
    assert estimate_darcy_error(identity, homogeneous_dirichlet=True).total > 0


def test_boundary_and_space_restrictions():
    solution = solve()
    with pytest.raises(ValueError, match="homogeneous_dirichlet"):
        recover_potential(solution, homogeneous_dirichlet=False)
    with pytest.raises(ValueError, match="primal"):
        recover_potential(replace(solution, formulation="mixed"), homogeneous_dirichlet=True)
    with pytest.raises(ValueError, match="k>=ell"):
        estimate_darcy_error(replace(solution, degree=1), homogeneous_dirichlet=True)
    with pytest.raises(ValueError, match="ell<=m"):
        estimate_darcy_error(solve(degree=3, trace_degree=1), degree=0, homogeneous_dirichlet=True)
    mesh = TriangleMesh.unit_square()
    nonzero = solve_darcy(mesh, degree=2, dirichlet=1.0)
    with pytest.raises(ValueError, match="boundary moments"):
        estimate_darcy_error(nonzero, homogeneous_dirichlet=True)
    with pytest.raises(ValueError, match="quadrature_order"):
        estimate_darcy_error(solution, homogeneous_dirichlet=True, quadrature_order=1)


def test_nonconforming_fine_mesh_is_rejected_before_averaging():
    solution = solve(resolution=1)
    other = solution.skeleton.mesh.submesh(1, 3)
    broken = replace(solution, local_meshes=(solution.local_meshes[0], other))
    with pytest.raises(ValueError, match="globally conforming"):
        recover_potential(broken, homogeneous_dirichlet=True)


def test_corrupted_source_moments_are_not_reported_as_an_error_bound():
    solution = solve()
    inconsistent = replace(solution, source=lambda x: source(x) + 1.0)
    with pytest.raises(ValueError, match="continuous-test equilibrium"):
        estimate_darcy_error(inconsistent, homogeneous_dirichlet=True)


def test_exact_triangle_bubble_has_vanishing_estimators():
    mesh = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    solution = solve_darcy(
        mesh,
        degree=4,
        skeleton=skeleton,
        local_refinement=1,
        source=lambda x: 2 * (x[:, 0] + x[:, 1]),
        quadrature_order=8,
    )
    estimate = estimate_darcy_error(solution, degree=2, homogeneous_dirichlet=True)
    assert estimate.total < 3e-13
    assert estimate.potential.l2_error(lambda x: x[:, 0] * x[:, 1] * (1 - x.sum(axis=1))) < 2e-14


def test_close_distinct_vertices_are_rejected_instead_of_merged():
    solution = solve(resolution=1)
    fine = solution.local_meshes[0]
    malformed = TriangleMesh(np.r_[fine.points, fine.points[[0]] + [1e-16, 0]], fine.cells)
    with pytest.raises(ValueError, match="too close"):
        recover_potential(
            replace(solution, local_meshes=(malformed, solution.local_meshes[1])),
            homogeneous_dirichlet=True,
        )
