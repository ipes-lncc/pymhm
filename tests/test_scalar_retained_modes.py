"""Scalar diffusion limits with physically constrained constant coarse modes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm._legacy.models.transport.solver import solve_heat, solve_transport


def affine(points):
    """Evaluate an exactly representable pressure with nonzero physical mean."""
    return 1 + points[:, 0] + 2 * points[:, 1]


@pytest.mark.parametrize("reaction", [0.0, 1e-16, 1e-12, 1e-8, 2.0])
@pytest.mark.parametrize("velocity", [(0.0, 0.0), (1e-12, -2e-12), (0.7, -0.2)])
def test_transport_affine_patch_across_nearly_null_constant_modes(reaction, velocity):
    """Keep pressure and Robin traction accurate as reaction/advection vanish."""
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))

    def source(points):
        """Supply the exact conservative transport operator on the affine field."""
        return reaction * affine(points) + velocity[0] + 2 * velocity[1]

    solution = solve_transport(
        mesh,
        reaction=reaction,
        velocity=velocity,
        source=source,
        dirichlet=affine,
        skeleton=skeleton,
        local_refinement=3,
    )
    assert solution.l2_error(affine) < 3e-12
    for local_mesh, values in zip(solution.local_meshes, solution.values, strict=True):
        assert_allclose(values, affine(local_mesh.points), rtol=0, atol=1e-11)
    for face, nodes in enumerate(mesh.faces):
        parameter = np.array([0.2, 0.8])
        points = mesh.points[nodes[0]] + parameter[:, None] * (
            mesh.points[nodes[1]] - mesh.points[nodes[0]]
        )
        expected = (
            -np.array([1.0, 2.0]) + affine(points)[:, None] * np.asarray(velocity) / 2
        ) @ mesh.normals[face]
        actual = (
            skeleton.faces[face].evaluate(parameter) @ solution.hybrid.trace[skeleton.dofs(face)]
        )
        assert_allclose(actual, expected, rtol=0, atol=2e-11)


@pytest.mark.parametrize("reaction", [1e-8, 1e-12, 1e-16])
def test_reaction_limit_agrees_with_the_independent_darcy_formulation(reaction):
    """Converge to the discrete Poisson solution for non-affine source response."""
    mesh = TriangleMesh.unit_square(2)
    limit = solve_darcy(mesh, source=1.0, dirichlet=affine, local_refinement=3)
    solution = solve_transport(
        mesh, reaction=reaction, source=1.0, dirichlet=affine, local_refinement=3
    )
    for actual, expected in zip(solution.values, limit.pressure, strict=True):
        assert_allclose(actual, expected, rtol=0, atol=2 * reaction + 1e-12)
    assert_allclose(solution.hybrid.trace, limit.hybrid.trace, rtol=0, atol=10 * reaction + 2e-11)


@pytest.mark.parametrize("dt", [1e-3, 1.0, 1e8, 1e12, 1e16])
def test_heat_preserves_affine_equilibrium_for_large_time_increments(dt):
    """Preserve a stationary harmonic field up to the steady diffusion limit."""
    mesh = TriangleMesh.unit_square(2)
    solution = solve_heat(
        mesh,
        [0.0, dt],
        initial=affine,
        dirichlet=lambda points, time: affine(points),
        local_refinement=3,
    )[0]
    assert solution.l2_error(affine) < 3e-12
    for local_mesh, values in zip(solution.local_meshes, solution.values, strict=True):
        assert_allclose(values, affine(local_mesh.points), rtol=0, atol=1e-11)


@pytest.mark.parametrize("dt", [0.01, 2.0, 1e8, 1e12])
def test_one_heat_step_matches_the_stationary_reaction_identity(dt):
    """Check backward Euler against RAD with reaction1/dt and augmented forcing."""
    mesh = TriangleMesh.unit_square(2)

    def forcing(points):
        """Give a nonconstant source that does not preserve affine equilibrium."""
        return 2 + points[:, 0] * points[:, 1]

    heat = solve_heat(
        mesh,
        [0.0, dt],
        initial=affine,
        source=lambda points, time: forcing(points),
        dirichlet=lambda points, time: affine(points),
        local_refinement=3,
    )[0]
    stationary = solve_transport(
        mesh,
        reaction=1 / dt,
        source=lambda points: forcing(points) + affine(points) / dt,
        dirichlet=affine,
        local_refinement=3,
    )
    for actual, expected in zip(heat.values, stationary.values, strict=True):
        assert_allclose(actual, expected, rtol=0, atol=2e-12)
    assert_allclose(heat.hybrid.trace, stationary.hybrid.trace, rtol=0, atol=2e-11)


def test_heat_large_step_converges_to_the_discrete_poisson_solution():
    """A long step with forcing approaches steady diffusion from zero initial data."""
    mesh = TriangleMesh.unit_square(2)
    steady = solve_darcy(mesh, source=1.0, dirichlet=affine, local_refinement=3)
    solution = solve_heat(
        mesh,
        [0.0, 1e12],
        source=lambda points, time: np.ones(len(points)),
        dirichlet=lambda points, time: affine(points),
        local_refinement=3,
    )[0]
    for actual, expected in zip(solution.values, steady.pressure, strict=True):
        assert_allclose(actual, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
def test_scalar_general_coarse_modes_reject_unsupported_amg_solvers(solver):
    """Require an explicit capability error instead of silently changing solvers."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="general coarse_basis"):
        solve_transport(mesh, reaction=1.0, local_solver=solver)
    with pytest.raises(ValueError, match="general coarse_basis"):
        solve_heat(mesh, [0.0, 1.0], local_solver=solver)
