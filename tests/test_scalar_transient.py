"""Short independent temporal patches, conservation and Darcy-coupling checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy, solve_heat, solve_transport
from pymhm.darcy_transport import HydrodynamicDispersion, RT0DarcyVelocity, solve_darcy_transport
from pymhm.elements import rt0_evaluate
from pymhm.scalar_transient import MacroCoefficient, solve_transient_transport


def affine(points):
    return 1 + points[:, 0] + 2 * points[:, 1]


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_variable_transport_linear_time_patch_and_operator_reuse(stabilization):
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))

    def diffusion(x):
        matrix = np.zeros((len(x), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = 1 + x[:, 0], 2 + x[:, 1]
        return matrix

    def source(x, t):
        return (1 + x[:, 0]) * affine(x) + (1 + t) * (-3 + x[:, 0] + 2 * x[:, 1] + 1.5 * affine(x))

    result = solve_transient_transport(
        mesh,
        [0, 0.125, 0.25, 0.5],
        skeleton=space,
        local_refinement=2,
        degree=3,
        initial=affine,
        capacity=lambda x: 1 + x[:, 0],
        diffusion=diffusion,
        diffusion_divergence=(1, 1),
        velocity=lambda x: x,
        velocity_divergence=2,
        reaction=-0.5,
        source=source,
        dirichlet=lambda x, t: (1 + t) * affine(x),
        stabilization=stabilization,
    )
    assert result.operator_builds == 2
    for time, solution, balance in zip(
        result.times[1:], result.solutions, result.balance_residuals, strict=True
    ):
        assert solution.l2_error(lambda x, t=time: (1 + t) * affine(x), order=6) < 2e-12
        assert_allclose(balance, 0, atol=3e-11)
    assert_allclose(result.total_mass(), (1 + result.times) * (23 / 6), atol=3e-12)


def test_closed_diffusion_preserves_mass_and_roundoff_uniform_time_grid():
    mesh = TriangleMesh.unit_square()
    result = solve_transient_transport(
        mesh,
        np.linspace(0, 0.3, 4),
        initial=1,
        capacity=lambda x: 1 + x[:, 0],
        diffusive_flux=dict.fromkeys(map(int, mesh.boundary_faces), 0),
    )
    assert result.operator_builds == 1
    assert_allclose(result.total_mass(), 1.5, atol=2e-12)
    assert max(s.l2_error(1) for s in result.solutions) < 2e-12


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_one_transient_step_equals_independent_stationary_rad(stabilization):
    mesh = TriangleMesh.unit_square()
    robin_face = int(mesh.boundary_faces[0])
    options = dict(
        velocity=(1, 0.2),
        reaction=2,
        dirichlet_enforcement="weak",
        stabilization=stabilization,
        local_refinement=2,
    )
    evolved = solve_transient_transport(
        mesh,
        [0, 0.25],
        initial=affine,
        source=lambda x, t: x[:, 0] + t,
        dirichlet=lambda x, t: 1 + t,
        neumann={robin_face: lambda x, t: t},
        **options,
    ).solutions[0]
    options["reaction"] += 4
    stationary = solve_transport(
        mesh,
        source=lambda x: x[:, 0] + 0.25 + 4 * affine(x),
        dirichlet=1.25,
        neumann={robin_face: 0.25},
        **options,
    )
    assert_allclose(evolved.hybrid.trace, stationary.hybrid.trace, atol=2e-12)
    for actual, reference in zip(evolved.values, stationary.values, strict=True):
        assert_allclose(actual, reference, atol=2e-12)


def test_heat_wrapper_agrees_with_general_transient_galerkin():
    mesh = TriangleMesh.unit_square()
    options = dict(initial=affine, source=lambda x, t: 1 + t, dirichlet=lambda x, t: affine(x))
    existing = solve_heat(mesh, [0, 0.1, 0.3], **options)
    general = solve_transient_transport(
        mesh, [0, 0.1, 0.3], dirichlet_enforcement="weak", **options
    )
    for first, second in zip(existing, general.solutions, strict=True):
        for a, b in zip(first.values, second.values, strict=True):
            assert_allclose(a, b, atol=2e-12)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"times": [0, 0]}, "times"),
        ({"degree": 0}, "degree"),
        ({"stabilization": "other"}, "invalid"),
        ({"dirichlet_enforcement": "other"}, "invalid"),
        ({"skeleton": SkeletonSpace(TriangleMesh.unit_square())}, "scalar skeleton"),
        ({"neumann": {0: 0}, "diffusive_flux": {0: 0}}, "disjoint"),
        ({"velocity": lambda x: x}, "velocity_divergence"),
        ({"stabilization": "supg", "diffusion": lambda x: 1}, "diffusion_divergence"),
        ({"velocity_divergence": 1}, "constant velocity"),
        ({"diffusion_divergence": (1, 0)}, "constant diffusion"),
        ({"capacity": 0}, "capacity"),
        ({"capacity": MacroCoefficient((1,))}, "one field"),
    ],
)
def test_transient_data_contract(options, match):
    times = options.pop("times", [0, 0.1])
    with pytest.raises(ValueError, match=match):
        solve_transient_transport(TriangleMesh.unit_square(), times, **options)


def test_rt0_velocity_and_dispersion_derivative_are_exact_inside_cells():
    mesh = TriangleMesh.unit_square(2)
    centers = mesh.points[mesh.faces].mean(axis=1)
    flux = np.sum((2 + centers) * mesh.normals, axis=1) * mesh.lengths
    velocity = RT0DarcyVelocity(mesh, flux)
    bary = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
    assert_allclose(velocity(points), rt0_evaluate(mesh, flux, bary).reshape(-1, 2), atol=1e-14)
    assert_allclose(velocity.divergence(points), 2, atol=1e-14)
    material = HydrodynamicDispersion(velocity, 0.1, 0.3, 0.05)
    h = 1e-5
    derivative = np.zeros((len(points), 2))
    for axis in range(2):
        delta = np.eye(2)[axis] * h
        derivative += (
            material(points + delta)[:, axis, :] - material(points - delta)[:, axis, :]
        ) / (2 * h)
    assert_allclose(material.divergence(points), derivative, rtol=1e-9, atol=1e-10)
    zero = HydrodynamicDispersion(RT0DarcyVelocity(mesh, np.zeros(len(mesh.faces))), 0.1, 0, 0)
    assert_allclose(zero(points), np.broadcast_to(0.1 * np.eye(2), (len(points), 2, 2)))
    assert_allclose(zero.divergence(points), 0)
    # Candidate lookup is only an acceleration: membership still controls selection.
    velocity.tree = SimpleNamespace(query=lambda x, k: (None, np.zeros((len(x), k), dtype=int)))
    assert_allclose(velocity(points), 2 + points, atol=1e-14)
    for invalid in (np.array([[2.0, 2.0]]), np.array([1.0, 2.0]), np.array([[np.nan, 0.0]])):
        with pytest.raises(ValueError):
            velocity(invalid)


@pytest.mark.parametrize("flux", [[1.0], [1j] * 5, [np.nan] * 5])
def test_invalid_rt0_flux(flux):
    with pytest.raises(ValueError, match="RT0 flux"):
        RT0DarcyVelocity(TriangleMesh.unit_square(), flux)


@pytest.mark.parametrize("parameters", [(0, 1, 0), (1, -0.1, 0), (1, 0.1, 0.2), (1, np.nan, 0)])
def test_invalid_dispersion(parameters):
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="molecular"):
        HydrodynamicDispersion(RT0DarcyVelocity(mesh, np.zeros(len(mesh.faces))), *parameters)


def test_darcy_driven_nonzero_transport_patch_and_coupling_contract():
    mesh = TriangleMesh.unit_square()
    darcy = solve_darcy(mesh, formulation="mixed", dirichlet=lambda x: -x[:, 0], local_refinement=2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    result = solve_darcy_transport(
        darcy,
        [0, 0.1, 0.2],
        skeleton=space,
        degree=2,
        molecular=0.1,
        longitudinal=0.2,
        transverse=0.1,
        initial=affine,
        source=lambda x, t: affine(x) + 1 + t,
        dirichlet=lambda x, t: (1 + t) * affine(x),
        stabilization="supg",
    )
    assert result.solutions[-1].l2_error(lambda x: 1.2 * affine(x)) < 3e-12
    for options, match in (
        ({"local_refinement": 3}, "match"),
        ({"velocity": (1, 0)}, "overridden"),
    ):
        with pytest.raises(ValueError, match=match):
            solve_darcy_transport(darcy, [0, 1], **options)
    with pytest.raises(ValueError, match="RT0"):
        solve_darcy_transport(solve_darcy(mesh), [0, 1])
