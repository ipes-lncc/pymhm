"""Literal sampled mass/load forms and user-written conservative time equations."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.formulations.transient import solve_transport_trajectory
from pymhm._legacy.models.transport.transient import solve_transient_transport
from pymhm.fem.loads import assemble_load, assemble_mass
from pymhm.materials.macro import MacroCoefficient
from pymhm.meshes.triangle import TriangleMesh


def test_sampled_rectangular_complex_mass_and_load_scatter():
    tests = np.array([[[1 + 2j, 3], [0.5, -1j]], [[-2, 1j], [3, 1]]])
    trials = np.array([[[1, 0, 2j], [3, 2, -1]], [[1j, 4, 0], [0, -1, 2]]])
    weights = np.array([[0.3, 0.8], [1.4, -0.2]])
    rows = np.array([[0, 1], [1, 2]])
    cols = np.array([[0, 2, 3], [1, 2, 3]])
    source = np.array([[1 - 1j, 2], [0.5j, -3]])
    expected = np.zeros((3, 4), complex)
    load = np.zeros(3, complex)
    for cell in range(2):
        for q in range(2):
            for i in range(2):
                load[rows[cell, i]] += weights[cell, q] * tests[cell, q, i] * source[cell, q]
                for j in range(3):
                    expected[rows[cell, i], cols[cell, j]] += (
                        weights[cell, q] * tests[cell, q, i] * trials[cell, q, j]
                    )
    np.testing.assert_allclose(
        assemble_mass(tests, trials, weights, rows, cols, (3, 4)).toarray(), expected
    )
    np.testing.assert_allclose(assemble_load(tests, source, weights, rows, 3), load)
    empty = np.empty((0, 2, 1))
    assert not assemble_load(empty, 1.0, 1.0, np.empty((0, 1), int), 0).size
    assert assemble_mass(
        empty, empty, 1.0, np.empty((0, 1), int), np.empty((0, 1), int), (0, 0)
    ).shape == (0, 0)


@pytest.mark.parametrize(
    "operation,arguments",
    [
        (assemble_load, (np.ones((2, 2)), 1.0, 1.0, [[0]], 1)),
        (assemble_load, (np.full((1, 1, 1), np.nan), 1.0, 1.0, [[0]], 1)),
        (assemble_load, (np.ones((1, 1, 1)), np.inf, 1.0, [[0]], 1)),
        (assemble_load, (np.ones((1, 1, 1)), 1.0, np.nan, [[0]], 1)),
        (assemble_mass, (np.ones((2, 2)), np.ones((2, 2, 1)), 1.0, [[0]], [[0]], (1, 1))),
        (assemble_mass, (np.ones((2, 2, 1)), np.ones((2, 2)), 1.0, [[0]], [[0]], (1, 1))),
        (assemble_mass, (np.ones((1, 2, 1)), np.ones((2, 2, 1)), 1.0, [[0]], [[0]], (1, 1))),
        (
            assemble_mass,
            (np.full((1, 1, 1), np.inf), np.ones((1, 1, 1)), 1.0, [[0]], [[0]], (1, 1)),
        ),
        (
            assemble_mass,
            (np.ones((1, 1, 1)), np.full((1, 1, 1), np.nan), 1.0, [[0]], [[0]], (1, 1)),
        ),
        (assemble_mass, (np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.inf, [[0]], [[0]], (1, 1))),
    ],
)
def test_sampled_form_invalid_tabs(operation, arguments):
    with pytest.raises(ValueError):
        operation(*arguments)


@pytest.mark.parametrize("scheme", ["galerkin", "supg"])
@pytest.mark.parametrize("enforcement", ["weak", "strong"])
def test_user_time_forms_match_full_current_capacity_residual_and_mixed_data(scheme, enforcement):
    mesh = TriangleMesh.unit_square()
    left, right = map(int, mesh.boundary_faces[:2])
    options = dict(
        degree=2,
        local_refinement=2,
        quadrature_order=8,
        diffusion=lambda p: 1 + 0.05 * p[:, 0],
        diffusion_divergence=(0.05, 0.0),
        velocity=lambda p: np.column_stack((0.2 + 0.1 * p[:, 0], np.full(len(p), 0.1))),
        velocity_divergence=0.1,
        capacity=lambda p: 1 + 0.3 * p[:, 0],
        reaction=0.2,
        initial=lambda p: 0.3 + p[:, 0],
        source=lambda p, t: (1 + t) * (1 + p[:, 1]),
        dirichlet=lambda p, t: 0.4 + t + p[:, 0],
        neumann={left: lambda p, t: 0.05 + t},
        diffusive_flux={right: lambda p, t: 0.1 - t},
        stabilization=scheme,
        dirichlet_enforcement=enforcement,
        check_original=True,
        output_steps=(2, 3),
    )
    with threadpool_limits(1):
        actual = solve_transport_trajectory(mesh, [0.0, 0.02, 0.04, 0.08], **options)
        expected = solve_transient_transport(mesh, [0.0, 0.02, 0.04, 0.08], **options)
    assert actual.operator_builds == expected.operator_builds == 2
    np.testing.assert_array_equal(actual.times, expected.times)
    for own, ref in zip(actual.solutions, expected.solutions, strict=True):
        for values, target in zip(
            (*own.values, own.hybrid.trace), (*ref.values, ref.hybrid.trace), strict=True
        ):
            np.testing.assert_allclose(values, target, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(actual.total_mass(), expected.total_mass(), atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(actual.balance_residuals, 0.0, atol=1e-12)
    assert actual.original_residual_norms is not None
    assert len(actual.original_residual_norms) == 3


def test_one_sided_capacity_and_supg_time_residual_preserve_rigid_growth():
    mesh = TriangleMesh.unit_square()
    capacity = MacroCoefficient((1.3, 2.1))
    source = MacroCoefficient(
        tuple((lambda p, t, value=rho: np.full(len(p), value)) for rho in capacity.fields)
    )
    with threadpool_limits(1):
        history = solve_transport_trajectory(
            mesh,
            [0.0, 0.02, 0.04],
            capacity=capacity,
            source=source,
            initial=0.7,
            dirichlet=lambda p, t: np.full(len(p), 0.7 + t),
            velocity=(0.4, 0.2),
            stabilization="supg",
            degree=2,
            local_refinement=2,
            check_original=True,
        )
    for time, solution in zip(history.times[1:], history.solutions, strict=True):
        assert solution.l2_error(0.7 + time) < 1e-12
    np.testing.assert_allclose(
        history.total_mass(), (0.7 + history.times) * 1.7, atol=1e-12, rtol=1e-10
    )
