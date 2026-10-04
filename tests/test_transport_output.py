"""Bounded output selection and durable per-step observation of transient fields."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pymhm import TriangleMesh
from pymhm._legacy.models.transport.transient import solve_transient_transport


def _options():
    """Exercise variable time increments, nonzero source and boundary data."""
    return dict(
        local_refinement=2,
        degree=2,
        initial=lambda x: 1 + x[:, 0] * (1 - x[:, 0]),
        source=lambda x, t: np.cos(x[:, 1]) + t,
        dirichlet=lambda x, t: 1 + t + x[:, 0],
        velocity=(0.3, -0.1),
        reaction=0.2,
    )


def test_selected_outputs_execute_every_step_and_preserve_fields_bitwise(tmp_path):
    """Every callback checkpoint matches the all-history solve without field mutation."""
    mesh = TriangleMesh.unit_square()
    times = np.array([0, 0.01, 0.02, 0.05, 0.08])
    expected = solve_transient_transport(mesh, times, **_options())
    observed = []

    def persist(step, time, solution, balance):
        """Persist independent step coordinates and reject writes to physical arrays."""
        observed.append((step, time))
        arrays = [
            *solution.values,
            solution.hybrid.trace,
            *solution.hybrid.coarse,
            *solution.hybrid.fields,
            solution.hybrid.gauge_multipliers,
            balance,
        ]
        assert not any(array.flags.writeable for array in arrays)
        with pytest.raises(ValueError, match="read-only"):
            solution.values[0][0] = 123
        np.savez(
            tmp_path / f"step-{step}.npz",
            trace=solution.hybrid.trace,
            balance=balance,
            **{f"cell-{c}": value for c, value in enumerate(solution.values)},
        )

    result = solve_transient_transport(
        mesh, times, output_steps=[2, 4], on_step=persist, **_options()
    )
    assert observed == list(enumerate(times[1:], start=1))
    assert_array_equal(result.times, times[[0, 2, 4]])
    assert_array_equal(result.integration_times, times)
    assert result.operator_builds == expected.operator_builds == 2
    assert_array_equal(result.total_mass(), expected.total_mass()[[0, 2, 4]])
    for step in range(1, len(times)):
        with np.load(tmp_path / f"step-{step}.npz") as data:
            assert_array_equal(data["trace"], expected.solutions[step - 1].hybrid.trace)
            assert_array_equal(data["balance"], expected.balance_residuals[step - 1])
            for c, value in enumerate(expected.solutions[step - 1].values):
                assert_array_equal(data[f"cell-{c}"], value)
    for index, step in enumerate((2, 4)):
        assert_array_equal(result.balance_residuals[index], expected.balance_residuals[step - 1])
        for first, second in zip(
            result.solutions[index].values, expected.solutions[step - 1].values, strict=True
        ):
            assert_array_equal(first, second)


@pytest.mark.parametrize(
    "selection",
    [[], [0], [4], [2, 1], [1, 1], [1.0], [True], [[1]], np.array([2, 1], dtype=np.uint64)],
)
def test_invalid_output_steps_are_rejected_before_factorization(selection):
    """Only ordered, distinct one-based integer steps have an unambiguous time contract."""
    with pytest.raises(ValueError, match="output_steps"):
        solve_transient_transport(TriangleMesh.unit_square(), [0, 0.1, 0.2], output_steps=selection)


def test_invalid_callback_is_rejected():
    """Observation requires an explicitly callable step consumer."""
    with pytest.raises(ValueError, match="on_step"):
        solve_transient_transport(TriangleMesh.unit_square(), [0, 0.1], on_step=0)


def test_callback_failure_closes_every_prepared_native_factor(monkeypatch):
    """Callback exceptions preserve the explicit lifetime of local/global factors."""
    from pymhm.core.offline import OfflineHybridSystem

    closed = []
    original = OfflineHybridSystem.close

    def close(self):
        """Observe the real close operation without replacing factorization."""
        original(self)
        closed.append(self)

    def fail(step, time, solution, balance):
        """Simulate a failed checkpoint filesystem."""
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(OfflineHybridSystem, "close", close)
    with pytest.raises(OSError, match="checkpoint unavailable"):
        solve_transient_transport(
            TriangleMesh.unit_square(), [0, 0.1, 0.2], local_refinement=2, on_step=fail
        )
    assert len(closed) == 1
    with pytest.raises(RuntimeError, match="closed"):
        closed[0].solve([np.zeros(3), np.zeros(3)])


def test_diffusion_limit_retains_selected_output_contract():
    """The zero-advection problem uses the same bounded output/callback contract."""
    observed = []
    result = solve_transient_transport(
        TriangleMesh.unit_square(),
        [0, 0.1, 0.2, 0.3],
        initial=1,
        dirichlet=1,
        output_steps=[3],
        on_step=lambda step, time, solution, balance: observed.append(step),
    )
    assert observed == [1, 2, 3]
    assert_array_equal(result.times, [0, 0.3])
    assert len(result.solutions) == 1
