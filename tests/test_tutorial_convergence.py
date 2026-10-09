"""Check rate diagnostics without interpreting a fitted slope as PDE validation."""

from typing import Any

import numpy as np
import pytest

from examples.tutorial_convergence import asymptotic_summary


def _series(sizes: np.ndarray, errors: np.ndarray, target: float | None) -> dict[str, Any]:
    """Build one physical observable with the exact, possibly non-dyadic ratios."""
    return {
        "sizes": sizes.tolist(),
        "errors": {"field": errors.tolist()},
        "rates": {
            "field": (np.log(errors[:-1] / errors[1:]) / np.log(sizes[:-1] / sizes[1:])).tolist()
        },
        "expected": {} if target is None else {"field": target},
    }


def test_non_dyadic_asymptotic_orders_and_amplitudes() -> None:
    """An exact power law has its declared exponent and constant amplitude."""
    sizes = np.array([0.25, 0.125, 1 / 12, 1 / 16, 1 / 24])
    result = asymptotic_summary(_series(sizes, 3.7 * sizes**2.5, 2.5))["field"]
    assert result["start_index"] == 1
    np.testing.assert_allclose(result["orders"], 2.5)
    assert result["fitted_order"] == pytest.approx(2.5)
    assert result["amplitude_ratio"] == pytest.approx(1)


def test_terminal_window_preserves_a_local_discretization_floor() -> None:
    """A misleading earlier target rate cannot replace the actual terminal data."""
    sizes = np.array([1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    errors = sizes**2 + 0.001
    result = asymptotic_summary(_series(sizes, errors, 2))["field"]
    np.testing.assert_array_equal(result["sizes"], sizes[-4:])
    assert result["orders"][-1] < 1.5
    assert result["fitted_order"] < 2
    assert result["amplitude_ratio"] > 1.5


def test_observed_order_does_not_acquire_an_uncited_target() -> None:
    """A measured field without a target remains an observation only."""
    sizes = np.array([1, 0.5, 0.25, 0.125])
    result = asymptotic_summary(_series(sizes, sizes**3, None))["field"]
    assert result["fitted_order"] == pytest.approx(3)
    assert result["target"] is None
    assert "amplitude_ratio" not in result
    with pytest.raises(ValueError, match="terminal window"):
        asymptotic_summary(_series(sizes, sizes**3, None), tail_intervals=4)
