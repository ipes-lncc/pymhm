"""Generic Newmark identities, response columns and explicit factor ownership."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm.linalg.dynamics import newmark_step
from pymhm.linalg.linear import factorize


@pytest.mark.parametrize("complex_operator", [False, True])
@pytest.mark.parametrize("columns", [False, True])
def test_literal_order_and_independent_original_equations(complex_operator, columns):
    """Vector/column updates satisfy both original endpoint Newmark equations."""
    mass = np.array([[2.0, 0.1], [0.1, 1.4]])
    stiffness = np.array([[3.0, -0.2], [-0.2, 2.0]])
    if complex_operator:
        stiffness = stiffness + 0.1j * np.eye(2)
    mass, stiffness = sparse.csc_matrix(mass), sparse.csc_matrix(stiffness)
    duration = 0.037
    u, v, old, new = np.random.default_rng(402).normal(size=(4, 2, 3) if columns else (4, 2))
    with factorize(mass) as inverse, factorize(mass + duration**2 / 4 * stiffness) as step:
        result, velocity = newmark_step(mass, stiffness, step, inverse, duration, u, v, old, new)
        expected = step.solve(
            mass @ (u + duration * v)
            - duration**2 / 4 * (stiffness @ u)
            + duration**2 / 4 * (old + new)
        )
        expected_v = v + duration / 2 * inverse.solve(old + new - stiffness @ (u + expected))
        assert_array_equal(result, expected)
        assert_array_equal(velocity, expected_v)
        assert_allclose(result - u, duration / 2 * (v + velocity), atol=4e-16, rtol=2e-14)
        assert_allclose(
            mass @ (velocity - v),
            duration / 2 * (old + new - stiffness @ (u + result)),
            atol=5e-16,
            rtol=2e-14,
        )
        assert step.matches(mass + duration**2 / 4 * stiffness)
        assert inverse.matches(mass)
    with pytest.raises(RuntimeError, match="closed"):
        newmark_step(mass, stiffness, step, inverse, duration, u, v, old, new)


def test_unforced_symmetric_system_preserves_physical_energy():
    """The free primitive preserves quadratic energy for its symmetric system."""
    mass = sparse.csc_matrix([[2.0, 0.1], [0.1, 1.3]])
    stiffness = sparse.csc_matrix([[3.0, -0.2], [-0.2, 2.0]])
    duration = 0.043
    u, v, zero = np.array([0.3, -0.2]), np.array([0.1, 0.4]), np.zeros(2)
    energy = u @ (stiffness @ u) + v @ (mass @ v)
    with factorize(mass) as inverse, factorize(mass + duration**2 / 4 * stiffness) as step:
        for _ in range(20):
            u, v = newmark_step(mass, stiffness, step, inverse, duration, u, v, zero, zero)
            assert_allclose(u @ (stiffness @ u) + v @ (mass @ v), energy, atol=1e-15, rtol=0)
