"""Operator-independent moment reconstruction and preserved represented equations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm.core.moments import energy_reconstruction
from pymhm.linalg import moments as linear_moments
from pymhm.linalg.linear import SolverUnavailableError
from pymhm.linalg.moments import solve_moment_system
from pymhm.methods import hho, hho_3d

HAS_EXTENDED = np.finfo(np.longdouble).nmant > np.finfo(float).nmant


@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_original_saddle_physical_moments_and_reduced_bilinear_action(
    nonsymmetric: bool,
) -> None:
    """The constrained lift satisfies both original block rows without symmetrization."""
    matrix = np.array([[3.0, 0.75, 0.5], [0.75, 2.0, 0.25], [0.5, 0.25, 4.0]])
    if nonsymmetric:
        matrix[0, 1] = 1.5
        matrix[1, 0] = -0.25
    operator = sparse.csc_matrix(matrix)
    moments = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.5]])
    reconstruction, reduced = energy_reconstruction(operator, moments)
    multiplier = np.linalg.solve(moments.T @ moments, -moments.T @ matrix @ reconstruction)
    assert_allclose(matrix @ reconstruction + moments @ multiplier, np.zeros((3, 2)), atol=2e-14)
    assert_allclose(moments.T @ reconstruction, np.eye(2), atol=2e-14)
    trial = np.array([1.25, -0.5])
    test = np.array([-0.4, 2.0])
    assert_allclose(
        test @ reduced @ trial,
        (reconstruction @ test) @ matrix @ (reconstruction @ trial),
        atol=2e-14,
    )
    if nonsymmetric:
        assert np.linalg.norm(reduced - reduced.T) > 0.1
    else:
        assert np.linalg.eigvalsh(reduced).min() > 0


def test_declared_integral_moment_fixes_the_constant_kernel() -> None:
    """A singular energy operator becomes reconstructible through a physical mean."""
    matrix = sparse.csc_matrix([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    moments = np.array([[0.25], [0.5], [0.25]])
    reconstruction, reduced = energy_reconstruction(matrix, moments)
    assert_allclose(reconstruction[:, 0], np.ones(3), atol=2e-14)
    assert_allclose(moments.T @ reconstruction, [[1.0]], atol=2e-14)
    assert_allclose(matrix @ reconstruction, np.zeros((3, 1)), atol=2e-14)
    assert_allclose(reduced, [[0.0]], atol=2e-14)


@pytest.mark.parametrize("zero", [False, True])
def test_missing_independent_physical_moments_are_rejected(zero: bool) -> None:
    """A numerical lift cannot silently remove a dependent declared moment."""
    moments = np.zeros((3, 1)) if zero else np.ones((3, 2))
    with pytest.raises(ValueError, match="independent"):
        energy_reconstruction(sparse.eye(3), moments)


@pytest.mark.parametrize("multiple", [False, True])
def test_original_moment_equations_and_native_resource_lifetime(
    multiple: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every RHS solves its original rows and the native factor closes before return."""
    matrix = np.array([[3.0, -1.0], [0.5, 2.0]])
    rhs = np.array([[1.0, -2.0], [3.0, 0.5]]) if multiple else np.array([1.0, 3.0])
    factors = []
    original = linear_moments.factorize

    def track_factorization(*args: Any, **kwargs: Any) -> Any:
        """Observe the actual owned factor while retaining the native solve implementation."""
        factor = original(*args, **kwargs)
        factors.append(factor)
        return factor

    monkeypatch.setattr(linear_moments, "factorize", track_factorization)
    result = solve_moment_system(matrix, rhs)
    assert_allclose(matrix @ result, rhs, atol=2e-14)
    assert_allclose(result, np.linalg.solve(matrix, rhs), atol=2e-14)
    assert factors and all(factor._solve is None for factor in factors)


def test_methods_delegate_to_shared_moment_implementations() -> None:
    """Method owners use the shared numerical functions directly."""
    assert hho.solve_moment_system is solve_moment_system
    assert hho.energy_reconstruction is energy_reconstruction
    assert hho_3d.energy_reconstruction is energy_reconstruction


@pytest.mark.skipif(not HAS_EXTENDED, reason="The platform has no wider long-double storage")
def test_wide_operator_digits_survive_moment_reconstruction_and_reduction() -> None:
    """The original operator's sub-binary64 digits remain in its represented energy."""
    value = np.longdouble(1) + np.longdouble(2) ** -54
    matrix = sparse.csc_matrix(np.array([[value, 0], [0, 2]], dtype=np.longdouble))
    reconstruction, reduced = energy_reconstruction(matrix, np.eye(2), "extended")
    assert reconstruction.dtype == np.dtype(np.longdouble)
    assert reduced.dtype == np.dtype(np.longdouble)
    assert_array_equal(reconstruction, np.eye(2))
    assert reduced[0, 0] == value
    assert reduced[0, 0] != np.longdouble(float(value))
    force = np.array([value, 3], dtype=np.longdouble)
    result = solve_moment_system(np.eye(2), force, refinement_precision="extended")
    assert result.dtype == np.dtype(np.longdouble)
    assert_array_equal(result, force)


def test_explicit_extended_representation_has_no_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared operation preserves the native owner's explicit precision policy."""
    from pymhm.linalg import linear

    monkeypatch.setattr(linear, "_EXTENDED_PRECISION", False)
    with pytest.raises(SolverUnavailableError, match="wider"):
        solve_moment_system(np.eye(2), np.ones(2), refinement_precision="extended")
