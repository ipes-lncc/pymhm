"""Global refinement retains correction digits and the original physical equations."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.solvers import LinearSolveError, SolverUnavailableError, factorize


def difficult_system() -> HybridSystem:
    """Assemble a genuine Schur system with nearly dependent trace responses."""
    coupling = np.array([[1.0, -1.0], [0.0, 1e-4]])
    local = LocalProblem(np.eye(2), coupling, np.zeros(2), np.arange(2))
    return HybridSystem([local], boundary_load=np.array([0.3, -0.3 + 1e-8]))


def test_reused_factor_preserves_extended_global_and_reconstructed_fields() -> None:
    """Tight residual accuracy survives the global coefficient and reconstruction storage."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("NumPy longdouble is not wider than double")
    system = difficult_system()
    with factorize(system.matrix, rtol=1e-17) as prepared:
        with pytest.raises(LinearSolveError, match="residual criterion"):
            system.solve(factorization=prepared)
        result = system.solve(factorization=prepared, refinement_precision="extended")
    assert result.trace.dtype == np.longdouble
    assert result.fields[0].dtype == np.longdouble
    error = system.matrix.astype(np.longdouble) @ result.trace - system.rhs
    assert np.linalg.norm(error) <= 1e-17 * np.linalg.norm(system.rhs)
    local = system.responses[0].problem
    assert_allclose(
        local.matrix @ result.fields[0] + local.coupling @ result.trace, 0.0, atol=1e-18
    )


def test_extended_global_fixed_values_gauges_and_empty_reduction() -> None:
    """Prescribed traces and physical means retain their equations in the explicit mode."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("NumPy longdouble is not wider than double")
    local = LocalProblem(
        [[1.0, -1.0], [-1.0, 1.0]],
        np.eye(2),
        [0.0, 0.0],
        np.arange(2),
        kernel=np.ones((2, 1)),
        constraints=np.ones((2, 1)) / 2,
    )
    system = HybridSystem([local])
    mean = system.mean_constraint([np.ones(2) / 2], 3.0)
    result = system.solve(
        fixed={0: -1.0, 1: 1.0}, constraints=[mean], refinement_precision="extended"
    )
    assert_allclose(result.fields[0], [3.5, 2.5], atol=1e-14)
    assert result.coarse[0].dtype == np.longdouble
    assert result.gauge_multipliers.dtype == np.longdouble
    with pytest.raises(ValueError, match="gauge changed"):
        system.solve(fixed={0: 0.0, 1: 1.0}, constraints=[mean], refinement_precision="extended")
    empty = HybridSystem([LocalProblem([[1.0]], [[1.0]], [1.0], np.array([0]))])
    fixed = empty.solve(fixed={0: 1.0}, refinement_precision="extended")
    assert_allclose(fixed.fields[0], 0.0)
    assert fixed.trace.dtype == np.longdouble


def test_extended_global_validates_precision_and_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported precision before backend dispatch, including all-fixed systems."""
    system = difficult_system()
    with pytest.raises(ValueError, match="double or extended"):
        system.solve(refinement_precision="quadruple")
    original = np.finfo

    def no_extended(dtype: Any) -> Any:
        """Emulate a platform whose longdouble has only a double significand."""
        return original(float) if dtype == np.longdouble else original(dtype)

    monkeypatch.setattr(np, "finfo", no_extended)
    with pytest.raises(SolverUnavailableError, match="wider long-double"):
        system.solve(refinement_precision="extended")


@pytest.mark.parametrize("dimension", [2, 3])
def test_rad_interfaces_retain_explicit_global_precision(dimension: int) -> None:
    """The public PDE paths retain the opt-in coefficient type through physical fields."""
    from pymhm import TetraMesh, TriangleMesh, solve_rad_3d, solve_transport

    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("NumPy longdouble is not wider than double")
    if dimension == 2:
        mesh, solve = TriangleMesh.unit_square(), solve_transport
    else:
        mesh, solve = TetraMesh.unit_cube(), solve_rad_3d
    options = dict(
        source=1.0,
        degree=2 if dimension == 2 else 4,
        local_refinement=2 if dimension == 2 else 1,
        coarse_space="kernel",
    )
    standard = solve(mesh, **options)
    extended = solve(mesh, global_refinement_precision="extended", **options)
    assert extended.hybrid.trace.dtype == np.longdouble
    for actual, expected in zip(extended.values, standard.values, strict=True):
        assert actual.dtype == np.longdouble
        assert_allclose(actual, expected, atol=3e-13)
