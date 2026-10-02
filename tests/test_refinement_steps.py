"""Explicit bounded defect corrections preserve the original equation criterion."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from threadpoolctl import threadpool_limits

import pymhm.solvers as backend
from pymhm.solvers import LinearFactorization, LinearSolveError, factorize, solve_linear


def conductive_factor(matrix: np.ndarray, ratio: float = 1e-3) -> tuple[LinearFactorization, list]:
    """Factor a slightly perturbed conductive operator with a known contracting defect."""
    inverse = np.linalg.inv(matrix / (1 - ratio))
    calls: list[np.ndarray] = []

    def solve(rhs: np.ndarray) -> np.ndarray:
        """Use the same approximate inverse for every original or defect right-hand side."""
        calls.append(rhs.copy())
        return inverse @ rhs

    def release() -> None:
        """Finish without releasing external resources."""

    factor = LinearFactorization(sparse.csr_matrix(matrix), solve, release, "controlled", 1e-13)
    return factor, calls


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
@pytest.mark.parametrize("columns", [False, True])
def test_four_corrections_recover_conductive_fields(scale: float, columns: bool) -> None:
    """A contracting factor needs four corrections at the unchanged columnwise tolerance."""
    matrix = scale * np.array([[2.0, -1.0], [-1.0, 2.0]])
    expected = np.array([[1.0, 2.0, 0.0], [-2.0, 1.0, 0.0]])
    if not columns:
        expected = expected[:, 0]
    rhs = matrix @ expected
    failed, failed_calls = conductive_factor(matrix)
    with failed, pytest.raises(LinearSolveError, match="residual criterion"):
        failed.solve(rhs, refinement_steps=3)
    assert len(failed_calls) == 4
    successful, calls = conductive_factor(matrix)
    with successful:
        result = successful.solve(rhs, refinement_steps=np.int64(4))
    assert len(calls) == 5
    assert_allclose(result, expected, rtol=2e-14, atol=2e-14)
    assert np.linalg.norm(matrix @ result - rhs) <= 1e-13 * np.linalg.norm(rhs)


@pytest.mark.parametrize("steps", [0, 1, 2, 5])
@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_stagnating_or_diverging_factors_respect_the_bound(steps: int, ratio: float) -> None:
    """Finite corrections cannot turn an exhausted or divergent iteration into success."""
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    calls = []

    def inaccurate(rhs: np.ndarray) -> np.ndarray:
        """Produce stagnation or divergence with a zero or sign-reversed inverse."""
        calls.append(rhs.copy())
        return (1 - ratio) * np.linalg.solve(matrix, rhs)

    def release() -> None:
        """Finish without releasing native resources."""

    factors = LinearFactorization(sparse.csr_matrix(matrix), inaccurate, release, "test", 1e-13)
    with factors, pytest.raises(LinearSolveError, match="residual criterion"):
        factors.solve(np.array([3.0, -1.0]), refinement_steps=steps)
    assert len(calls) == steps + 1


@pytest.mark.parametrize("steps", [-1, True, False, np.bool_(True), 2.0, 1.2, "4", None, np.nan])
def test_correction_count_contract_is_shared(steps: Any) -> None:
    """Both entry points reject negative, Boolean and noninteger limits before solving."""
    with (
        factorize(np.eye(2)) as factors,
        pytest.raises(ValueError, match="nonnegative integer"),
    ):
        factors.solve([1.0, 2.0], refinement_steps=steps)
    with pytest.raises(ValueError, match="nonnegative integer"):
        solve_linear(np.eye(2), [1.0, 2.0], refinement_steps=steps)


def test_zero_default_and_early_success_paths() -> None:
    """Zero permits an exact initial solve and explicit two reproduces default bytes."""
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([4.0, 3.0])
    with factorize(matrix) as factors:
        zero = factors.solve(rhs, refinement_steps=0)
        default = factors.solve(rhs)
        explicit = factors.solve(rhs, refinement_steps=2)
    assert np.array_equal(zero, default)
    assert np.array_equal(default, explicit)
    assert_allclose(solve_linear(matrix, rhs, refinement_steps=0), [1, 1])
    controlled, calls = conductive_factor(matrix, ratio=0.0)
    with controlled:
        assert_allclose(controlled.solve(rhs, refinement_steps=100), [1, 1])
    assert len(calls) == 1


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_default_two_preserves_every_correction(precision: str) -> None:
    """Match default and explicit-two outputs and all intermediate forcing arrays bitwise."""
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("this explicit precision control requires wider long double")
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = matrix @ np.array([1.0, -2.0])
    default, first = conductive_factor(matrix, ratio=1e-5)
    explicit, second = conductive_factor(matrix, ratio=1e-5)
    with default, explicit:
        a = default.solve(rhs, refinement_precision=precision)
        b = explicit.solve(rhs, refinement_precision=precision, refinement_steps=2)
    assert len(first) == len(second) == 3
    assert np.array_equal(a, b)
    assert all(np.array_equal(x, y) for x, y in zip(first, second, strict=True))


def test_solve_linear_forwards_the_direct_limit(monkeypatch: Any) -> None:
    """The one-call API uses the same controlled reusable factor and requested budget."""
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = matrix @ np.array([1.0, -2.0])

    def factory(*args: Any, **kwargs: Any) -> LinearFactorization:
        """Provide an identically perturbed factor to each independent solve."""
        return conductive_factor(matrix)[0]

    monkeypatch.setattr(backend, "factorize", factory)
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solve_linear(matrix, rhs, refinement_steps=3)
    assert_allclose(solve_linear(matrix, rhs, refinement_steps=4), [1, -2], atol=2e-14)


def test_solve_linear_krylov_uses_the_same_bound(monkeypatch: Any) -> None:
    """A premature Krylov acceptance receives the requested number of true defects."""
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = matrix @ np.array([1.0, -2.0])
    calls = []

    def incomplete(operator: Any, forcing: np.ndarray, **kwargs: Any) -> tuple:
        """Report recursive convergence while retaining a controlled true defect."""
        calls.append(forcing.copy())
        return 0.999 * np.linalg.solve(matrix, forcing), 0

    monkeypatch.setattr(backend.splinalg, "cg", incomplete)
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solve_linear(matrix, rhs, solver="cg", rtol=1e-13, refinement_steps=3)
    assert len(calls) == 4
    calls.clear()
    result = solve_linear(matrix, rhs, solver="cg", rtol=1e-13, refinement_steps=4)
    assert len(calls) == 5
    assert_allclose(result, [1, -2], atol=2e-14)


@pytest.mark.parametrize("solver", ["amgx", "cupy"])
def test_one_call_accelerators_reject_unavailable_outer_loop(solver: str) -> None:
    """Unsupported explicit limits fail without importing an optional accelerator."""
    with pytest.raises(ValueError, match="Krylov or reusable direct"):
        solve_linear(np.eye(2), [1, 2], solver=solver, refinement_steps=4)


def test_native_optional_cpu_integration_pypardiso_extended_well(monkeypatch: Any) -> None:
    """Solve the F8 mixed saddle and verify both congruent and physical equations."""
    pytest.importorskip("pypardiso")
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("this explicit precision control requires wider long double")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.solve_mapped_well_invariant import HexMesh, OscillatoryWellData, assemble

    data = OscillatoryWellData()
    mesh = HexMesh.annular_prism(
        np.geomspace(data.inner_radius, data.outer_radius, 5), data.height, 8
    ).refined((8, 8, 1))
    with threadpool_limits(1):
        physical, rhs, scaling, ids, signs, divergence, _ = assemble(mesh, data, 12)
        transform = sparse.diags(scaling)
        balanced = (transform @ physical @ transform).tocsr()
        forcing = scaling * rhs
        result = solve_linear(
            balanced,
            forcing,
            solver="pypardiso-symmetric-matching",
            rtol=1e-17,
            refinement_precision="extended",
            refinement_steps=8,
        )
    residual = backend._accurate_residual(balanced, forcing, result)
    assert np.linalg.norm(residual) <= 1e-17 * np.linalg.norm(forcing.astype(np.longdouble))
    solution = scaling.astype(np.longdouble) * result
    defect = backend._accurate_residual(physical, rhs, solution)
    action = abs(physical).astype(np.longdouble) @ abs(solution) + abs(rhs)
    nq = int(ids.max()) + 1
    for part in (slice(0, nq), slice(nq, None)):
        assert np.linalg.norm(defect[part]) <= 1e-10 * np.linalg.norm(action[part])
    flux = solution[:nq][ids] * signs
    balance = flux @ divergence.astype(np.longdouble).T
    magnitude = abs(flux) @ abs(divergence.astype(np.longdouble)).T
    assert np.all(np.linalg.norm(balance, axis=1) <= 1e-10 * np.linalg.norm(magnitude, axis=1))
