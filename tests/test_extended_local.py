"""Explicit wider accumulation for checked direct factors and local MHM responses."""

from typing import Any

import numpy as np
import pytest

from pymhm import solvers
from pymhm.darcy import solve_darcy
from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.mesh import TriangleMesh
from pymhm.solvers import LinearSolveError, SolverUnavailableError, factorize


def affine(points: np.ndarray) -> np.ndarray:
    """Return a nonhomogeneous polynomial field with exactly represented gradient."""
    return 1 + points @ np.array([1.0, 2.0])


def independent_problem(index: int) -> LocalProblem:
    """Build a spawn-safe coercive local problem with one distinct trace."""
    return LocalProblem([[2.0]], [[1.0]], [1.0], np.array([index]))


@pytest.mark.parametrize("complex_rhs", [False, True])
def test_direct_refinement_retains_digits_without_changing_criterion(complex_rhs: bool) -> None:
    """A rounding-limited double iterate fails while retained correction digits pass."""
    if not solvers._EXTENDED_PRECISION:
        pytest.skip("NumPy longdouble is not wider than double on this platform")
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0 + 1e-8]])
    rhs = np.array([0.3, -0.3 + 1e-8]) * (1 + 2j if complex_rhs else 1)
    with factorize(matrix, rtol=1e-17) as prepared:
        with pytest.raises(LinearSolveError, match="residual criterion"):
            prepared.solve(rhs)
        result = prepared.solve(rhs, refinement_precision="extended")
        target_type = np.clongdouble if complex_rhs else np.longdouble
        assert result.dtype == target_type
        residual = matrix.astype(target_type) @ result - rhs.astype(target_type)
        assert np.linalg.norm(residual) <= 1e-17 * np.linalg.norm(rhs)
        multi = prepared.solve(np.column_stack((rhs, 2 * rhs)), refinement_precision="extended")
        np.testing.assert_allclose(multi, np.column_stack((result, 2 * result)), atol=1e-16)


def test_refinement_precision_contract_and_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported precision explicitly, without replacing the requested backend."""
    with factorize([[1.0]]) as prepared:
        with pytest.raises(ValueError, match="double or extended"):
            prepared.solve([1.0], refinement_precision="quadruple")
        monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", False)
        with pytest.raises(SolverUnavailableError, match="wider long-double"):
            prepared.solve([1.0], refinement_precision="extended")
    with pytest.raises(ValueError, match="double or extended"):
        independent_problem(0).condense(refinement_precision="quadruple")
    for backend in ("pyamg", "amgx"):
        with pytest.raises(ValueError, match="AMG local"):
            independent_problem(0).condense(backend, refinement_precision="extended")


def test_response_decoding_preserves_precision_and_validation() -> None:
    """Preserve constrained response storage while rejecting complex and invalid arrays."""
    problem = independent_problem(0)
    response = problem.response_from_solution(np.array([[0.5, 0.5]], dtype=np.longdouble))
    assert response.source.dtype == np.longdouble
    assert response.lifts.dtype == np.longdouble
    assert response.reconstruct(np.array([0.2]), np.empty(0)).dtype == np.longdouble
    for invalid in (np.array([[1j, 0]]), np.ones((2, 2)), np.array([[np.nan, 1.0]])):
        with pytest.raises(ValueError, match="condensation solution"):
            problem.response_from_solution(invalid)


@pytest.mark.parametrize("backend", ["serial", "process"])
def test_hybrid_factory_and_darcy_preserve_local_extended_fields(backend: str) -> None:
    """Exercise both preparation routes and physical Darcy reconstruction in wider storage."""
    if not solvers._EXTENDED_PRECISION:
        pytest.skip("NumPy longdouble is not wider than double on this platform")
    options: dict[str, Any] = dict(local_refinement_precision="extended", backend=backend)
    if backend == "process":
        options["workers"] = 2
    system = HybridSystem.from_local_factory(independent_problem, [0, 1], **options)
    result = system.solve()
    for response, field in zip(system.responses, result.fields, strict=True):
        assert response.lifts.dtype == np.longdouble
        assert field.dtype == np.longdouble
        np.testing.assert_allclose(field, 0, atol=2e-15)
    mesh = TriangleMesh.unit_square(1)
    actual = solve_darcy(mesh, dirichlet=affine, **options)
    assert all(values.dtype == np.longdouble for values in actual.pressure)
    assert actual.l2_error(affine) < 2e-12
    assert actual.flux_l2_error([-1, -2]) < 1e-11


@pytest.mark.parametrize(
    "backend,module",
    [
        ("scipy", "scipy"),
        ("pypardiso", "pypardiso"),
        pytest.param("petsc", "petsc4py", marks=pytest.mark.fem),
        pytest.param("cudss", "nvmath.sparse.advanced", marks=pytest.mark.gpu),
    ],
)
def test_native_direct_dispatch_preserves_requested_extended_precision(
    backend: str, module: str
) -> None:
    """Exercise the public facade with real direct factors and a strict physical residual."""
    if not solvers._EXTENDED_PRECISION:
        pytest.skip("NumPy longdouble is not wider than double on this platform")
    pytest.importorskip(module)
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0 + 1e-8]])
    rhs = np.array([0.3, -0.3 + 1e-8])
    actual = solvers.solve_linear(
        matrix, rhs, solver=backend, rtol=1e-17, refinement_precision="extended"
    )
    assert actual.dtype == np.longdouble
    assert np.linalg.norm(matrix.astype(np.longdouble) @ actual - rhs) <= 1e-17 * np.linalg.norm(
        rhs
    )
    columns = np.column_stack((rhs, -2 * rhs))
    multiple = solvers.solve_linear(
        matrix, columns, solver=backend, rtol=1e-17, refinement_precision="extended"
    )
    np.testing.assert_allclose(multiple, np.column_stack((actual, -2 * actual)), atol=1e-16)
