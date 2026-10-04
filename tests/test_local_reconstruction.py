"""Direct local fields, executed retained coordinates and independent saddle invariants."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.solvers import SolverUnavailableError, factorize, solve_linear


def local_problem(kind: str) -> LocalProblem:
    """Build a physical constraint complement with distinct trial/test conventions."""
    matrix = np.diag(np.arange(4.0, 9)) + np.diag(np.full(4, 0.3), 1)
    coupling = np.random.default_rng(20).normal(size=(5, 2))
    load = np.arange(0.2, 1.2, 0.2)
    if kind == "invertible":
        return LocalProblem(matrix, coupling, load, np.arange(2))
    if kind in {"kernel", "rounded_kernel"}:
        matrix = np.diag([1.0, 2, 2, 2, 1]) - np.diag(np.ones(4), 1) - np.diag(np.ones(4), -1)
        if kind == "rounded_kernel":
            matrix += 1e-14 * np.eye(5)
        return LocalProblem(
            matrix, coupling, load, np.arange(2), np.ones((5, 1)), np.arange(1.0, 6)[:, None]
        )
    basis = np.column_stack((np.ones(5), np.arange(5.0)))
    test_basis = np.column_stack((np.arange(5.0), np.array([1.0, 0, 1, 0, 1])))
    return LocalProblem(
        matrix,
        coupling,
        load,
        np.arange(2),
        coarse_basis=basis,
        constraints=np.arange(1.0, 6)[:, None] * basis,
        test_basis=test_basis,
        test_constraints=np.arange(2.0, 7)[:, None] * test_basis,
        test_coupling=coupling + 0.2,
    )


@pytest.mark.parametrize("kind", ["invertible", "kernel", "rounded_kernel", "petrov"])
@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("precision", ["double", "extended"])
def test_direct_fields_match_full_responses(kind: str, multiple: bool, precision: str) -> None:
    problem = local_problem(kind)
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        with pytest.raises(SolverUnavailableError, match="wider"):
            problem.reconstruct(
                np.ones(2), np.ones(problem.coarse_basis.shape[1]), refinement_precision=precision
            )
        return
    response = problem.condense(refinement_precision=precision)
    trace = np.array([0.25, -0.75])
    coarse = np.arange(problem.coarse_basis.shape[1], dtype=float) + 0.5
    if multiple:
        trace = trace[:, None] * np.array([1.0, -0.2, 0.0])
        coarse = coarse[:, None] * np.array([0.5, 1.0, -0.3])
        expected = np.column_stack(
            [response.reconstruct(trace[:, column], coarse[:, column]) for column in range(3)]
        )
    else:
        expected = response.reconstruct(trace, coarse)
    assert_allclose(
        problem.condensation_matrix().toarray(), problem.condensation_system()[0].toarray()
    )
    for basis in (None, response.retained_basis):
        actual = problem.reconstruct(
            trace, coarse, retained_basis=basis, refinement_precision=precision
        )
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_direct_petrov_fields_match_original_nonsymmetric_saddle() -> None:
    problem = local_problem("petrov")
    boundary = np.array([0.3, -0.2])
    solution = HybridSystem([problem], boundary_load=boundary).solve()
    expected = np.linalg.solve(
        np.block(
            [
                [problem.matrix.toarray(), problem.coupling],
                [problem.test_coupling.T, np.zeros((2, 2))],
            ]
        ),
        np.r_[problem.load, boundary],
    )
    actual = problem.reconstruct(solution.trace, solution.coarse[0])
    assert_allclose(actual, expected[:5], rtol=2e-12, atol=2e-12)
    assert_allclose(solution.trace, expected[5:], rtol=2e-12, atol=2e-12)


def test_archived_retained_basis_preserves_coordinates_under_rotation() -> None:
    problem = local_problem("petrov")
    response = problem.condense()
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    fresh = LocalProblem(
        problem.matrix,
        problem.coupling,
        problem.load,
        problem.trace_dofs,
        coarse_basis=problem.coarse_basis @ rotation,
        constraints=problem.constraints @ rotation,
        test_basis=problem.test_basis @ rotation,
        test_constraints=problem.test_constraints @ rotation,
        test_coupling=problem.test_coupling,
    )
    trace, coarse = np.array([0.2, 0.3]), np.array([-0.5, 0.75])
    expected = response.reconstruct(trace, coarse)
    assert_allclose(
        fresh.reconstruct(trace, coarse, retained_basis=response.retained_basis),
        expected,
        atol=2e-12,
    )
    assert_allclose(fresh.reconstruct(trace, rotation.T @ coarse), expected, atol=2e-12)
    assert not np.allclose(fresh.reconstruct(trace, coarse), expected)


def test_direct_reconstruction_reuses_caller_owned_factor() -> None:
    problem = local_problem("petrov")
    response = problem.condense()
    with factorize(problem.condensation_matrix()) as factor:
        for coarse in ([0.3, 0.1], [0.2, -0.5]):
            result = problem.reconstruct(
                [0.2, 0.3], coarse, retained_basis=response.retained_basis, factorization=factor
            )
            assert_allclose(
                result, response.reconstruct(np.array([0.2, 0.3]), np.array(coarse)), atol=2e-12
            )
        assert factor.matches(problem.condensation_matrix())
    with factorize(np.eye(2)) as other:
        with pytest.raises(ValueError, match="does not match"):
            problem.reconstruct([0.2, 0.3], [0.1, 0.5], factorization=other)
        assert_allclose(other.solve([1, 2]), [1, 2])


def test_extended_combined_source_and_archived_coordinates_are_preserved() -> None:
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        return
    epsilon = np.longdouble(2) ** -60
    problem = LocalProblem([[1]], [[1]], [1], np.array([0]))
    result = problem.reconstruct(np.array([1 - epsilon]), [], refinement_precision="extended")
    assert result.dtype == np.dtype(np.longdouble)
    assert result[0] == epsilon
    kernel = LocalProblem([[0]], np.empty((1, 0)), [0], np.array([], dtype=int), [[1]])
    coarse = np.array([1 + epsilon])
    assert kernel.reconstruct([], coarse, refinement_precision="extended")[0] == coarse[0]
    assert (
        kernel.reconstruct(
            [], [1], retained_basis=np.array([[1 + epsilon]]), refinement_precision="extended"
        )[0]
        == coarse[0]
    )


@pytest.mark.parametrize(
    ("trace", "coarse", "basis", "match"),
    [
        (0.0, [], None, "vector"),
        (np.ones((2, 1, 1)), [], None, "vector"),
        (np.zeros((2, 0)), [], None, "nonempty"),
        ([1], [], None, "trace"),
        ([1, 2], [1], None, "coarse"),
        ([1j, 2], [], None, "real"),
        ([np.nan, 2], [], None, "finite"),
        (np.ones((2, 2)), np.empty((0, 1)), None, "coarse"),
        ([1, 2], [], np.empty((4, 0)), "retained_basis"),
        ([1, 2], [], np.ones((5, 1), dtype=complex), "real"),
    ],
)
def test_direct_reconstruction_rejects_invalid_coordinates(
    trace: Any, coarse: Any, basis: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        local_problem("invertible").reconstruct(trace, coarse, retained_basis=basis)


def test_convertible_coordinate_inputs_and_precision_validation() -> None:
    problem = local_problem("invertible")
    assert_allclose(problem.reconstruct(["0.2", "0.3"], []), problem.reconstruct([0.2, 0.3], []))
    with pytest.raises(ValueError, match="refinement_precision"):
        problem.reconstruct([0.2, 0.3], [], refinement_precision="invalid")


def native_combination_worker(backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep native factors in their creating process and return portable field arrays."""
    problem = local_problem("petrov")
    response = problem.condense(solver=backend, refinement_precision="extended")
    trace, coarse = np.array([[0.2, -0.1], [0.3, 0.4]]), np.array([[0.1, -0.5], [0.4, 0.2]])
    with factorize(problem.condensation_matrix(), solver=backend) as factor:
        actual = problem.reconstruct(
            trace,
            coarse,
            retained_basis=response.retained_basis,
            factorization=factor,
            refinement_precision="extended",
        )
    expected = np.column_stack([response.reconstruct(trace[:, i], coarse[:, i]) for i in range(2)])
    rhs = np.array([1 + np.longdouble(2) ** -60, 0], dtype=np.longdouble)
    identity = solve_linear(
        np.eye(2), rhs, solver=backend, rtol=1e-22, refinement_precision="extended"
    )
    assert_allclose(identity, rhs, rtol=0, atol=0)
    return actual, expected, identity


@pytest.mark.parametrize("backend", ["scipy", "pypardiso"])
def test_native_direct_fields_preserve_serial_spawn_contract(backend: str) -> None:
    """Native serial and spawned workers execute the same archived trial coordinates."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        return
    if backend == "pypardiso":
        pytest.importorskip("pypardiso")
    serial = native_combination_worker(backend)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
        spawned = executor.submit(native_combination_worker, backend).result(timeout=60)
    for actual, expected, identity in (serial, spawned):
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        assert actual.dtype == np.dtype(np.longdouble)
        assert identity.dtype == np.dtype(np.longdouble)
    assert_allclose(serial[0], spawned[0], rtol=2e-12, atol=2e-12)
