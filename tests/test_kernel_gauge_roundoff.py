"""Check kernel-roundoff acceptance without deleting representable physics."""

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.solvers import _accurate_residual


@pytest.mark.parametrize("scale", [1e-18, 1.0, 1e18])
def test_rounded_kernel_nonzero_mean(scale: float) -> None:
    """Keep nonzero kernel actions while accepting a mean within an explicit error bound."""
    A = scale * np.array([[1.0, -1.0], [-1.0, 1.0]])
    A[0, 0] = np.nextafter(A[0, 0], np.inf)
    p = LocalProblem(
        A, np.empty((2, 0)), np.zeros(2), np.empty(0, dtype=int), kernel=np.ones((2, 1))
    )
    system = HybridSystem([p])
    response = system.responses[0]
    value = system.solve(constraints=[system.mean_constraint([np.array([0.5, 0.5])], 2)])
    np.testing.assert_allclose(value.fields[0], 2, rtol=0, atol=2e-15)
    raw = abs(float(np.sum(_accurate_residual(sparse.csr_matrix(A), np.zeros(2), value.fields[0]))))
    bound = float(response.kernel_roundoff_bound(value.fields[0])[0])
    assert 0 < raw <= bound
    assert value.residual == 0
    assert value.raw_residual > 1e-8 and value.raw_residual_norm > 0
    np.testing.assert_array_equal(p.matrix.toarray(), A)


@pytest.mark.parametrize("scale", [1e-18, 1.0, 1e18])
@pytest.mark.parametrize("kind", ["load", "kernel-reaction", "general-reaction"])
def test_representable_incompatibility_is_not_hidden(scale: float, kind: str) -> None:
    """Reject represented sources and reactions independently of operator units."""
    A = scale * np.array([[1.0, -1.0], [-1.0, 1.0]])
    load = np.zeros(2)
    if kind == "load":
        load[0] = 1e-12 * scale
    else:
        A += np.eye(2) * 1e-12 * scale
    retained = {"coarse_basis" if kind == "general-reaction" else "kernel": np.ones((2, 1))}
    p = LocalProblem(A, np.empty((2, 0)), load, np.empty(0, dtype=int), **retained)
    system = HybridSystem([p])
    with pytest.raises(ValueError, match="gauge changed physical equations"):
        system.solve(constraints=[system.mean_constraint([np.array([0.5, 0.5])], 2)])


@pytest.mark.parametrize("count", [1, 2])
def test_petrov_kernels_and_multiple_moments(count: int) -> None:
    """Project the componentwise certificate with each actual retained test mode."""
    A = np.array([[1.0, -2.0], [-3.0, 6.0]])
    A[0, 0] = np.nextafter(A[0, 0], np.inf)
    A = sparse.block_diag([sparse.csc_matrix(A)] * count).tocsc()
    Z = sparse.block_diag([sparse.csc_matrix([[2.0], [1.0]])] * count).toarray()
    W = sparse.block_diag([sparse.csc_matrix([[3.0], [1.0]])] * count).toarray()
    p = LocalProblem(
        A,
        np.empty((2 * count, 0)),
        np.zeros(2 * count),
        np.empty(0, dtype=int),
        kernel=Z,
        left_kernel=W,
    )
    system = HybridSystem([p])
    constraints = []
    for j in range(count):
        weights = np.zeros(2 * count)
        weights[2 * j : 2 * j + 2] = [1.0, 0.0]
        constraints.append(system.mean_constraint([weights], 2 * (j + 1)))
    value = system.solve(constraints=constraints)
    np.testing.assert_allclose(value.fields[0], Z @ np.arange(1, count + 1), rtol=0, atol=2e-15)
    raw = abs(W.T @ _accurate_residual(A.tocsr(), p.load, value.fields[0]))
    assert np.all(raw <= system.responses[0].kernel_roundoff_bound(value.fields[0]))


@pytest.fixture
def simulated_petsc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the serial PETSc API simulation only for interface regression coverage."""
    from petsc_contract import PETSC

    from pymhm import distributed

    monkeypatch.setattr(distributed, "_optional", lambda *args: PETSC)


@pytest.mark.parametrize("reaction", [0.0, 1e-12])
def test_distributed_kernel_bound(simulated_petsc: Any, reaction: float) -> None:
    """Distributed acceptance uses the same bound and rejects physical reaction."""
    from types import SimpleNamespace

    from pymhm.distributed import solve_distributed

    A = np.array([[1.0, -1.0], [-1.0, 1.0]]) + reaction * np.eye(2)
    A[0, 0] = np.nextafter(A[0, 0], np.inf)
    p = LocalProblem(A, np.eye(2), np.zeros(2), np.arange(2), kernel=np.ones((2, 1)))
    options = dict(
        trace_size=2,
        comm=SimpleNamespace(rank=0, allgather=lambda v: [v], allreduce=lambda v: v),
        fixed={0: 0.0, 1: 0.0},
        moments=[([np.ones(2) / 2], 2.0)],
    )
    if reaction:
        with pytest.raises(ValueError, match="gauge changed"):
            solve_distributed(lambda _: p, [0], **options)
    else:
        value = solve_distributed(lambda _: p, [0], **options)
        np.testing.assert_allclose(value.fields[0], 2, atol=2e-15, rtol=0)
        assert value.residual == 0
        assert value.raw_residual > 1e-8 and value.raw_residual_norm > 0


def test_nested_report_keeps_original_kernel_bound() -> None:
    """A nested field reports the same small retained roundoff without altering data."""
    from pymhm.nested import NestedLocalProblem

    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    A[0, 0] = np.nextafter(A[0, 0], np.inf)
    p = LocalProblem(A, np.eye(2), np.zeros(2), np.arange(2), kernel=np.ones((2, 1)))
    inner = HybridSystem([p])
    value = inner.solve(
        fixed={0: 0.0, 1: 0.0}, constraints=[inner.mean_constraint([np.ones(2) / 2], 2.0)]
    )
    nested = NestedLocalProblem(inner, p, np.arange(2), np.eye(2))
    result = nested.reconstruct(np.r_[value.trace, *value.coarse, np.zeros(2)])
    np.testing.assert_array_equal(result.fields[0], value.fields[0])
    assert result.interior_residual == value.residual


def test_previously_accepted_equations_do_not_consult_kernel_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the complete accepted solve branch unchanged, including reported residuals."""
    from pymhm.hybrid import LocalResponse

    matrix = 1e6 * np.array([[1.0, -1, 0], [-1, 2, -1], [0, -1, 1]])
    matrix += np.diag([1e-5, -2e-5, 1e-5])
    problem = LocalProblem(matrix, np.eye(3), np.zeros(3), np.arange(3), kernel=np.ones((3, 1)))
    system = HybridSystem([problem], boundary_load=np.ones(3))
    expected = system.solve()

    def forbidden(self: LocalResponse, field: np.ndarray) -> np.ndarray:
        """Fail if the existing accepted branch starts evaluating a new allowance."""
        raise AssertionError("an accepted system must not consult the kernel bound")

    monkeypatch.setattr(LocalResponse, "kernel_roundoff_bound", forbidden)
    actual = system.solve()
    for before, after in zip(
        (expected.trace, *expected.coarse, *expected.fields, expected.gauge_multipliers),
        (actual.trace, *actual.coarse, *actual.fields, actual.gauge_multipliers),
        strict=True,
    ):
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
    assert expected.residual == actual.residual


@pytest.mark.parametrize("uncertified", ["trace", "general", "exact-kernel"])
@pytest.mark.parametrize("distributed_case", [False, True])
def test_large_kernel_bound_cannot_hide_uncertified_row(
    simulated_petsc: Any, uncertified: str, distributed_case: bool
) -> None:
    """A certificate belongs to its row, never to another trace or retained equation."""
    from types import SimpleNamespace

    from pymhm.distributed import solve_distributed

    matrix = 1e12 * np.array([[1.0, -1], [-1, 1]])
    matrix[0, 0] = np.nextafter(matrix[0, 0], np.inf)
    first = LocalProblem(
        matrix, np.empty((2, 0)), np.zeros(2), np.empty(0, dtype=int), kernel=np.ones((2, 1))
    )
    if uncertified == "trace":
        second = LocalProblem([[1.0]], [[1.0]], [0.0], np.array([0]))
        target = 1e-3
    elif uncertified == "general":
        second = LocalProblem(
            [[1e-3]], np.empty((1, 0)), [0.0], np.empty(0, dtype=int), coarse_basis=[[1.0]]
        )
        target = 1.0
    else:
        second = LocalProblem(
            [[1.0, -1], [-1, 1]],
            np.empty((2, 0)),
            [1e-3, 0.0],
            np.empty(0, dtype=int),
            kernel=np.ones((2, 1)),
        )
        target = 1.0
    cells = [first, second]
    moments = [
        ([np.ones(2) / 2, np.zeros(len(second.load))], 2.0),
        ([np.zeros(2), np.ones(len(second.load))], target),
    ]
    # The first row's allowance is deliberately greater than the entire
    # incompatible defect of the unrelated row. A norm-only test is invalid.
    assert first.condense().kernel_roundoff_bound(np.full(2, 2.0))[0] > 1e-3
    if distributed_case:
        with pytest.raises(ValueError, match="gauge changed"):
            solve_distributed(
                lambda i: cells[i],
                [0, 1],
                trace_size=1 if uncertified == "trace" else 0,
                comm=SimpleNamespace(
                    rank=0, allgather=lambda value: [value], allreduce=lambda value: value
                ),
                moments=moments,
            )
    else:
        system = HybridSystem(cells)
        with pytest.raises(ValueError, match="gauge changed"):
            system.solve(
                constraints=[system.mean_constraint(weights, target) for weights, target in moments]
            )
