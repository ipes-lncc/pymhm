"""AMG response accuracy under amplification by physical reconstruction coefficients."""

from contextlib import contextmanager
from math import fsum
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.core import condensation
from pymhm.core.contracts import LocalProblem


def test_local_worker_rejects_provider_data_without_equations() -> None:
    """External local providers must return an explicit problem, not an untyped array."""
    with pytest.raises(TypeError, match="LocalProblem or LocalAssembly"):
        condensation._assemble_and_condense(0, factory=lambda _: np.zeros(2), solver="amgx")


def test_amgx_zero_dimensional_complement_requires_no_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loads entirely carried by the mean multiplier have exactly zero responses."""
    problem = LocalProblem(
        [[0.0]], [[1.0, -2.0]], [3.0], np.arange(2), kernel=np.ones((1, 1)), constraints=[[2.0]]
    )

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        """Reject an unnecessary native resource acquisition for an empty complement."""
        raise AssertionError("no pinned coordinates require a native hierarchy")

    monkeypatch.setattr(condensation, "prepare_amgx", unexpected)
    response = problem.condense("amgx")
    assert_allclose(response.source, 0, rtol=0, atol=0)
    assert_allclose(response.lifts, 0, rtol=0, atol=0)
    assert_allclose(response.retained_basis, 1, rtol=0, atol=0)


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
@pytest.mark.parametrize("kernel_kind", ["none", "exact", "rounded"])
def test_amg_combined_field_preserves_original_physical_residual(
    solver: str, kernel_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admissible per-column defects must not spoil a weighted reconstructed field."""
    matrix = np.eye(2) if kernel_kind == "none" else np.array([[1.0, -1.0], [-1.0, 1.0]])
    if kernel_kind == "rounded":
        matrix[1, 1] += 1e-12
    forcing = np.array([1.0, -1.0])
    arguments = {} if kernel_kind == "none" else {"kernel": np.ones((2, 1))}
    problem = LocalProblem(matrix, forcing[:, None], forcing, np.array([0]), **arguments)
    tolerances = []

    def admissible_defect(operator: Any, rhs: Any, *, solver: str, rtol: float = 1e-10) -> Any:
        """Return a finite iterate at half the requested relative residual tolerance."""
        tolerances.append(rtol)
        return (1 + 0.5 * rtol) * np.linalg.solve(operator.toarray(), rhs)

    monkeypatch.setattr(condensation, "solve_linear", admissible_defect)

    @contextmanager
    def prepared(operator: Any, *, rtol: float) -> Any:
        """Preserve the same admissible defect through the reusable solver interface."""
        yield SimpleNamespace(
            solve=lambda rhs: admissible_defect(operator, rhs, solver="amgx", rtol=rtol)
        )

    monkeypatch.setattr(condensation, "prepare_amgx", prepared)
    response = problem.condense(solver)
    field = response.source - response.lifts[:, 0] * 20.0
    defect = matrix @ field + forcing * 20.0 - forcing
    assert np.linalg.norm(defect) <= 1e-10 * np.linalg.norm(forcing)
    assert len(tolerances) >= 2 and set(tolerances) == {1e-10}
    # The same admissible error budget at 1e-10 fails after combining 20 lifts.
    assert 19 * 0.5 * 1e-10 > 1e-10


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
@pytest.mark.parametrize("has_kernel", [False, True])
def test_amg_accepts_the_last_permitted_original_equation_correction(
    solver: str, has_kernel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A returned fourth correction must satisfy the unchanged original equation bound.

    Controlled backend iterates exercise the outer stopping invariant; native
    pinned-solve accuracy is verified independently by the native integrations.
    """
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]]) if has_kernel else np.eye(2)
    forcing = np.array([1.0, -1.0])
    problem = LocalProblem(
        matrix,
        forcing[:, None],
        forcing,
        np.array([0]),
        **({"kernel": np.ones((2, 1))} if has_kernel else {}),
    )
    requested_tolerances = []

    def controlled_correction(operator: Any, rhs: Any, *, solver: str, rtol: float) -> Any:
        """Return deterministic finite progress without changing the outer acceptance test."""
        requested_tolerances.append(rtol)
        return 0.9995 * np.linalg.solve(operator.toarray(), rhs)

    monkeypatch.setattr(condensation, "solve_linear", controlled_correction)

    @contextmanager
    def prepared(operator: Any, *, rtol: float) -> Any:
        """Use identical controlled progress through the prepared AMG interface."""
        yield SimpleNamespace(
            solve=lambda rhs: controlled_correction(operator, rhs, solver="amgx", rtol=rtol)
        )

    monkeypatch.setattr(condensation, "prepare_amgx", prepared)
    response = problem.condense(solver)
    columns = np.column_stack((response.source, response.lifts))
    loads = np.column_stack((forcing, forcing))
    assert np.all(
        np.linalg.norm(matrix @ columns - loads, axis=0) <= 1e-12 * np.linalg.norm(loads, axis=0)
    )
    assert len(requested_tolerances) == 4
    assert set(requested_tolerances) == {1e-10}
    assert_allclose(problem.constraints.T @ columns, 0.0, rtol=0, atol=1e-15)


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_amg_rejects_final_stagnant_or_inaccurate_original_equations(
    solver: str, gain: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting the correction budget cannot accept an original residual above 1e-12."""
    problem = LocalProblem(np.eye(2), [[1.0], [-1.0]], [1.0, -1.0], np.array([0]))
    calls = []

    def controlled_correction(operator: Any, rhs: Any, *, solver: str, rtol: float) -> Any:
        """Expose finite stagnation or insufficient progress independently of solver internals."""
        calls.append(rtol)
        return gain * np.linalg.solve(operator.toarray(), rhs)

    monkeypatch.setattr(condensation, "solve_linear", controlled_correction)

    @contextmanager
    def prepared(operator: Any, *, rtol: float) -> Any:
        """Forward the same intentionally unconverged corrections to both public paths."""
        yield SimpleNamespace(
            solve=lambda rhs: controlled_correction(operator, rhs, solver="amgx", rtol=rtol)
        )

    monkeypatch.setattr(condensation, "prepare_amgx", prepared)
    with pytest.raises(ValueError, match="constrained residual refinement did not converge"):
        problem.condense(solver)
    assert len(calls) == 4 and set(calls) == {1e-10}


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("modes", [1, 2])
def test_amg_projected_source_preserves_cancelled_compatibility_moment(
    solver: str, wide: bool, modes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small exact source moment survives large cancelling Neumann loads.

    The deliberately stagnant backend exposes its first pinned forcing. The
    test checks compatibility using an independent compensated scalar sum;
    it makes no claim that this backend satisfies the native solver criterion.
    """
    import pymhm.linalg.linear as solvers

    if wide and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("the host does not provide a wider accumulator")
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", wide)
    block = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    matrix = sparse.block_diag([block] * modes, format="csr")
    forcing = np.tile([1e16, 1.0, -1e16], modes)
    kernel = np.kron(np.eye(modes), np.ones((3, 1)))
    problem = LocalProblem(matrix, np.zeros((3 * modes, 1)), forcing, np.array([0]), kernel=kernel)
    pinned = []

    def observe_forcing(operator: Any, rhs: Any, *, solver: str, rtol: float) -> Any:
        """Observe the original projected source without introducing a converged fake solve."""
        pinned.append(np.array(rhs, copy=True))
        return np.zeros_like(rhs)

    monkeypatch.setattr(condensation, "solve_linear", observe_forcing)

    @contextmanager
    def prepared(operator: Any, *, rtol: float) -> Any:
        """Use the same nonconvergent observer for the prepared native-AMG interface."""
        yield SimpleNamespace(
            solve=lambda rhs: observe_forcing(operator, rhs, solver="amgx", rtol=rtol)
        )

    monkeypatch.setattr(condensation, "prepare_amgx", prepared)
    with pytest.raises(ValueError, match="constrained residual refinement did not converge"):
        problem.condense(solver)
    multiplier = fsum(forcing[:3].tolist()) / 3.0
    assert_allclose(pinned[0][::2, 0], forcing[1::3] - multiplier, rtol=0, atol=0)
    assert len(pinned) == 4


@pytest.mark.parametrize(
    "solver",
    ["pyamg", pytest.param("amgx", marks=[pytest.mark.gpu, pytest.mark.serial])],
)
def test_native_amg_neumann_reconstruction_matches_original_kkt(solver: str) -> None:
    """Check native reconstruction, its columnwise error budget and independent KKT.

    The second trace load deliberately exceeds the source scale. Its weighted
    defect budget need not satisfy the physical source-relative criterion;
    local accuracy alone never certifies an arbitrarily amplified field.
    """
    if solver == "amgx":
        pytest.importorskip("pyamgx")
    else:
        pytest.importorskip("pyamg")
    edge = sparse.diags(
        [-np.ones(7), np.r_[1.0, np.full(6, 2.0), 1.0], -np.ones(7)], [-1, 0, 1], format="csr"
    )
    identity = sparse.eye(8, format="csr")
    matrix = (
        sparse.kron(edge, sparse.kron(identity, identity))
        + sparse.kron(identity, sparse.kron(edge, identity))
        + sparse.kron(sparse.kron(identity, identity), edge)
    )
    coordinate = np.linspace(-1, 1, matrix.shape[0])
    forcing = np.asarray(matrix @ coordinate)
    coupling = np.column_stack((forcing, matrix @ np.sin(np.arange(matrix.shape[0]))))
    problem = LocalProblem(
        matrix, coupling, forcing, np.arange(2), kernel=np.ones((matrix.shape[0], 1))
    )
    response = problem.condense(solver)
    reference = problem.condense("scipy")
    columns = np.column_stack((response.source, response.lifts))
    loads = np.column_stack((forcing, coupling))
    defects = np.linalg.norm(matrix @ columns - loads, axis=0)
    assert np.all(defects <= 1e-10 * np.linalg.norm(loads, axis=0))
    assert_allclose(response.retained_basis, reference.retained_basis, rtol=0, atol=1e-12)
    for trace in (np.zeros(2), np.array([20.0, -7.0])):
        field = response.source - response.lifts @ trace
        direct = reference.source - reference.lifts @ trace
        field_defect = np.linalg.norm(matrix @ field + coupling @ trace - forcing)
        budget = defects[0] + np.abs(trace) @ defects[1:]
        roundoff = (
            64
            * np.finfo(float).eps
            * np.linalg.norm(abs(matrix) @ abs(field) + abs(coupling) @ abs(trace) + abs(forcing))
        )
        assert field_defect <= budget + roundoff
        if not np.any(trace):
            assert field_defect <= 1e-10 * np.linalg.norm(forcing)
        assert abs(np.sum(field)) <= 1e-10 * np.linalg.norm(field)
        assert_allclose(field, direct, rtol=1e-9, atol=1e-10)
