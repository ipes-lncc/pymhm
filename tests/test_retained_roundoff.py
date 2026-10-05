"""Original-equation regressions for finite precision kernel actions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.linalg.linear import _accurate_residual


@pytest.mark.parametrize("solver,left_only", [("scipy", False), ("scipy", True), ("pyamg", False)])
def test_algebraic_kernel_action_is_retained(solver, left_only):
    """A declared numerical null mode cannot delete terms of the original matrix."""
    if solver == "pyamg":
        pytest.importorskip("pyamg")
    A = 1e6 * np.array([[1.0, -1, 0], [-1, 2, -1], [0, -1, 1]])
    if left_only:
        A[0, :2] += [1e-5, -1e-5]
    else:
        A += np.diag([1e-5, -2e-5, 1e-5])
    problem = LocalProblem(A, np.eye(3), np.zeros(3), np.arange(3), kernel=np.ones((3, 1)))
    prescribed = np.array([1.0, 2.0, 3.0]) if left_only else np.ones(3)
    system = HybridSystem([problem], boundary_load=prescribed, local_solver=solver)
    value = system.solve()
    response = system.responses[0]
    assert response.coarse_vectors is not None
    assert_allclose(value.fields[0], prescribed, rtol=0, atol=1e-10)
    defect = _accurate_residual(sparse.csr_matrix(A), -value.trace, value.fields[0])
    assert np.max(abs(defect)) < 2e-8
    expected = -_accurate_residual(sparse.csr_matrix(A), np.zeros((3, 1)), np.ones((3, 1)))
    assert_allclose(problem._retained_action, expected, rtol=0, atol=0)


@pytest.mark.parametrize("wide", [False, True])
def test_retained_action_dtype_and_portable_cancellation(monkeypatch, wide):
    """Numerical-kernel correction works without a wider host floating-point dtype."""
    import pymhm.linalg.linear as solvers
    from pymhm.core.contracts import _matrix_action

    if wide and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("the host does not provide a wider accumulator")
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", wide)
    matrix = sparse.csc_matrix([[1e16, 1, -1e16], [2, -2, 0], [0, 0, 0]])
    values = np.ones((3, 1))
    action = _matrix_action(matrix, values)
    assert action.dtype == values.dtype
    assert_allclose(action[:, 0], [1, 0, 0], rtol=0, atol=0)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_rectangular_moment_action_preserves_output_shape_and_cancelled_values(
    monkeypatch: pytest.MonkeyPatch, wide: bool, multiple: bool
) -> None:
    """Accurate dual moments use the operator's output rows for vectors and columns."""
    import pymhm.linalg.linear as solvers
    from pymhm.core.contracts import _matrix_action

    if wide and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("the host does not provide a wider accumulator")
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", wide)
    matrix = sparse.csr_matrix([[1e16, 1.0, -1e16, 1.0], [-1.0, 2.0, -3.0, 4.0]])
    values = np.column_stack((np.ones(4), np.full(4, 2.0))) if multiple else np.ones(4)
    expected = np.array([[2.0, 4.0], [2.0, 4.0]]) if multiple else np.array([2.0, 2.0])
    result = _matrix_action(matrix, values)
    assert result.dtype == values.dtype
    assert result.shape == expected.shape
    assert_allclose(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("modes", [1, 2])
def test_constraint_admissibility_uses_accurate_pairing_and_rejects_exact_zero(
    monkeypatch: pytest.MonkeyPatch, wide: bool, modes: int
) -> None:
    """A true unit moment is admissible even when ordinary summation erases it."""
    import pymhm.linalg.linear as solvers
    from pymhm.core.contracts import _matrix_action

    if wide and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("the host does not provide a wider accumulator")
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", wide)
    block = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    matrix = sparse.block_diag([block] * modes, format="csr")
    kernel = np.kron(np.eye(modes), np.ones((3, 1)))
    constraints = np.kron(np.eye(modes), np.array([[1e16], [1.0], [-1e16]]))
    problem = LocalProblem(
        matrix,
        np.zeros((3 * modes, 0)),
        np.zeros(3 * modes),
        np.array([], dtype=np.int64),
        kernel=kernel,
        constraints=constraints,
    )
    assert_allclose(
        _matrix_action(sparse.csr_matrix(problem.constraints.T), problem.kernel),
        np.eye(modes),
        rtol=0,
        atol=0,
    )
    excluded = constraints.copy()
    excluded[1::3] = 0.0
    for arguments in (
        {"constraints": excluded},
        {"constraints": constraints, "test_constraints": excluded},
    ):
        with pytest.raises(ValueError, match="pair nonsingularly"):
            LocalProblem(
                matrix,
                np.zeros((3 * modes, 0)),
                np.zeros(3 * modes),
                np.array([], dtype=np.int64),
                kernel=kernel,
                **arguments,
            )


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
@pytest.mark.parametrize("width", [0, 2])
@pytest.mark.parametrize("wide", [False, True])
def test_rectangular_moment_action_handles_empty_rows_columns_and_rhs(
    shape: tuple[int, int], width: int, wide: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty moment operators preserve their declared row and right-side dimensions."""
    import pymhm.linalg.linear as solvers
    from pymhm.core.contracts import _matrix_action

    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", wide)
    result = _matrix_action(sparse.csr_matrix(shape), np.zeros((shape[1], width)))
    assert result.shape == (shape[0], width)
    assert_allclose(result, 0.0, rtol=0, atol=0)


def test_amg_rejects_unresolved_constrained_original_equations(monkeypatch):
    """A failed correction solve cannot return a spuriously accepted condensed operator."""
    import pymhm.core.condensation as hybrid

    matrix = 1e6 * np.array([[1.0, -1, 0], [-1, 2, -1], [0, -1, 1]])
    matrix += np.diag([1e-5, -2e-5, 1e-5])
    problem = LocalProblem(matrix, np.eye(3), np.zeros(3), np.arange(3), kernel=np.ones((3, 1)))
    monkeypatch.setattr(hybrid, "solve_linear", lambda matrix, rhs, **kwargs: np.zeros_like(rhs))
    with pytest.raises(ValueError, match="constrained residual refinement"):
        problem.condense(solver="pyamg")
