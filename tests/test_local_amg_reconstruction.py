"""AMG response accuracy under amplification by physical reconstruction coefficients."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.core import condensation
from pymhm.core.contracts import LocalProblem


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
    response = problem.condense(solver)
    field = response.source - response.lifts[:, 0] * 20.0
    defect = matrix @ field + forcing * 20.0 - forcing
    assert np.linalg.norm(defect) <= 1e-10 * np.linalg.norm(forcing)
    assert len(tolerances) >= 2 and set(tolerances) == {1e-10}
    # The same admissible error budget at 1e-10 fails after combining 20 lifts.
    assert 19 * 0.5 * 1e-10 > 1e-10


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
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
