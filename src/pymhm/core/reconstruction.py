"""Physical reconstruction and reduced loads in executed retained coordinates."""

from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import (
    LocalProblem,
    LocalResponse,
    _array,
    _matrix_action,
    _preserved_array,
)
from pymhm.core.validation import FloatArray
from pymhm.linalg.linear import (
    LinearFactorization,
    solve_linear,
)


def reconstruct_local(
    problem: LocalProblem,
    trace: Any,
    coarse: Any,
    *,
    retained_basis: Any = None,
    solver: str = "scipy",
    factorization: LinearFactorization | None = None,
    refinement_precision: Literal["double", "extended"] = "double",
) -> FloatArray:
    """Reconstruct local fields without storing every trace response.

    ``trace`` uses local coupling-column order and ``coarse`` uses retained
    trial coordinates. Vectors produce one field; matching nonempty column
    arrays produce several fields with one factorization. With R the
    volumetric block of the augmented inverse,
    the field is R(f-B trace-A Z coarse)+Z coarse for an arbitrary retained Z.
    This includes nonsymmetric Petrov problems and an empty retained basis.
    ``retained_basis`` supplies an executed E=Z-R A Z for persisted coarse
    coordinates; then the field is R(f-B trace)+E coarse. Its numerical basis
    and orientation belong to the caller's archived coefficient contract.
    Only combined source columns are solved. Explicit extended precision preserves
    forcing and correction digits; backend factors stay double precision.
    A supplied factorization must match the augmented operator exactly and
    remains caller-owned. The linear residual uses the combined augmented
    RHS; full physical compatibility remains a global original-equation check.
    """
    n, k = problem.matrix.shape[0], problem.coarse_basis.shape[1]

    def coefficients(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
        """Validate coefficient dimensions without truncating archived wider values."""
        checked = _array(value, shape, name)
        raw = np.asarray(value)
        return raw.copy() if raw.dtype.kind == "f" else checked

    raw_trace = np.asarray(trace)
    if raw_trace.ndim not in (1, 2) or (raw_trace.ndim == 2 and not raw_trace.shape[1]):
        raise ValueError("trace must be a vector or a nonempty column array")
    suffix = raw_trace.shape[1:]
    local_trace = coefficients(trace, (problem.coupling.shape[1], *suffix), "trace")
    local_coarse = coefficients(coarse, (k, *suffix), "coarse")
    basis = (
        problem.coarse_basis
        if retained_basis is None
        else coefficients(retained_basis, (n, k), "retained_basis")
    )
    dtype = np.result_type(local_trace.dtype, local_coarse.dtype, basis.dtype)
    if refinement_precision == "extended":
        dtype = np.result_type(dtype, np.longdouble)
    load = problem.load[:, None] if suffix else problem.load
    forcing = load.astype(dtype) - np.einsum(
        "ij,j...->i...", problem.coupling, local_trace, dtype=dtype
    )
    if retained_basis is None:
        forcing -= np.einsum("ij,j...->i...", problem._retained_action, local_coarse, dtype=dtype)
    rhs = np.concatenate((forcing, np.zeros((k, *suffix), dtype=dtype)))
    if n == 0 and factorization is None:
        if refinement_precision not in {"double", "extended"}:
            raise ValueError("refinement_precision must be double or extended")
        return forcing
    matrix = problem.condensation_matrix()
    if factorization is not None:
        if not factorization.matches(matrix):
            raise ValueError("prepared factorization does not match the local augmented matrix")
        solved = factorization.solve(rhs, refinement_precision=refinement_precision)
    else:
        solved = solve_linear(matrix, rhs, solver=solver, refinement_precision=refinement_precision)
    return solved[:n] + basis @ local_coarse


def local_condensed_load(
    problem: LocalProblem,
    source_response: Any,
    *,
    load: Any = None,
    corrected_retained: bool | None = None,
) -> FloatArray:
    """Project executed source responses to the original reduced Petrov RHS.

    ``source_response`` is R times the supplied ``load`` (the stored source
    by default). A vector gives one reduced RHS; nonempty column arrays give
    several. Real wider source/load digits are preserved. The trace part is
    test_coupling.T R load. General or rounded retained modes use
    (A.T W).T R load-W.T load; exact represented kernels use -W.T load.
    ``corrected_retained`` can preserve an executed LocalResponse branch.
    It does not change the augmented operator or test/trial conventions.
    """
    raw = np.asarray(source_response)
    if raw.ndim not in (1, 2) or (raw.ndim == 2 and not raw.shape[1]):
        raise ValueError("source_response must be a vector or nonempty column array")
    n = len(problem.load)
    suffix = raw.shape[1:]
    response = _preserved_array(source_response, (n, *suffix), "source_response")
    forcing = problem.load if load is None else load
    if suffix and np.asarray(forcing).ndim == 1:
        forcing = np.broadcast_to(np.asarray(forcing)[:, None], (n, *suffix))
    forcing = _preserved_array(forcing, (n, *suffix), "load")
    if corrected_retained is not None and not isinstance(corrected_retained, (bool, np.bool_)):
        raise ValueError("corrected_retained must be a boolean or None")
    general = problem.coarse_basis.shape[1] != problem.kernel.shape[1] or problem._correct_kernel
    corrected = general if corrected_retained is None else corrected_retained
    coarse_rhs = (
        problem._test_action.T @ response - problem.test_basis.T @ forcing
        if corrected
        else -problem.left_kernel.T @ forcing
    )
    return np.concatenate((problem.test_coupling.T @ response, coarse_rhs), axis=0)


def reconstruct_response(
    response: LocalResponse, trace: FloatArray, coarse: FloatArray
) -> FloatArray:
    """Recover one local coefficient vector from trace and retained amplitudes.

    ``trace`` follows the local coupling columns, and ``coarse`` follows the
    executed retained basis columns. The field is source-lifts@trace+E@coarse;
    wider real operands follow NumPy's existing arithmetic promotion. This
    cached-response operation takes vectors; ``reconstruct_local`` also accepts
    matching nonempty RHS column arrays.
    """
    return response.source - response.lifts @ trace + response.retained_basis @ coarse


def kernel_roundoff_bound(response: LocalResponse, field: FloatArray) -> FloatArray:
    """Bound floating-point evaluation of retained kernel equations.

    For declared left/right kernels only, sum the componentwise row bound
    ``gamma_(nnz_i+1) * (abs(A) @ abs(u))_i`` against ``abs(W)`` and
    include the final test projection's dot-product bound. Here gamma uses
    the stored double-precision operator, even for wider reconstruction.
    No operator entry, represented kernel action or physical residual is
    replaced by zero. General retained modes receive no kernel allowance.
    """
    p = response.problem
    if not p.kernel.shape[1] or not p._correct_kernel:
        return np.zeros(p.coarse_basis.shape[1])
    units = (p.matrix.getnnz(axis=1) + 1) * np.finfo(float).eps
    row_bound = (units / (1 - units)) * (abs(p.matrix) @ abs(field))
    projected = abs(p.test_basis).T @ row_bound
    units_test = (len(field) + 1) * np.finfo(float).eps
    action = _matrix_action(p.matrix, field)
    return projected + units_test / (1 - units_test) * (abs(p.test_basis).T @ abs(action))
