"""Exact restriction of prepared local responses to nested skeletal subspaces."""

from typing import Any

import numpy as np

from pymhm.core.contracts import LocalProblem, LocalResponse, _array
from pymhm.core.validation import FloatArray, real_array


def moment_complement(moments: Any, *, pivots: Any = None) -> FloatArray:
    """Construct coordinates annihilating independent declared moments.

    ``moments`` is a real ``(n, m)`` matrix: its columns represent linear
    functionals on an ``n``-coefficient vector. A vector declares one moment.
    The returned ``(n, n - m)`` matrix ``Z`` satisfies ``moments.T @ Z = 0``.
    Its free-coordinate rows form the identity, in ascending coordinate order;
    pivot rows enforce the moments. This convention fixes scale and orientation
    without choosing an arbitrary numerical nullspace basis.

    ``pivots`` optionally declares the ``m`` constrained coordinates. Otherwise
    the largest absolute entry is selected for one moment, and column-pivoted
    QR of ``moments.T`` selects coordinates for several moments. Near-ties in
    QR can depend on the numerical platform: archive the executed matrix with
    persisted coefficients, or supply fixed admissible pivots. No physical
    kernel or integration measure is inferred from these algebraic moments.
    """
    matrix = real_array(moments, "moments")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("moments must be a vector or a matrix")
    size, count = matrix.shape
    if count > size:
        raise ValueError("there cannot be more independent moments than coordinates")
    if count and np.linalg.matrix_rank(matrix) != count:
        raise ValueError("moments must be linearly independent")
    if pivots is None:
        if not count:
            selected = np.empty(0, dtype=int)
        elif count == 1:
            selected = np.array([np.argmax(np.abs(matrix[:, 0]))])
        else:
            from scipy.linalg import qr

            _, _, coordinates = qr(matrix.T, mode="economic", pivoting=True)
            selected = np.sort(coordinates[:count])
    else:
        selected = np.asarray(pivots)
        if (
            selected.shape != (count,)
            or (selected.size and selected.dtype.kind not in "iu")
            or np.any(selected < 0)
            or np.any(selected >= size)
            or len(np.unique(selected)) != count
        ):
            raise ValueError("pivots must be distinct valid integer coordinates, one per moment")
        selected = selected.astype(int, copy=False)
    if count and np.linalg.matrix_rank(matrix[selected]) != count:
        raise ValueError("the selected pivot coordinates must determine all moments")
    free = np.setdiff1d(np.arange(size), selected)
    basis = np.eye(size)[:, free]
    if count == 1:
        basis[selected] = -matrix[free, 0] / matrix[selected[0], 0]
    elif count:
        basis[selected] = np.linalg.solve(matrix[selected].T, -matrix[free].T)
    return basis


def restrict_response(
    response: LocalResponse, injection: Any, trace_dofs: Any, *, test_injection: Any = None
) -> LocalResponse:
    """Reuse harmonic lifts on an exactly embedded trial/test trace subspace.

    ``injection`` maps the new trace coefficients to the prepared trial space.
    ``test_injection`` defaults to the same map, or independently restricts the
    Petrov test space. Both maps must have full column rank. Their geometrical
    meaning is supplied by the caller; this operation does not project or fit
    unresolved traces. The local operator, source and retained spaces are fixed.
    """
    problem = response.problem
    dofs = np.asarray(trace_dofs)
    if dofs.ndim != 1:
        raise ValueError("trace_dofs must be a vector")
    trial = _array(injection, (len(problem.trace_dofs), len(dofs)), "injection")
    test = (
        trial if test_injection is None else _array(test_injection, trial.shape, "test_injection")
    )
    if not len(dofs) or any(np.linalg.matrix_rank(matrix) != len(dofs) for matrix in (trial, test)):
        raise ValueError("trial and test injections must have full column rank")
    narrowed = LocalProblem(
        problem.matrix,
        problem.coupling @ trial,
        problem.load,
        dofs,
        kernel=problem.kernel if problem.kernel.shape[1] else None,
        coarse_basis=problem.coarse_basis if not problem.kernel.shape[1] else None,
        constraints=problem.constraints,
        test_coupling=problem.test_coupling @ test,
        left_kernel=problem.left_kernel if problem.kernel.shape[1] else None,
        test_basis=problem.test_basis if not problem.kernel.shape[1] else None,
        test_constraints=problem.test_constraints,
    )
    return LocalResponse(narrowed, response.source, response.lifts @ trial, response.coarse_vectors)
