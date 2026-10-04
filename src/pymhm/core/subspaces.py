"""Exact restriction of prepared local responses to nested skeletal subspaces."""

from typing import Any

import numpy as np

from pymhm.core.contracts import LocalProblem, LocalResponse, _array


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
