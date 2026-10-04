"""Constrained local elimination and portable local assembly workers."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from scipy import linalg, sparse

from pymhm.core.contracts import LocalAssembly, LocalProblem, LocalResponse, _matrix_action
from pymhm.core.validation import FloatArray
from pymhm.linalg.linear import (
    factorize,
    solve_linear,
)


def condense_local(
    problem: LocalProblem,
    solver: str = "scipy",
    *,
    refinement_precision: Literal["double", "extended"] = "double",
) -> LocalResponse:
    """Factor once for source, trace and retained responses, with explicit precision.

    ``problem`` declares A u+B lambda=f and the oriented local trace columns.
    The response stores R f, R B and the executed retained basis E=Z-R A Z;
    an exactly represented kernel uses Z directly. Constraints, Petrov test
    data and solver residual criteria remain those of the supplied problem.
    Direct factors can retain corrected lifts in extended precision without
    changing their residual criterion. AMG local projection currently accepts
    only the default double-precision accumulation.
    """
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    n, k = problem.matrix.shape[0], problem.coarse_basis.shape[1]
    if n == 0:
        return problem.response_from_solution(np.empty((0, problem.coupling.shape[1] + 1)))
    general = k != problem.kernel.shape[1]
    if solver in {"pyamg", "amgx"}:
        if refinement_precision != "double":
            raise ValueError("AMG local solvers require double refinement precision")
        if general:
            raise ValueError("AMG local solvers do not support a general coarse_basis")
        if not np.array_equal(problem.kernel, problem.left_kernel) or not np.array_equal(
            problem.constraints, problem.test_constraints
        ):
            raise ValueError("AMG kernel projection requires matching test and trial spaces")
        if problem._correct_kernel:
            rhs = np.column_stack((problem.load, problem.coupling, problem._retained_action))
            response = np.zeros_like(rhs)
            _, _, pivots = linalg.qr(problem.kernel.T, pivoting=True)
            free = np.setdiff1d(np.arange(n), pivots[:k])
            pairing = problem.kernel.T @ problem.constraints
            for step in range(4):
                defect = rhs - _matrix_action(problem.matrix, response)
                defect -= problem.constraints @ np.linalg.solve(pairing, problem.kernel.T @ defect)
                norms = np.linalg.norm(rhs, axis=0)
                if step and np.all(np.linalg.norm(defect, axis=0) <= 1e-10 * norms):
                    break
                correction = np.zeros_like(rhs)
                if len(free):
                    correction[free] = solve_linear(
                        problem.matrix[free][:, free], defect[free], solver=solver
                    )
                correction -= problem.kernel @ np.linalg.solve(
                    problem.constraints.T @ problem.kernel, problem.constraints.T @ correction
                )
                response += correction
            else:
                raise ValueError("AMG constrained residual refinement did not converge")
            width = problem.coupling.shape[1] + 1
            return LocalResponse(
                problem,
                response[:, 0],
                response[:, 1:width],
                problem.coarse_basis - response[:, width:],
            )
        rhs = np.column_stack((problem.load, problem.coupling))
        if k:
            # Pin independent kernel coordinates only during the elliptic solve.
            # Afterwards restore the physical mean, with no dense rank update.
            rhs -= problem.constraints @ np.linalg.solve(
                problem.kernel.T @ problem.constraints, problem.kernel.T @ rhs
            )
            _, _, pivots = linalg.qr(problem.kernel.T, pivoting=True)
            free = np.setdiff1d(np.arange(n), pivots[:k])
            response = np.zeros_like(rhs)
            if len(free):
                response[free] = solve_linear(
                    problem.matrix[free][:, free], rhs[free], solver=solver
                )
            response -= problem.kernel @ np.linalg.solve(
                problem.constraints.T @ problem.kernel, problem.constraints.T @ response
            )
        else:
            response = solve_linear(problem.matrix, rhs, solver=solver)
        return LocalResponse(problem, response[:, 0], response[:, 1:])
    matrix, rhs = problem.condensation_system()
    with factorize(matrix, solver=solver) as decomposition:
        return problem.response_from_solution(
            decomposition.solve(rhs, refinement_precision=refinement_precision)
        )


def local_condensation_system(problem: LocalProblem) -> tuple[Any, FloatArray]:
    """Return the constrained operator and source/trace/retained-mode right sides.

    The augmented operator is ``[[A, C_left], [C_right.T, 0]]`` with
    constraint columns scaled down only when they exceed the operator's
    largest entry. This change of auxiliary coordinates prevents constraints
    from dominating small-unit operators without magnifying constraints in
    mixed high-contrast saddles. It leaves the physical complement unchanged.
    Its inverse is never explicitly formed. Backends may factor this matrix
    once, retain the factorization, and solve new source columns online.
    """
    n, k = problem.matrix.shape[0], problem.coarse_basis.shape[1]
    general = k != problem.kernel.shape[1] or problem._correct_kernel
    matrix = problem.condensation_matrix()
    width = problem.coupling.shape[1] + 1
    rhs = np.zeros(
        (n + k, width + (k if general else 0)), dtype=np.result_type(problem.load.dtype, float)
    )
    rhs[:n, 0] = problem.load
    rhs[:n, 1:width] = problem.coupling
    if general:
        rhs[:n, width:] = problem._retained_action
    return matrix, rhs


def local_condensation_matrix(problem: LocalProblem) -> Any:
    """Return the augmented operator without allocating response right-hand sides.

    This is the same scaled left/right constraint operator returned by
    ``condensation_system``. It permits one caller-owned factorization for
    several direct reconstructions, including nonsymmetric Petrov data.
    """
    k = problem.coarse_basis.shape[1]
    matrix = problem.matrix
    if k:
        scale = float(np.max(np.abs(matrix.data), initial=0.0)) or 1.0
        left = problem.test_constraints * np.minimum(
            1.0, scale / np.max(np.abs(problem.test_constraints), axis=0)
        )
        right = problem.constraints * np.minimum(
            1.0, scale / np.max(np.abs(problem.constraints), axis=0)
        )
        matrix = sparse.bmat(
            [
                [matrix, sparse.csc_matrix(left)],
                [sparse.csc_matrix(right.T), None],
            ],
            format="csc",
        )
    return matrix


def local_response_from_solution(problem: LocalProblem, solution: Any) -> LocalResponse:
    """Decode finite real responses, preserving explicitly retained wider precision."""
    n, k = problem.matrix.shape[0], problem.coarse_basis.shape[1]
    width = problem.coupling.shape[1] + 1
    general = k != problem.kernel.shape[1] or problem._correct_kernel
    shape = (n + k, width + (k if general else 0))
    if np.iscomplexobj(solution):
        raise ValueError("condensation solution must be real")
    dtype = np.longdouble if np.asarray(solution).dtype == np.dtype(np.longdouble) else float
    response = np.array(solution, dtype=dtype, copy=True)
    if response.shape != shape or not np.isfinite(response).all():
        raise ValueError(f"condensation solution must be finite with shape {shape}")
    response = response[:n]
    coarse_vectors = problem.coarse_basis - response[:, width:] if general else None
    return LocalResponse(problem, response[:, 0], response[:, 1:width], coarse_vectors)


def _condense(
    problem: LocalProblem,
    solver: str,
    refinement_precision: Literal["double", "extended"] = "double",
) -> LocalResponse:
    """Provide a spawn-pickleable worker for local elimination."""
    return problem.condense(solver, refinement_precision=refinement_precision)


def _assemble_and_condense(
    item: Any,
    *,
    factory: Callable[[Any], LocalProblem | LocalAssembly],
    solver: str,
    refinement_precision: Literal["double", "extended"] = "double",
) -> tuple[LocalResponse, Any]:
    """Keep factory construction and one numerical condensation in the same worker."""
    assembled = factory(item)
    if isinstance(assembled, LocalAssembly):
        problem, metadata = assembled.problem, assembled.metadata
    elif isinstance(assembled, LocalProblem):
        problem, metadata = assembled, None
    else:
        raise TypeError("local factory must return LocalProblem or LocalAssembly")
    return problem.condense(solver, refinement_precision=refinement_precision), metadata
