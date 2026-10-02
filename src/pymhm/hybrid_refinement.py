"""Optional defect correction of the original uncondensed hybrid equations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.hybrid import HybridSolution, HybridSystem, LocalResponse, _array
from pymhm.mesh import FloatArray
from pymhm.solvers import (
    LinearSolveError,
    _accurate_residual,
    _refinement_steps,
    _tolerances,
    factorize,
)


@dataclass(frozen=True)
class HybridRefinement:
    """Corrected physical fields and full-equation residual history in input units.

    ``residual_norms`` includes the initial state and every accepted correction.
    It combines the original local rows, unconstrained weak-trace rows and
    supplied physical moments. ``rhs_norm`` uses these same rows and units;
    this algebraic norm is not a finite-element error estimate.
    Persist ``solution.fields`` with their executed evaluation basis: trace and
    coarse coefficients alone cannot replay the added defect-source responses.
    ``solution.residual`` checks free weak-trace and retained-test equations on
    these corrected fields; artificial gauge rows are excluded from that
    diagnostic. Gauge multipliers include all accepted corrections.
    """

    solution: HybridSolution
    residual_norms: tuple[float, ...]
    rhs_norm: float


def refine_hybrid(
    system: HybridSystem,
    solution: HybridSolution,
    *,
    boundary_load: Any = None,
    fixed: dict[int, float] | None = None,
    moments: Sequence[tuple[Sequence[FloatArray], float]] = (),
    max_steps: int = 2,
    rtol: float = 1e-10,
    atol: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> HybridRefinement:
    """Refine ``A_i u_i+B_i lambda=f_i`` and the original weak trace equations.

    Supply the original ``boundary_load`` (zero by default), prescribed trace
    coefficients ``fixed``, and physical ``moments`` as in the initial solve.
    A moment is ``(local_weights, target)``; do not pass reduced gauge rows.
    Distinct test/trial couplings and multiple retained modes are supported.

    Each correction retains the original lifts and condensed operator. Local
    direct factors solve only the new defect source and are released cellwise;
    no full uncondensed matrix is assembled. The default solution is untouched
    unless this function is explicitly called. ``max_steps=0`` checks it only.
    Acceptance requires the full original residual to be at most
    ``max(atol, rtol*rhs_norm)``. Failure raises ``LinearSolveError`` with the
    achieved residual, including stagnation at a precision floor.
    Extended correction storage is explicit and requires a wider NumPy type.
    """
    _refinement_steps(max_steps)
    _tolerances(rtol, atol)
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        from pymhm.solvers import SolverUnavailableError

        raise SolverUnavailableError("extended refinement requires a wider long-double type")
    dtype: Any = np.longdouble if refinement_precision == "extended" else np.float64
    responses = system.responses
    if len(solution.fields) != len(responses) or len(solution.coarse) != len(responses):
        raise ValueError("one field and retained coefficient array per local response required")
    trace = _array(solution.trace, (system.trace_size,), "trace").astype(dtype)
    fields, coarse = [], []
    for response, field, coefficients in zip(
        responses, solution.fields, solution.coarse, strict=True
    ):
        # Validate without discarding the extra digits of an extended input.
        _array(field, response.source.shape, "field")
        _array(coefficients, (response.problem.coarse_basis.shape[1],), "coarse")
        fields.append(np.array(field, dtype=dtype, copy=True))
        coarse.append(np.array(coefficients, dtype=dtype, copy=True))
    trace[:] = solution.trace
    multipliers = np.array(solution.gauge_multipliers, dtype=dtype, copy=True)
    boundary = _array(
        np.zeros(system.trace_size) if boundary_load is None else boundary_load,
        (system.trace_size,),
        "boundary_load",
    )
    prescribed = {} if fixed is None else dict(fixed)
    if (
        any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or not 0 <= i < system.trace_size
            for i in prescribed
        )
        or not np.isfinite(list(prescribed.values())).all()
    ):
        raise ValueError("fixed DOFs must be valid trace indices with finite values")
    if any(trace[i] != value for i, value in prescribed.items()):
        raise ValueError("solution does not satisfy the supplied fixed trace coefficients")
    free = np.setdiff1d(np.arange(system.trace_size), list(prescribed))
    weights = []
    targets = []
    for local_weights, target in moments:
        if len(local_weights) != len(responses) or not np.isfinite(target):
            raise ValueError("one finite weight array per cell and finite moment target required")
        weights.append(
            tuple(
                _array(w, r.source.shape, "moment weights")
                for w, r in zip(local_weights, responses, strict=True)
            )
        )
        targets.append(float(target))
    if len(weights) != len(solution.gauge_multipliers):
        raise ValueError("supply each original physical moment used by the gauged solve")
    fixed_trace = np.zeros(system.trace_size)
    for index, value in prescribed.items():
        fixed_trace[index] = value
    rhs = np.concatenate(
        [r.problem.load - r.problem.coupling @ fixed_trace[r.problem.trace_dofs] for r in responses]
        + [boundary[free], np.asarray(targets)]
    )
    rhs_norm = float(np.linalg.norm(rhs))
    threshold = max(atol, rtol * rhs_norm)
    history = []
    step = 0
    while True:
        defects = []
        weak = boundary.astype(dtype)
        for response, field in zip(responses, fields, strict=True):
            problem = response.problem
            forcing = (
                problem.load.astype(dtype)
                - problem.coupling.astype(dtype) @ trace[problem.trace_dofs]
            )
            defects.append(_accurate_residual(problem.matrix.tocsr(), forcing, field))
            np.add.at(weak, problem.trace_dofs, -problem.test_coupling.astype(dtype).T @ field)
        moment_defects = np.asarray(
            [
                target
                - sum(np.asarray(w, dtype=dtype) @ u for w, u in zip(local, fields, strict=True))
                for local, target in zip(weights, targets, strict=True)
            ],
            dtype=dtype,
        )
        residual = float(np.linalg.norm(np.concatenate([*defects, weak[free], moment_defects])))
        history.append(residual)
        if np.isfinite(residual) and residual <= threshold:
            vector = np.r_[trace, *coarse]
            retained = [
                response.problem.test_basis.T @ defect
                for response, defect in zip(responses, defects, strict=True)
            ]
            reduced = np.concatenate([weak[free], *retained])
            active = np.setdiff1d(np.arange(len(system.rhs)), list(prescribed))
            scale = max(
                float(np.linalg.norm(system.rhs[active])),
                float(np.linalg.norm((abs(system.matrix) @ abs(vector))[active])),
                np.finfo(float).tiny,
            )
            corrected = HybridSolution(
                trace,
                tuple(coarse),
                tuple(fields),
                float(np.linalg.norm(reduced)) / scale,
                multipliers,
            )
            return HybridRefinement(corrected, tuple(history), rhs_norm)
        if step == max_steps or not np.isfinite(residual):
            raise LinearSolveError(
                f"original hybrid residual {residual:.6e} exceeds {threshold:.6e} "
                f"after {step} corrections"
            )
        corrected_responses = []
        for response, defect in zip(responses, defects, strict=True):
            problem = response.problem
            matrix, _ = problem.condensation_system()
            forcing = np.r_[
                np.asarray(defect, dtype=float), np.zeros(problem.coarse_basis.shape[1])
            ]
            with factorize(matrix, solver=local_solver) as factor:
                source = factor.solve(forcing, refinement_precision=refinement_precision)[
                    : len(problem.load)
                ]
            corrected_responses.append(
                LocalResponse(
                    problem.with_load(defect), source, response.lifts, response.coarse_vectors
                )
            )
        correction_system = HybridSystem.from_responses(corrected_responses, boundary_load=weak)
        constraints = [
            correction_system.mean_constraint(w, float(target))
            for w, target in zip(weights, moment_defects, strict=True)
        ]
        correction = correction_system.solve(
            solver=solver,
            fixed={i: 0.0 for i in prescribed},
            constraints=constraints,
            refinement_precision=refinement_precision,
        )
        trace += correction.trace
        multipliers += correction.gauge_multipliers
        fields = [u + du for u, du in zip(fields, correction.fields, strict=True)]
        coarse = [c + dc for c, dc in zip(coarse, correction.coarse, strict=True)]
        step += 1
