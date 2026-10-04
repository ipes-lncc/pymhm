"""Exact global restriction of prepared responses for original resolution studies.

The returned field remains represented in the prepared (larger) trace basis.
Only the assembled saddle is restricted; local matrices and harmonic lifts are
shared without copies. This is a coordinate injection, not a fitted projection.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution
from pymhm.core.system import HybridSystem
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.linalg.linear import solve_linear
from pymhm.methods.petrov_galerkin import PGMHMSolution


def p0_injection(coarse: SkeletonSpace, fine: SkeletonSpace) -> sparse.csr_matrix:
    """Embed nested discontinuous scalar P0 partitions exactly, with coefficients one."""
    if coarse.mesh is not fine.mesh or coarse.components != 1 or fine.components != 1:
        raise ValueError("trace spaces must be scalar on the same macro mesh")
    rows, columns = [], []
    for face, (small, large) in enumerate(zip(coarse.faces, fine.faces, strict=True)):
        if small.continuous or large.continuous or any(small.degrees) or any(large.degrees):
            raise ValueError("trace injection requires discontinuous P0 spaces")
        if not np.all(np.isin(small.breaks, large.breaks)):
            raise ValueError("each coarse breakpoint must be present in the fine trace")
        centers = (np.asarray(large.breaks[:-1]) + large.breaks[1:]) / 2
        owners = np.searchsorted(small.breaks, centers, side="right") - 1
        rows.extend(fine.dofs(face))
        columns.extend(coarse.dofs(face)[owners])
    return sparse.coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(fine.size, coarse.size)
    ).tocsr()


def projected_solve(
    system: HybridSystem,
    injection: sparse.spmatrix,
    *,
    fixed: dict[int, float] | None = None,
    constraints: list[tuple[np.ndarray, float]] | None = None,
    solver: str = "scipy",
) -> tuple[HybridSolution, np.ndarray, int]:
    """Solve I.T S I and reconstruct with original responses and original trace coordinates.

    Fixed indices refer to the smaller trace. Constraint rows refer to the
    complete original saddle and are pulled back by I. Retained modes and the
    already assembled physical RHS remain unchanged. Physical compatibility is
    checked against the reduced equations, including any gauge augmentation.
    """
    trace = sparse.csr_matrix(injection)
    if (
        trace.shape[0] != system.trace_size
        or trace.shape[1] < 1
        or not np.isfinite(trace.data).all()
    ):
        raise ValueError("injection must map a nonempty finite trace into the prepared space")
    retained = len(system.rhs) - system.trace_size
    embedding = sparse.block_diag((trace, sparse.eye(retained)), format="csc")
    matrix = embedding.T @ system.matrix @ embedding
    rhs = embedding.T @ system.rhs
    load_scale = abs(embedding).T @ system.load_scale
    fixed = {} if fixed is None else fixed
    if (
        any(i < 0 or i >= trace.shape[1] for i in fixed)
        or not np.isfinite(list(fixed.values())).all()
    ):
        raise ValueError("fixed coefficients must be finite indices of the smaller trace")
    value = np.zeros(len(rhs))
    prescribed = np.array(list(fixed), dtype=int)
    value[prescribed] = list(fixed.values())
    free = np.setdiff1d(np.arange(len(rhs)), prescribed)
    original_matrix = matrix[free][:, free]
    prescribed_action = (matrix @ value)[free]
    original_rhs = rhs[free] - prescribed_action
    operator, forcing = original_matrix, original_rhs
    if constraints:
        rows = np.array([row @ embedding for row, _ in constraints])
        targets = np.array([target for _, target in constraints]) - rows @ value
        operator = sparse.bmat(
            [
                [operator, sparse.csc_matrix(rows[:, free].T)],
                [sparse.csc_matrix(rows[:, free]), None],
            ],
            format="csc",
        )
        forcing = np.r_[forcing, targets]
    solved = solve_linear(operator, forcing, solver=solver)
    value[free] = solved[: len(free)]
    action_scale = abs(original_matrix) @ abs(value[free])
    denominator = max(
        np.linalg.norm(original_rhs),
        np.linalg.norm(rhs[free]),
        np.linalg.norm(load_scale[free]),
        np.linalg.norm(prescribed_action),
        np.linalg.norm(action_scale),
        np.finfo(float).tiny,
    )
    residual = float(np.linalg.norm(original_matrix @ value[free] - original_rhs) / denominator)
    if residual > 1e-8:
        raise ValueError("projected gauge changed the physical equations")
    complete = embedding @ value
    coarse = tuple(
        complete[system.kernel_offsets[i] : system.kernel_offsets[i + 1]]
        for i in range(len(system.responses))
    )
    fields = tuple(
        response.reconstruct(complete[response.problem.trace_dofs], coefficient)
        for response, coefficient in zip(system.responses, coarse, strict=True)
    )
    return (
        HybridSolution(
            complete[: system.trace_size], coarse, fields, residual, solved[len(free) :]
        ),
        value[: trace.shape[1]],
        len(rhs),
    )


def restrict_pgmhm(
    prepared: PGMHMSolution,
    injection: sparse.spmatrix,
    *,
    fixed: dict[int, float] | None = None,
    constraints: list[tuple[np.ndarray, float]] | None = None,
    local_solver: str = "scipy",
    local_refinement_precision: str = "double",
) -> tuple[PGMHMSolution, np.ndarray, int]:
    """Restrict the prepared PGMHM saddle and recompute its zero-mean residual enrichment.

    The physical boundary projection and jump quadrature are the prepared ones.
    Equivalence with a separately assembled smaller space therefore requires the
    same boundary functional; polynomial degree-at-most-k data satisfy that
    condition on either common face partition. Returned coefficients and
    penalties retain the original basis, while the separate vector and integer
    identify the smaller trace and actual global dimension.
    """
    hybrid, small_trace, count = projected_solve(
        prepared.system, injection, fixed=fixed, constraints=constraints
    )
    coefficients = np.r_[hybrid.trace, np.concatenate(hybrid.coarse)]
    extra = [np.zeros_like(base) for base in hybrid.fields]
    for data in prepared.penalties:
        jump = data.jump @ coefficients[data.indices] + data.source_jump - data.prescribed
        for cell, sign, evaluation in data.sides:
            extra[cell] -= sign * data.coefficient * (evaluation.T @ (data.weights * jump))
    enriched = []
    for response, base, load in zip(prepared.system.responses, hybrid.fields, extra, strict=True):
        # Only a new source column is required. Keep original matrices/couplings
        # shared; the saddle imposes the same physical mean complement directly.
        problem = response.problem
        operator = sparse.bmat(
            [
                [problem.matrix, sparse.csc_matrix(problem.test_constraints)],
                [sparse.csc_matrix(problem.constraints.T), None],
            ],
            format="csc",
        )
        forcing = np.r_[-load, np.zeros(problem.constraints.shape[1])]
        correction = solve_linear(
            operator, forcing, solver=local_solver, refinement_precision=local_refinement_precision
        )[: len(load)]
        enriched.append(base + correction)
    return (
        replace(
            prepared,
            pressure=hybrid.fields,
            enriched_pressure=tuple(enriched),
            hybrid=hybrid,
            enrichment_loads=tuple(extra),
        ),
        small_trace,
        count,
    )
