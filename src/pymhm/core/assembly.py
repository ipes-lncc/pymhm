"""Ordered local providers and assembly of explicitly declared hybrid forms.

Workers own local construction and solves. Only the coordinator accumulates
shared trace entries, in macrocell order and without intermediate batch sums.
The module requires no optional FEM, MPI or accelerator imports.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np

from pymhm.core.condensation import (
    condense_local,
    local_condensation_system,
    local_response_from_solution,
)
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem, LocalResponse
from pymhm.core.contributions import assemble_hybrid_contributions, local_global_contribution
from pymhm.core.system import HybridSystem, _prepare_solver
from pymhm.core.validation import FloatArray, IntArray
from pymhm.core.variational import GlobalForm
from pymhm.execution.cpu import ExecutionConfig, _prepare_callable, iter_local
from pymhm.linalg.linear import check_linear_solution, validate_invertible

Item = TypeVar("Item")


class LocalLinearSolver(Protocol):
    """Solve all columns of one constrained local system in its declared basis.

    A callable receives owned copies of the augmented sparse operator and
    source/trace/retained-mode right-hand sides. Return a finite real array of
    the same shape as ``rhs``, including moment multipliers. External codes or
    learned response models can implement this contract without inheritance.
    They own their native resources and may implement ``close()`` to release
    resident solver state when local execution ends. The original
    constrained operator has an independent numerical rank check, and each
    column is checked before decoding a local response.
    """

    def __call__(self, matrix: Any, rhs: FloatArray) -> FloatArray:
        """Return coefficients satisfying ``matrix @ result = rhs``."""
        ...


@dataclass(frozen=True)
class SolverConfig:
    """Choose local and global algebra without changing a variational form.

    Named solvers use the existing PyMHM adapters. A callable ``local_solver``
    implements :class:`LocalLinearSolver` and is checked at the unchanged
    relative residual tolerance 1e-10, separately for each right-hand side.
    Explicit precision choices affect native correction arithmetic, not the
    operator, quadrature, gauge or acceptance criterion. A custom solver owns
    its precision policy; use the default local precision setting with it.
    """

    local_solver: str | LocalLinearSolver = "scipy"
    global_solver: str = "scipy"
    local_refinement_precision: Literal["double", "extended"] = "double"
    global_refinement_precision: Literal["double", "extended"] = "double"

    def __post_init__(self) -> None:
        """Reject ambiguous solver and precision settings before executing work."""
        if not (isinstance(self.local_solver, str) or callable(self.local_solver)):
            raise TypeError("local_solver must be a solver name or callable")
        if not isinstance(self.global_solver, str):
            raise TypeError("global_solver must be a solver name")
        if self.local_refinement_precision not in {"double", "extended"}:
            raise ValueError("local_refinement_precision must be double or extended")
        if self.global_refinement_precision not in {"double", "extended"}:
            raise ValueError("global_refinement_precision must be double or extended")
        if callable(self.local_solver) and self.local_refinement_precision != "double":
            raise ValueError("a custom local solver owns its refinement precision")


@dataclass(frozen=True)
class HybridProblem(Generic[Item]):
    """Combine a global hybrid form with an independent local provider.

    ``items`` supplies one specification per macrocell, in the order declared
    by ``global_form.coarse_sizes``. The provider returns ``LocalProblem`` or
    ``LocalAssembly`` with oriented trace maps, actual bases and physical
    moments. It is called exactly once per item per assembly. An iterator is
    consumed once; use a reusable iterable for repeated assemblies. Native
    resources remain worker-owned and must not cross a process boundary.
    A provider may retain reusable workspaces between items and implement
    ``close()`` for worker-lifetime cleanup. Threaded providers must keep
    mutable native workspaces separate across concurrent calls. Providers and
    items must be picklable for the spawn backend.
    ``contribution`` can supply explicit direct global terms in addition to a
    cell's Schur block. By default only the coordinator invokes a custom
    callback, in cell order. ``contribution_execution="worker"`` declares an
    independent, picklable callback that runs after the local solve on that
    worker; it must not write shared state or require coordinator resources.
    The default Schur contribution always runs on its local worker. Only
    the ordered coordinator reduction writes shared global entries. Set
    ``require_local_trace_coverage=False`` when a separate declared global form
    owns the remaining trace coordinates; retained intervals are checked.
    """

    global_form: GlobalForm
    local_provider: Callable[[Item], LocalProblem | LocalAssembly]
    items: Iterable[Item]
    contribution: (
        Callable[[LocalResponse, Any, IntArray], tuple[IntArray, FloatArray, FloatArray]] | None
    ) = None
    require_local_trace_coverage: bool = True
    contribution_execution: Literal["coordinator", "worker"] = "coordinator"

    def __post_init__(self) -> None:
        """Check the global form and provider without consuming local items."""
        if not isinstance(self.global_form, GlobalForm):
            raise TypeError("global_form must be a GlobalForm")
        if not callable(self.local_provider):
            raise TypeError("local_provider must be callable")
        if self.contribution is not None and not callable(self.contribution):
            raise TypeError("contribution must be callable")
        if self.contribution_execution not in {"coordinator", "worker"}:
            raise ValueError("contribution_execution must be coordinator or worker")


_DEFAULT_EXECUTION = ExecutionConfig()
_DEFAULT_SOLVERS = SolverConfig()


def _provide_response(
    item: Item,
    *,
    provider: Callable[[Item], LocalProblem | LocalAssembly],
    solvers: SolverConfig,
) -> tuple[LocalResponse, Any]:
    """Construct and solve one cell entirely within the selected worker."""
    supplied = provider(item)
    if isinstance(supplied, LocalAssembly):
        local, metadata = supplied.problem, supplied.metadata
    elif isinstance(supplied, LocalProblem):
        local, metadata = supplied, None
    else:
        raise TypeError("local_provider must return LocalProblem or LocalAssembly")
    solver = solvers.local_solver
    if isinstance(solver, str):
        response = condense_local(
            local, solver=solver, refinement_precision=solvers.local_refinement_precision
        )
    else:
        matrix, rhs = local_condensation_system(local)
        validate_invertible(matrix)
        # User code may modify its arguments; acceptance uses the original rows.
        result = solver(matrix.copy(), rhs.copy())
        checked = check_linear_solution(matrix, rhs, result)
        response = local_response_from_solution(local, checked)
    return response, metadata


def _provide_cell(
    indexed_item: tuple[int, Item],
    *,
    provider: Callable[[Item], LocalProblem | LocalAssembly],
    solvers: SolverConfig,
    trace_size: int,
    coarse_sizes: tuple[int, ...],
    offsets: IntArray,
    contribution: (
        Callable[[LocalResponse, Any, IntArray], tuple[IntArray, FloatArray, FloatArray]] | None
    ),
    reduce_in_worker: bool,
) -> tuple[LocalResponse, Any, tuple[IntArray, FloatArray, FloatArray] | None]:
    """Own one local solve and its independent reduced block on the same worker.

    Only coefficient arrays leave the worker. Global trace indices remain the
    provider's oriented indices; retained coordinates come from the declared
    item partition. No shared global entry is written here.
    """
    cell, item = indexed_item
    if cell >= len(coarse_sizes):
        raise ValueError("one local item per coarse cell partition is required")
    response, record = _provide_response(item, provider=provider, solvers=solvers)
    local = response.problem
    if local.coarse_basis.shape[1] != coarse_sizes[cell]:
        raise ValueError("local retained basis does not match its coarse cell partition")
    if np.any(local.trace_dofs >= trace_size):
        raise ValueError("local trace map exceeds the declared global trace space")
    reduced = None
    if reduce_in_worker:
        coarse_dofs = np.arange(offsets[cell], offsets[cell + 1])
        reduced = (
            local_global_contribution(response, coarse_dofs)
            if contribution is None
            else contribution(response, record, coarse_dofs)
        )
    return response, record, reduced


@dataclass(frozen=True)
class _CellWorker(Generic[Item]):
    """Own a local provider and its cleanup for one ordered execution lifetime."""

    provider: Callable[[Item], LocalProblem | LocalAssembly]
    solvers: SolverConfig
    trace_size: int
    coarse_sizes: tuple[int, ...]
    offsets: IntArray
    contribution: (
        Callable[[LocalResponse, Any, IntArray], tuple[IntArray, FloatArray, FloatArray]] | None
    )
    reduce_in_worker: bool

    def prepare_runtime(self) -> None:
        """Load explicitly selected backend/provider libraries before native limits."""
        _prepare_callable(self.provider)
        _prepare_solver(self.solvers.local_solver)

    def __call__(
        self, indexed_item: tuple[int, Item]
    ) -> tuple[LocalResponse, Any, tuple[IntArray, FloatArray, FloatArray] | None]:
        """Delegate unchanged local algebra and oriented global contribution."""
        return _provide_cell(
            indexed_item,
            provider=self.provider,
            solvers=self.solvers,
            trace_size=self.trace_size,
            coarse_sizes=self.coarse_sizes,
            offsets=self.offsets,
            contribution=self.contribution,
            reduce_in_worker=self.reduce_in_worker,
        )

    def close(self) -> None:
        """Release the provider and solver's explicitly declared resident resources."""
        try:
            close = getattr(self.provider, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(self.solvers.local_solver, "close", None)
            if callable(close):
                close()


def assemble_hybrid(
    problem: HybridProblem[Item],
    *,
    execution: ExecutionConfig = _DEFAULT_EXECUTION,
    solvers: SolverConfig = _DEFAULT_SOLVERS,
) -> HybridSystem:
    """Assemble a hybrid form using ordered serial work or bounded parallel jobs.

    Serial execution solves and adds each cell before requesting the next item.
    Thread/process execution completes one batch before consuming the next;
    contributions are reduced individually in input order by the coordinator.
    With ``execution.pipeline=True``, a rolling window replaces batch barriers;
    other workers continue while the coordinator scatters the next ordered
    block. The default Schur reduction and explicitly worker-owned callbacks
    run with the local solve, before that handoff.
    Shared faces therefore have no concurrent writes or batch-dependent sums.
    The declared trace and retained layout is checked against every response.
    Local responses are retained for reconstruction and physical mean rows;
    this API bounds in-flight work, not total response/global matrix storage.
    """
    form = problem.global_form
    offsets = np.r_[form.trace_size, form.trace_size + np.cumsum(form.coarse_sizes)].astype(
        np.int64
    )
    responses: list[LocalResponse] = []
    metadata: list[Any] = []
    reduce_in_worker = problem.contribution is None or problem.contribution_execution == "worker"
    computed = iter_local(
        _CellWorker(
            provider=problem.local_provider,
            solvers=solvers,
            trace_size=form.trace_size,
            coarse_sizes=form.coarse_sizes,
            offsets=offsets,
            contribution=problem.contribution if reduce_in_worker else None,
            reduce_in_worker=reduce_in_worker,
        ),
        enumerate(problem.items),
        backend=execution.backend,
        workers=execution.workers,
        native_threads=execution.native_threads,
        batch_size=execution.batch_size,
        pipeline=execution.pipeline,
    )

    def contributions() -> Iterable[tuple[IntArray, FloatArray, FloatArray]]:
        """Yield each response's declared global block before requesting another."""
        for cell, (response, record, reduced) in enumerate(computed):
            responses.append(response)
            metadata.append(record)
            coarse_dofs = np.arange(offsets[cell], offsets[cell + 1])
            if reduced is None:
                # Arbitrary application callbacks retain coordinator ownership.
                assert problem.contribution is not None
                reduced = problem.contribution(response, record, coarse_dofs)
            yield reduced

    try:
        matrix, rhs, load_scale = assemble_hybrid_contributions(
            contributions(),
            trace_size=form.trace_size,
            kernel_offsets=offsets,
            boundary_load=form.boundary_load,
            require_local_trace_coverage=problem.require_local_trace_coverage,
        )
    finally:
        computed.close()
    system = HybridSystem.__new__(HybridSystem)
    system.responses = tuple(responses)
    system.local_metadata = tuple(metadata)
    system.trace_size = form.trace_size
    system.kernel_offsets = offsets
    system.matrix, system.rhs, system.load_scale = matrix, rhs, load_scale
    return system


def solve_hybrid(
    problem: HybridProblem[Item],
    *,
    execution: ExecutionConfig = _DEFAULT_EXECUTION,
    solvers: SolverConfig = _DEFAULT_SOLVERS,
) -> HybridSolution:
    """Assemble, solve and reconstruct with the declared boundaries and gauges.

    ``fixed_trace`` prescribes oriented trace coefficients. ``constraints``
    holds physical rows and targets in global trace/coarse coordinates. The
    unchanged global solver verifies the original equations after eliminating
    prescribed values and enforcing gauges. To derive mean rows from local
    physical weights, call :func:`assemble_hybrid` and ``system.mean_constraint``
    before ``system.solve`` instead.
    """
    system = assemble_hybrid(problem, execution=execution, solvers=solvers)
    form = problem.global_form
    return system.solve(
        solver=solvers.global_solver,
        fixed=dict(form.fixed_trace or {}),
        constraints=list(form.constraints) or None,
        refinement_precision=solvers.global_refinement_precision,
    )
