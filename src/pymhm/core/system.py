"""Global hybrid layout, boundary elimination, gauges and physical compatibility."""

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import linalg, sparse

from pymhm.core.condensation import _assemble_and_condense, _condense
from pymhm.core.contracts import (
    HybridSolution,
    LocalAssembly,
    LocalProblem,
    LocalResponse,
    _array,
    _preserved_array,
)
from pymhm.core.contributions import assemble_hybrid_contributions
from pymhm.core.validation import FloatArray, IntArray
from pymhm.execution.cpu import map_local
from pymhm.linalg.linear import (
    LinearFactorization,
    SolverUnavailableError,
    solve_linear,
    validate_invertible,
)


class HybridSystem:
    """Assemble the global MHM equations independently of the local FEM backend.

    The unknown is (lambda, c). Dirichlet traces enter through ``boundary_load``
    as integrals against oriented skeleton basis functions. Prescribed Neumann
    flux coefficients are passed to ``solve`` as ``fixed`` DOFs. Arbitrary linear
    mean constraints can remove global pressure or rigid-motion nullspaces.
    """

    def __init__(
        self,
        problems: list[LocalProblem] | tuple[LocalProblem, ...],
        *,
        boundary_load: Any = None,
        local_solver: str = "scipy",
        local_refinement_precision: Literal["double", "extended"] = "double",
        backend: Literal["serial", "thread", "process"] = "serial",
        workers: int | None = None,
        native_threads: int | None = 1,
        batch_size: int | None = None,
    ) -> None:
        """Condense independent subdomains and assemble their sparse contributions."""
        responses = tuple(
            map_local(
                partial(
                    _condense,
                    solver=local_solver,
                    refinement_precision=local_refinement_precision,
                ),
                problems,
                backend=backend,
                workers=workers,
                native_threads=native_threads,
                batch_size=batch_size,
            )
        )
        self._assemble_global(responses, (None,) * len(responses), boundary_load)

    @classmethod
    def from_local_factory(
        cls,
        factory: Callable[[Any], LocalProblem | LocalAssembly],
        items: Iterable[Any],
        *,
        boundary_load: Any = None,
        local_solver: str = "scipy",
        local_refinement_precision: Literal["double", "extended"] = "double",
        backend: Literal["serial", "thread", "process"] = "serial",
        workers: int | None = None,
        native_threads: int | None = 1,
        batch_size: int | None = None,
    ) -> "HybridSystem":
        """Assemble and condense each independent local problem inside its worker.

        ``factory(item)`` returns ``LocalProblem`` or ``LocalAssembly``. The
        factory and one condensation execute together, exactly once per item;
        global assembly and solution remain in the parent. Input order defines
        response order and ``local_metadata``, including ``None`` for a bare
        problem. Exceptions propagate without a serial fallback.

        Serial and thread backends accept closures. The portable spawn backend
        requires picklable factories, items and metadata, and an executable
        script protected by ``if __name__ == '__main__':``. Native FEM, PETSc,
        MPI and CUDA resources must be created and released inside a worker,
        never sent to or returned from a worker. ``native_threads`` defaults to
        one thread; backend library thread safety is the factory's concern.
        ``batch_size`` limits submitted local jobs while preserving input order.
        Responses are retained before global assembly because this constructor
        infers the global layout. For immediate serial contribution with an
        explicitly declared layout, use ``assemble_hybrid``.
        """
        assembled = map_local(
            partial(
                _assemble_and_condense,
                factory=factory,
                solver=local_solver,
                refinement_precision=local_refinement_precision,
            ),
            items,
            backend=backend,
            workers=workers,
            native_threads=native_threads,
            batch_size=batch_size,
        )
        system = cls.__new__(cls)
        system._assemble_global(
            tuple(response for response, _ in assembled),
            tuple(metadata for _, metadata in assembled),
            boundary_load,
        )
        return system

    @classmethod
    def from_responses(
        cls,
        responses: Iterable[LocalResponse],
        *,
        boundary_load: Any = None,
        metadata: Iterable[Any] | None = None,
    ) -> "HybridSystem":
        """Assemble cached local responses without repeating any factorization."""
        values = tuple(responses)
        if any(not isinstance(value, LocalResponse) for value in values):
            raise TypeError("responses must contain LocalResponse objects")
        records = (None,) * len(values) if metadata is None else tuple(metadata)
        if len(records) != len(values):
            raise ValueError("one metadata record per local response is required")
        system = cls.__new__(cls)
        system._assemble_global(values, records, boundary_load)
        return system

    @classmethod
    def from_contributions(
        cls,
        contributions: Iterable[tuple[IntArray, FloatArray, FloatArray]],
        *,
        trace_size: int,
        coarse_sizes: Iterable[int],
        boundary_load: Any = None,
    ) -> "HybridSystem":
        """Assemble compact cell contributions without retaining local lifts.

        Each tuple is the output of ``LocalResponse.global_contribution`` with
        coarse indices numbered after the ``trace_size`` skeleton unknowns.
        ``coarse_sizes`` declares their ordered cell partition. ``solve`` returns
        trace/coarse coefficients and an empty field tuple; callers reconstruct
        one local response at a time using the executed, persisted local basis.
        Physical mean rows must be supplied explicitly to ``solve``: this compact
        representation cannot derive them or certify kernel evaluation roundoff.
        """
        sizes = tuple(coarse_sizes)
        if (
            isinstance(trace_size, (bool, np.bool_))
            or not isinstance(trace_size, (int, np.integer))
            or trace_size < 0
            or not sizes
            or any(
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or size < 0
                for size in sizes
            )
        ):
            raise ValueError("trace_size and each coarse size must be nonnegative integers")
        values = tuple(contributions)
        if len(values) != len(sizes):
            raise ValueError("one contribution per coarse cell partition is required")
        system = cls.__new__(cls)
        system.responses = ()
        system.local_metadata = ()
        system.trace_size = int(trace_size)
        system.kernel_offsets = np.r_[trace_size, trace_size + np.cumsum(sizes)].astype(np.int64)
        system._assemble_contributions(values, boundary_load)
        return system

    def _assemble_global(
        self,
        responses: tuple[LocalResponse, ...],
        metadata: tuple[Any, ...],
        boundary_load: Any,
    ) -> None:
        """Initialize one global skeleton system from already condensed local responses."""
        if not responses:
            raise ValueError("at least one local problem is required")
        self.responses = responses
        self.local_metadata = metadata
        problems = tuple(response.problem for response in responses)
        all_dofs = np.concatenate([p.trace_dofs for p in problems])
        self.trace_size = int(all_dofs.max()) + 1 if len(all_dofs) else 0
        if not np.array_equal(np.unique(all_dofs), np.arange(self.trace_size)):
            raise ValueError("global trace numbering must be contiguous")
        self.kernel_offsets = np.r_[
            self.trace_size,
            self.trace_size + np.cumsum([p.coarse_basis.shape[1] for p in problems]),
        ].astype(np.int64)
        contributions = (
            response.global_contribution(
                np.arange(self.kernel_offsets[cell], self.kernel_offsets[cell + 1])
            )
            for cell, response in enumerate(self.responses)
        )
        self._assemble_contributions(contributions, boundary_load)

    def _assemble_contributions(
        self,
        contributions: Iterable[tuple[IntArray, FloatArray, FloatArray]],
        boundary_load: Any,
    ) -> None:
        """Accumulate cell blocks and loads without truncating wider real arithmetic."""
        self.matrix, self.rhs, self.load_scale = assemble_hybrid_contributions(
            contributions,
            trace_size=self.trace_size,
            kernel_offsets=self.kernel_offsets,
            boundary_load=boundary_load,
        )

    def mean_constraint(
        self, local_weights: list[FloatArray] | tuple[FloatArray, ...], value: float = 0.0
    ) -> tuple[FloatArray, float]:
        """Express a prescribed integral of reconstructed fields as ``r.T x=b``.

        For incompressible flow, weights integrate only the pressure components;
        for scalar pure Neumann diffusion, they integrate the pressure field.
        """
        return hybrid_mean_constraint(self, local_weights, value)

    def with_rhs(self, rhs: Any, *, load_scale: Any = None) -> "HybridSystem":
        """Reuse the executed condensed matrix with a new explicit reduced load.

        This compact online system retains the original trace numbering and
        ordered retained partitions, and returns no reconstructed fields.
        ``load_scale`` is the sum of absolute contributions before cancellation;
        omitting it uses ``abs(rhs)``. Wider real floating loads are preserved.
        Physical moment rows must be supplied explicitly to ``solve``.
        """
        shape = self.rhs.shape
        load = _preserved_array(rhs, shape, "rhs")
        scale = (
            np.abs(load)
            if load_scale is None
            else _preserved_array(load_scale, shape, "load_scale")
        )
        if np.any(scale < 0) or np.any(scale < np.abs(load)):
            raise ValueError("load_scale must bound the absolute reduced load")
        system = self.__class__.__new__(self.__class__)
        system.responses = ()
        system.local_metadata = ()
        system.trace_size = self.trace_size
        system.kernel_offsets = self.kernel_offsets.copy()
        system.matrix = self.matrix
        system.rhs = load
        system.load_scale = scale
        return system

    def solve(
        self,
        *,
        solver: str = "scipy",
        fixed: dict[int, float] | None = None,
        constraints: list[tuple[FloatArray, float]] | None = None,
        factorization: LinearFactorization | None = None,
        refinement_precision: Literal["double", "extended"] = "double",
        rtol: float | None = None,
    ) -> HybridSolution:
        """Solve with boundary elimination and physical checks; see `solve_hybrid_system`."""
        return solve_hybrid_system(
            self,
            solver=solver,
            fixed=fixed,
            constraints=constraints,
            factorization=factorization,
            refinement_precision=refinement_precision,
            rtol=rtol,
        )


def solve_hybrid_system(
    system: HybridSystem,
    *,
    solver: str = "scipy",
    fixed: dict[int, float] | None = None,
    constraints: list[tuple[FloatArray, float]] | None = None,
    factorization: LinearFactorization | None = None,
    refinement_precision: Literal["double", "extended"] = "double",
    rtol: float | None = None,
) -> HybridSolution:
    """Solve the saddle system, enforce gauges and verify original equations.

    A singular skeleton typically indicates missing gauges or a trace space
    richer than the local response space. Such systems are never repaired by
    an undocumented diagonal perturbation or least-squares solution. For a
    nonsymmetric global operator, the symmetric gauge augmentation requires
    constraint rows pairing with both left and right nullspaces.
    A supplied ``factorization`` must match the complete constrained matrix
    exactly. It is reused without transferring ownership, allowing repeated
    sources and boundary values through an offline/online preparation.
    Explicit ``refinement_precision="extended"`` retains correction digits
    in trace, coarse coefficients and reconstructed fields. Backend factors
    remain double precision. ``rtol`` selects the global linear-solver
    relative residual tolerance; ``None`` uses 1e-10 for a fresh solve or
    the tolerance of a supplied factorization. An explicit tolerance must
    match that prepared factorization. Physical compatibility and local
    reconstruction criteria remain unchanged. Extended arithmetic requires
    a wider NumPy long-double type.
    The original-equation residual uses the absolute local load contributions
    before assembly, and the net prescribed action before elimination. This
    accounts for cancellation roundoff without applying a relative tolerance
    to cancelled large prescribed fluxes or to artificial gauge equations.
    Prescribed elimination also admits its componentwise floating-point
    bound ``gamma_(m+1) * abs(A_fixed) @ abs(x_fixed)`` per physical row,
    where ``m`` is its number of terms and gamma uses double precision.
    A gauged declared kernel can additionally use the componentwise
    evaluation bound of its retained physical rows. This bound is consulted
    only after the usual compatibility test fails; it changes neither the
    represented operator nor the raw residual or previously accepted result.
    General retained modes receive no kernel allowance.
    """
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if rtol is not None and (np.iscomplexobj(rtol) or not np.isfinite(rtol) or rtol <= 0):
        raise ValueError("global rtol must be finite and positive")
    if factorization is not None and rtol is not None and rtol != factorization.rtol:
        raise ValueError("explicit rtol must match the prepared factorization tolerance")
    solve_rtol = 1e-10 if rtol is None else float(rtol)
    extended = refinement_precision == "extended"
    if extended and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise SolverUnavailableError("extended refinement requires a wider long-double type")
    fixed = {} if fixed is None else fixed
    n = len(system.rhs)
    solution = np.zeros(n, dtype=np.longdouble if extended else float)
    indices = np.array(list(fixed), dtype=int)
    if (
        any(
            not isinstance(i, (int, np.integer))
            or isinstance(i, bool)
            or i < 0
            or i >= system.trace_size
            for i in fixed
        )
        or not np.isfinite(list(fixed.values())).all()
    ):
        raise ValueError("fixed DOFs must be valid trace indices with finite values")
    solution[indices] = list(fixed.values())
    free = np.setdiff1d(np.arange(n), indices)
    matrix = system.matrix[free][:, free]
    prescribed_action = (system.matrix @ solution)[free]
    prescribed_matrix = system.matrix[free][:, indices]
    rounding_units = (prescribed_matrix.getnnz(axis=1) + 1) * np.finfo(float).eps
    prescribed_roundoff = (rounding_units / (1 - rounding_units)) * (
        abs(prescribed_matrix) @ np.abs(solution[indices])
    )
    rhs = system.rhs[free] - prescribed_action
    physical_matrix, physical_rhs = matrix, rhs
    count = len(constraints) if constraints else 0
    if constraints:
        array = _preserved_array if extended else _array
        rows = np.array([array(row, (n,), "constraint") for row, _ in constraints])
        targets = (
            array([target for _, target in constraints], (count,), "constraint targets")
            - rows @ solution
        )
        matrix = sparse.bmat(
            [
                [matrix, sparse.csc_matrix(rows[:, free].T)],
                [sparse.csc_matrix(rows[:, free]), None],
            ],
            format="csc",
        )
        rhs = np.r_[rhs, targets]
    if factorization is not None and (not len(rhs) or not factorization.matches(matrix)):
        raise ValueError("prepared factorization does not match the constrained global matrix")
    if len(rhs) and solver != "scipy" and factorization is None:
        validate_invertible(matrix)
    if factorization is not None:
        solved = (
            factorization.solve(rhs, refinement_precision="extended")
            if extended
            else factorization.solve(rhs)
        )
    else:
        if extended:
            solved = (
                solve_linear(
                    matrix, rhs, solver=solver, rtol=solve_rtol, refinement_precision="extended"
                )
                if len(rhs)
                else np.empty(0, dtype=np.longdouble)
            )
        else:
            solved = (
                solve_linear(matrix, rhs, solver=solver, rtol=solve_rtol)
                if len(rhs)
                else np.empty(0)
            )
    solution[free] = solved[: len(free)]
    multipliers = solved[len(free) :]
    action_scale = abs(physical_matrix) @ np.abs(solution[free])
    norm = (
        np.linalg.norm
        if extended or system.matrix.dtype.itemsize > np.dtype(float).itemsize
        else linalg.norm
    )
    physical_defect = physical_matrix @ solution[free] - physical_rhs
    raw_residual_norm = float(norm(physical_defect))
    physical_scale = max(
        norm(system.rhs[free]),
        norm(system.load_scale[free]),
        norm(prescribed_action),
        norm(physical_rhs),
        norm(action_scale),
        norm(prescribed_roundoff) / 1e-8,
        np.finfo(float).tiny,
    )
    raw_residual = float(norm(physical_defect) / physical_scale)
    residual = raw_residual
    if residual > 1e-8 and constraints:
        bounds = np.zeros(n)
        for cell, response in enumerate(system.responses):
            first, last = system.kernel_offsets[cell : cell + 2]
            field = response.reconstruct(
                solution[response.problem.trace_dofs], solution[first:last]
            )
            bounds[first:last] = response.kernel_roundoff_bound(field)
        # This is an absolute evaluation-error certificate, not a relaxed
        # relative tolerance on the small, cancelled retained equation.
        certified_defect = np.maximum(abs(physical_defect) - bounds[free], 0)
        residual = float(norm(certified_defect) / physical_scale)
    if residual > 1e-8:
        raise ValueError("incompatible data: gauge changed physical equations")
    coarse = tuple(
        solution[system.kernel_offsets[i] : system.kernel_offsets[i + 1]].copy()
        for i in range(len(system.kernel_offsets) - 1)
    )
    fields = (
        tuple(
            response.reconstruct(solution[response.problem.trace_dofs], c)
            for response, c in zip(system.responses, coarse, strict=True)
        )
        if system.responses
        else ()
    )
    return HybridSolution(
        solution[: system.trace_size],
        coarse,
        fields,
        residual,
        multipliers,
        raw_residual,
        raw_residual_norm,
    )


def hybrid_mean_constraint(
    system: HybridSystem,
    local_weights: list[FloatArray] | tuple[FloatArray, ...],
    value: float = 0.0,
) -> tuple[FloatArray, float]:
    """Express a prescribed integral of reconstructed fields as ``r.T x=b``.

    For incompressible flow, weights integrate only the pressure components;
    for scalar pure Neumann diffusion, they integrate the pressure field.
    """
    if not system.responses:
        raise ValueError("compact contributions require explicit physical mean rows")
    if len(local_weights) != len(system.responses) or not np.isfinite(value):
        raise ValueError("one weight vector per local problem and finite value required")
    weights = tuple(
        _preserved_array(w, r.source.shape, "local_weights")
        for w, r in zip(local_weights, system.responses, strict=True)
    )
    moment_value = _preserved_array([value], (1,), "value")[0]
    dtype = np.result_type(
        np.asarray(moment_value).dtype,
        float,
        *(response.source.dtype for response in system.responses),
        *(response.lifts.dtype for response in system.responses),
        *(response.retained_basis.dtype for response in system.responses),
        *(w.dtype for w in weights),
    )
    row = np.zeros(len(system.rhs), dtype=dtype)
    target = np.asarray(moment_value, dtype=dtype).item()
    for i, (response, local) in enumerate(zip(system.responses, weights, strict=True)):
        np.add.at(row, response.problem.trace_dofs, -response.lifts.T @ local)
        row[system.kernel_offsets[i] : system.kernel_offsets[i + 1]] = (
            response.retained_basis.T @ local
        )
        target -= local @ response.source
    return row, target
