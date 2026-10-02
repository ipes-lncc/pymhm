"""Reusable MHM operators for repeated sources and fixed material/geometry.

An offline object keeps the local augmented factorizations, harmonic lifts and
one constrained global factorization. Online calls may change source vectors,
boundary moments and prescribed trace values, but not matrices, spaces, fixed
DOF locations or the definition of physical mean constraints.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from copy import copy
from types import TracebackType
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.hybrid import HybridSolution, HybridSystem, LocalProblem, LocalResponse
from pymhm.mesh import FloatArray
from pymhm.solvers import LinearFactorization, factorize


class LocalFactorCache:
    """Reuse direct factors only for exactly identical constrained local matrices.

    Sources, trace couplings and retained responses may differ between cells.
    Keys contain the canonical CSC shape, index arrays and every numerical byte;
    no tolerance, geometric equivalence or approximate hashing is used. The cache
    owns all factors until ``close`` and is local to one process/thread of use.
    """

    def __init__(self, *, solver: str = "scipy") -> None:
        """Create an empty explicit-lifetime cache for one direct solver backend."""
        self.solver = solver
        self._factors: dict[tuple[Any, ...], LinearFactorization] = {}
        self._closed = False
        self.hits = 0

    @property
    def size(self) -> int:
        """Return the number of distinct retained factorizations."""
        return len(self._factors)

    def condense(self, problem: LocalProblem) -> LocalResponse:
        """Condense one validated local problem using an exact-matrix cache entry."""
        if self._closed:
            raise RuntimeError("local factor cache is closed")
        matrix, rhs = problem.condensation_system()
        matrix = sparse.csc_matrix(matrix, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        key = (
            matrix.shape,
            matrix.indptr.tobytes(),
            matrix.indices.tobytes(),
            matrix.data.tobytes(),
        )
        if key in self._factors:
            self.hits += 1
        else:
            self._factors[key] = factorize(matrix, solver=self.solver)
        return problem.response_from_solution(self._factors[key].solve(rhs))

    def close(self) -> None:
        """Release all factors exactly once and invalidate further use."""
        self._closed = True
        for factor in self._factors.values():
            factor.close()
        self._factors.clear()

    def __enter__(self) -> LocalFactorCache:
        """Enter an open exact-matrix cache context."""
        if self._closed:
            raise RuntimeError("local factor cache is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release cached CPU or accelerator resources on every exit path."""
        self.close()


def condense_cached(
    problems: Iterable[LocalProblem], *, solver: str = "scipy"
) -> tuple[LocalResponse, ...]:
    """Condense ordered cells using exact-matrix reuse within this single call."""
    with LocalFactorCache(solver=solver) as cache:
        return tuple(cache.condense(problem) for problem in problems)


class OfflineLocalProblem:
    """Keep a local augmented factorization and source-independent responses.

    The input ``LocalProblem`` validates the mathematical kernel/basis contract.
    New loads are expressed in that same test space. This object owns native
    solver resources and must be closed; concurrent calls are unsupported.
    """

    def __init__(self, problem: LocalProblem, *, solver: str = "scipy") -> None:
        """Factor once and build the harmonic and retained-mode responses."""
        if not isinstance(problem, LocalProblem):
            raise TypeError("problem must be a LocalProblem")
        self.problem = problem
        matrix, rhs = problem.condensation_system()
        self.augmented_size = matrix.shape[0]
        self.factor = factorize(matrix, solver=solver)
        try:
            self.template = problem.response_from_solution(self.factor.solve(rhs))
        except BaseException:
            self.factor.close()
            raise

    def response(self, load: Any) -> LocalResponse:
        """Solve only the new source response, reusing all homogeneous lifts."""
        problem = self.problem.with_load(load)
        rhs = np.zeros(self.augmented_size)
        rhs[: len(problem.load)] = problem.load
        source = self.factor.solve(rhs)[: len(problem.load)]
        return LocalResponse(problem, source, self.template.lifts, self.template.coarse_vectors)

    def close(self) -> None:
        """Release the reusable local factorization exactly once."""
        self.factor.close()

    def __enter__(self) -> OfflineLocalProblem:
        """Enter an open local-factorization context."""
        self.factor.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release native resources even if an online solve fails."""
        self.close()


class OfflineHybridSystem:
    """Prepare a fixed MHM discretization for multiple source/boundary queries.

    ``moments`` contains ``(local_weights, target)`` pairs, with one physical
    weight vector per local problem. Their rows are fixed; their right-hand
    sides account for each new source lifting. Online ``targets`` may change
    prescribed integrals. ``fixed`` values can change, but their keys cannot.

    This is an algebraic offline/online organization of the supplied method;
    it does not change that method into MsHHO or project arbitrary sources onto
    the polynomial source space of a different formulation.
    """

    def __init__(
        self,
        problems: Iterable[LocalProblem],
        *,
        boundary_load: Any = None,
        fixed: dict[int, float] | None = None,
        moments: Sequence[tuple[Sequence[FloatArray], float]] | None = None,
        local_solver: str = "scipy",
        solver: str = "scipy",
    ) -> None:
        """Prepare local lifts and the constrained global factorization once."""
        self._resources = ExitStack()
        self._closed = False
        self.fixed = {} if fixed is None else dict(fixed)
        self.moments = tuple(
            (tuple(np.array(w, dtype=float, copy=True) for w in weights), float(target))
            for weights, target in (() if moments is None else moments)
        )
        try:
            self.locals = tuple(
                self._resources.enter_context(OfflineLocalProblem(problem, solver=local_solver))
                for problem in problems
            )
            self.system = HybridSystem.from_responses(
                tuple(local.template for local in self.locals), boundary_load=boundary_load
            )
            self.boundary_load = (
                np.zeros(self.system.trace_size)
                if boundary_load is None
                else np.array(boundary_load, dtype=float, copy=True)
            )
            self.constraints = [
                self.system.mean_constraint(w, target) for w, target in self.moments
            ]
            n = len(self.system.rhs)
            indices = np.array(list(self.fixed), dtype=int)
            if (
                any(
                    isinstance(i, bool)
                    or not isinstance(i, (int, np.integer))
                    or i < 0
                    or i >= self.system.trace_size
                    for i in self.fixed
                )
                or not np.isfinite(list(self.fixed.values())).all()
            ):
                raise ValueError("fixed DOFs must be valid trace indices with finite values")
            free = np.setdiff1d(np.arange(n), indices)
            matrix = self.system.matrix[free][:, free]
            if self.constraints:
                rows = np.array([row[free] for row, _ in self.constraints])
                matrix = sparse.bmat([[matrix, rows.T], [rows, None]], format="csc")
            self.factor = (
                self._resources.enter_context(factorize(matrix, solver=solver))
                if matrix.shape[0]
                else None
            )
        except BaseException:
            self.close()
            raise

    def solve(
        self,
        loads: Sequence[Any],
        *,
        boundary_load: Any = None,
        fixed: dict[int, float] | None = None,
        targets: Sequence[float] | None = None,
    ) -> HybridSolution:
        """Solve one new source query without rebuilding or refactoring operators."""
        if self._closed:
            raise RuntimeError("offline system is closed")
        if len(loads) != len(self.locals):
            raise ValueError("one load vector per local problem is required")
        fixed = self.fixed if fixed is None else fixed
        if set(fixed) != set(self.fixed):
            raise ValueError("online fixed DOF locations must match the offline preparation")
        targets = [target for _, target in self.moments] if targets is None else targets
        if len(targets) != len(self.moments):
            raise ValueError("one target per prepared physical moment is required")
        responses = tuple(
            local.response(load) for local, load in zip(self.locals, loads, strict=True)
        )
        boundary = np.asarray(self.boundary_load if boundary_load is None else boundary_load)
        if (
            np.iscomplexobj(boundary)
            or boundary.shape != (self.system.trace_size,)
            or not np.isfinite(boundary).all()
        ):
            raise ValueError("boundary_load must be a real finite vector matching the skeleton")
        system = copy(self.system)
        system.responses = responses
        system.rhs = np.zeros_like(self.system.rhs)
        system.load_scale = np.zeros_like(self.system.load_scale)
        for index, response in enumerate(responses):
            dofs = np.r_[
                response.problem.trace_dofs,
                np.arange(system.kernel_offsets[index], system.kernel_offsets[index + 1]),
            ]
            local_load = response.global_load()
            np.add.at(system.rhs, dofs, local_load)
            np.add.at(system.load_scale, dofs, np.abs(local_load))
        system.rhs[: system.trace_size] -= boundary
        system.load_scale[: system.trace_size] += np.abs(boundary)
        constraints = [
            system.mean_constraint(weights, target)
            for (weights, _), target in zip(self.moments, targets, strict=True)
        ]
        return system.solve(fixed=fixed, constraints=constraints, factorization=self.factor)

    def solve_many(
        self, loads: Iterable[Sequence[Any]], **kwargs: Any
    ) -> tuple[HybridSolution, ...]:
        """Evaluate source cases in order while retaining all offline resources."""
        return tuple(self.solve(case, **kwargs) for case in loads)

    def close(self) -> None:
        """Release all local and global native resources; repeated calls are safe."""
        self._closed = True
        self._resources.close()

    def __enter__(self) -> OfflineHybridSystem:
        """Enter an open offline/online context."""
        if self._closed:
            raise RuntimeError("offline system is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources after the last query or an exception."""
        self.close()
