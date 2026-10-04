"""Reuse local inverse responses only under identical executed algebraic contracts."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.hybrid import LocalProblem, LocalResponse
from pymhm.mesh import positive_int
from pymhm.solvers import LinearFactorization, factorize

_OPERATORS = (
    "coupling",
    "test_coupling",
    "kernel",
    "left_kernel",
    "coarse_basis",
    "test_basis",
    "constraints",
    "test_constraints",
)


def array_identity(value: Any) -> str:
    """Digest shape, executed dtype/mantissa and actual coefficient bytes.

    This is a byte-identity rule, including signed zeros and native extended
    storage. It never equates numerically close or permuted matrices. Padding
    in an extended host representation may cause safe cache misses.
    """
    value = np.asarray(value)
    header = {
        "shape": value.shape,
        "dtype": value.dtype.str,
        "mantissa_bits": np.finfo(value.dtype).nmant if value.dtype.kind in "fc" else None,
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def operator_identity(problem: LocalProblem) -> str:
    """Identify A, B, test coupling, kernels, retained bases and left/right moments.

    Global trace IDs belong to the caller's explicit injection map, so they
    are retained by each new problem instead of entering a local inverse key.
    The independently fingerprinted source is not part of the inverse operator.
    """
    identity: dict[str, Any] = {name: array_identity(getattr(problem, name)) for name in _OPERATORS}
    identity["matrix"] = {
        "format": problem.matrix.format,
        "shape": problem.matrix.shape,
        **{
            name: array_identity(getattr(problem.matrix, name))
            for name in ("data", "indices", "indptr")
        },
    }
    identity["correct_kernel"] = problem._correct_kernel
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def readonly(value: np.ndarray | None) -> np.ndarray | None:
    """Own immutable executed coefficients so one response cannot corrupt another."""
    if value is None:
        return None
    result = value.copy()
    result.setflags(write=False)
    return result


@dataclass
class CachedOperator:
    """One native inverse, actual harmonic/retained basis and exact source responses."""

    factor: LinearFactorization
    lifts: np.ndarray
    coarse_vectors: np.ndarray | None
    sources: dict[str, np.ndarray]
    resources: ExitStack


class ExactResponseCache:
    """Serial context-managed cache with every executed local input fingerprinted.

    Each response retains its caller's LocalProblem and global trace IDs. Exact
    repeated source bytes reuse their original coefficients; changed sources
    use the shared original local reconstruction through the cached factor.
    Harmonic/retained coefficients are immutable and keep their executed dtype.
    All native factors close on normal exit or any exception. There is no
    geometry, material, case-name or wave-frequency exemption from identity.
    """

    def __init__(
        self,
        *,
        solver: str = "scipy",
        refinement_precision: Literal["double", "extended"] = "double",
        max_operators: int = 32,
        max_sources_per_operator: int = 32,
    ) -> None:
        """Declare the same native factor and correction policy as a direct condensation."""
        if refinement_precision not in ("double", "extended"):
            raise ValueError("refinement precision must be double or extended")
        if solver in ("pyamg", "amgx"):
            raise ValueError("exact response caching requires a native direct factorization")
        self.solver, self.precision = solver, refinement_precision
        self.max_operators = positive_int(max_operators, "max_operators")
        self.max_sources = positive_int(max_sources_per_operator, "max_sources_per_operator")
        self._stack: ExitStack | None = None
        self._operators: dict[str, CachedOperator] = {}
        self.factorizations = 0
        self.source_solves = 0
        self.response_hits = 0

    def __enter__(self) -> ExactResponseCache:
        """Open the serial factor owner once."""
        if self._stack is not None:
            raise RuntimeError("response cache is already open")
        self._stack = ExitStack()
        return self

    def condense(self, problem: LocalProblem) -> LocalResponse:
        """Reuse an exact inverse contract and preserve the current problem/injection map."""
        if self._stack is None:
            raise RuntimeError("response cache must be open")
        key = f"{self.solver}:{self.precision}:{operator_identity(problem)}"
        source_key = array_identity(problem.load)
        cached = self._operators.get(key)
        if cached is None:
            matrix, rhs = problem.condensation_system()
            resources = ExitStack()
            try:
                factor = resources.enter_context(factorize(matrix, solver=self.solver))
                response = problem.response_from_solution(
                    factor.solve(rhs, refinement_precision=self.precision)
                )
            except BaseException:
                resources.close()
                raise
            cached = CachedOperator(
                factor,
                readonly(response.lifts),
                readonly(response.coarse_vectors),
                {source_key: readonly(response.source)},
                resources,
            )
            if len(self._operators) == self.max_operators:
                evicted = self._operators.pop(next(iter(self._operators)))
                evicted.resources.close()
            self._operators[key] = cached
            self.factorizations += 1
            self.source_solves += 1
        else:
            self._operators.pop(key)
            self._operators[key] = cached
            if not cached.factor.matches(problem.condensation_matrix()):
                raise ValueError("cached factor differs from the original augmented operator")
            if source_key not in cached.sources:
                source = problem.reconstruct(
                    np.zeros(problem.coupling.shape[1]),
                    np.zeros(problem.coarse_basis.shape[1]),
                    factorization=cached.factor,
                    refinement_precision=self.precision,
                )
                if len(cached.sources) == self.max_sources:
                    cached.sources.pop(next(iter(cached.sources)))
                cached.sources[source_key] = readonly(source)
                self.source_solves += 1
            else:
                source = cached.sources.pop(source_key)
                cached.sources[source_key] = source
                self.response_hits += 1
        return LocalResponse(
            problem, cached.sources[source_key], cached.lifts, cached.coarse_vectors
        )

    def close(self) -> None:
        """Release native factors and immutable template arrays explicitly."""
        if self._stack is not None:
            for cached in self._operators.values():
                cached.resources.close()
            self._stack.close()
            self._stack = None
        self._operators.clear()

    def __exit__(self, *exc: Any) -> None:
        """Close factors even after a failed original-equation check."""
        self.close()
