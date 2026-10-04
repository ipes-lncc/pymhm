"""Retain exact Helmholtz local responses with sparse boundary couplings.

This example adapter consumes the package's local assembler and condensation.
It changes storage and global COO construction, not quadrature, variational
operators, response coefficients or polynomial trace subspaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from examples.helmholtz_trace_family import (
    solve_restricted_coordinates,
    verify_helmholtz_local_field,
    verify_helmholtz_trace_fields,
)
from examples.local_response_cache import ExactResponseCache
from pymhm.helmholtz_forms import complex_vector, real_vector
from pymhm.hybrid import LocalResponse
from pymhm.parallel import map_local


@dataclass(frozen=True)
class CompactLocal:
    """Original local equations and lifts, without duplicate dense test coupling."""

    matrix: Any
    coupling: Any
    load: np.ndarray
    dofs: np.ndarray
    source: np.ndarray
    lifts: np.ndarray
    schur: np.ndarray
    rhs: np.ndarray
    boundary: np.ndarray
    fixed: dict[int, float]

    @classmethod
    def from_response(cls, response: LocalResponse, metadata: tuple) -> CompactLocal:
        """Compress structural zeros only after the original Schur calculation."""
        problem = response.problem
        if problem.coarse_basis.shape[1]:
            raise ValueError("compact Helmholtz families require no retained modes")
        _, block, rhs = response.global_contribution(np.empty(0, dtype=np.int64))
        _, _, boundary, prescribed = metadata
        fixed = {}
        for scalar, value in prescribed.items():
            fixed[int(problem.trace_dofs[2 * scalar])] = float(value.real)
            fixed[int(problem.trace_dofs[2 * scalar + 1])] = float(value.imag)
        return cls(
            problem.matrix,
            sparse.csr_matrix(problem.coupling),
            problem.load,
            problem.trace_dofs,
            response.source,
            response.lifts,
            block,
            rhs,
            boundary,
            fixed,
        )

    def reconstruct(self, trace: np.ndarray) -> tuple[np.ndarray, complex, float]:
        """Recover coefficients and check the original local equations and balance."""
        local_trace = trace[self.dofs]
        values = self.source - self.lifts @ local_trace
        balance, residual = verify_helmholtz_local_field(
            self.matrix, self.coupling, self.load, values, local_trace
        )
        return complex_vector(values), balance, residual

    @property
    def storage_bytes(self) -> int:
        """Count retained coefficient buffers, excluding Python object overhead."""
        arrays = [
            self.load,
            self.dofs,
            self.source,
            self.lifts,
            self.schur,
            self.rhs,
            self.boundary,
        ]
        for matrix in (self.matrix, self.coupling):
            arrays.extend((matrix.data, matrix.indices, matrix.indptr))
        return sum(array.nbytes for array in arrays)


@dataclass(frozen=True)
class CompactFactory:
    """Spawn-safe local condensation with explicitly selected correction precision.

    The local solver is independent of the global trace solver. Extended
    precision delegates to LocalProblem and retains its source/trace responses
    without casting them before archival or reconstruction.
    """

    factory: Any
    local_solver: str = "scipy"
    local_refinement_precision: Literal["double", "extended"] = "double"

    def __call__(self, cell: int) -> CompactLocal:
        """Use the shared assembler/factorization and retain its exact response data."""
        assembled = self.factory(cell)
        return CompactLocal.from_response(
            assembled.problem.condense(
                self.local_solver, refinement_precision=self.local_refinement_precision
            ),
            assembled.metadata,
        )


def acquire_local_responses(
    factory: Any,
    cells: Iterable[int],
    *,
    backend: str = "serial",
    workers: int = 1,
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    cache: ExactResponseCache | None = None,
) -> list[CompactLocal]:
    """Assemble every original cell and optionally reuse identical inverse contracts.

    Cached execution is serial because the caller owns live native factors.
    Every cell is assembled and fingerprinted; each response retains its own
    global trace injection map. No geometric or coefficient grouping bypasses
    the original algebraic identity check.
    """
    if cache is None:
        return map_local(
            CompactFactory(factory, local_solver, local_refinement_precision),
            cells,
            backend=backend,
            workers=workers,
        )
    if backend != "serial" or workers != 1:
        raise ValueError("exact response caching requires one serial native factor owner")
    result = []
    for cell in cells:
        assembled = factory(cell)
        result.append(
            CompactLocal.from_response(cache.condense(assembled.problem), assembled.metadata)
        )
    return result


@dataclass(frozen=True)
class CompactField:
    """Physical fields and original-equation diagnostics in one restricted space."""

    pressure: tuple[np.ndarray, ...]
    trace: np.ndarray
    residual: float
    local_residual_max: float
    balance: np.ndarray
    original_trace_residual: float


@dataclass(frozen=True)
class CompactTrace:
    """Solved trace with its real coordinates in the prepared execution basis.

    ``trace`` uses complex coefficients in the requested polynomial subspace.
    ``prepared_coordinates`` retains interleaved real/imaginary coefficients
    after injection into the original response basis. Local reconstruction
    consumes those coordinates without loading or retaining every local field.
    """

    trace: np.ndarray
    prepared_coordinates: np.ndarray
    residual: float
    degree: int
    free_dofs: np.ndarray


@dataclass(frozen=True)
class CompactFamily:
    """A single prepared Schur system and shared responses for nested trace solves."""

    skeleton: Any
    local: Iterable[CompactLocal]
    matrix: Any
    rhs: np.ndarray

    @classmethod
    def prepare(
        cls,
        factory: Any,
        *,
        backend: str = "serial",
        workers: int = 1,
        local_solver: str = "scipy",
        local_refinement_precision: Literal["double", "extended"] = "double",
        exact_response_cache: bool = False,
    ) -> CompactFamily:
        """Assemble original local blocks and accumulate COO arrays in cell order."""
        if not isinstance(exact_response_cache, bool):
            raise ValueError("exact_response_cache must be a boolean")
        owner = (
            ExactResponseCache(solver=local_solver, refinement_precision=local_refinement_precision)
            if exact_response_cache
            else nullcontext(None)
        )
        with owner as cache:
            local = tuple(
                acquire_local_responses(
                    factory,
                    range(len(factory.mesh.cells)),
                    backend=backend,
                    workers=workers,
                    local_solver=local_solver,
                    local_refinement_precision=local_refinement_precision,
                    cache=cache,
                )
            )
        return cls.from_locals(factory.skeleton, local)

    @classmethod
    def from_locals(cls, skeleton: Any, local: Iterable[CompactLocal]) -> CompactFamily:
        """Assemble in cell order from a repeatable collection of local responses.

        The collection may stream stored batches, but must support independent
        iterations: assembly, boundary elimination and reconstruction each visit
        the same responses without retaining the collection in memory.
        """
        if iter(local) is local:
            raise ValueError("local responses must be repeatable, not a one-shot iterator")
        count = 0
        dtype = np.dtype(float)
        for item in local:
            count += len(item.dofs) ** 2
            dtype = np.result_type(dtype, item.schur.dtype, item.rhs.dtype, item.boundary.dtype)
        rows = np.empty(count, dtype=np.int64)
        columns = np.empty_like(rows)
        entries = np.empty(count, dtype=dtype)
        rhs = np.zeros(skeleton.size, dtype=dtype)
        start = 0
        for item in local:
            stop = start + len(item.dofs) ** 2
            rows[start:stop] = np.repeat(item.dofs, len(item.dofs))
            columns[start:stop] = np.tile(item.dofs, len(item.dofs))
            entries[start:stop] = item.schur.ravel()
            np.add.at(rhs, item.dofs, item.rhs)
            start = stop
        matrix = sparse.coo_matrix(
            (entries, (rows, columns)), shape=(skeleton.size, skeleton.size)
        ).tocsc()
        matrix.eliminate_zeros()
        for item in local:
            rhs[item.dofs] -= item.boundary
        return cls(skeleton, local, matrix, rhs)

    def solve_trace(self, degree: int, *, solver: str = "scipy") -> CompactTrace:
        """Solve only the original trace equations, retaining no reconstructed fields."""
        fixed = {dof: value for item in self.local for dof, value in item.fixed.items()}
        _, injection, free, trace, residual = solve_restricted_coordinates(
            self.skeleton, self.matrix, self.rhs, fixed, degree, solver=solver
        )
        lifted = np.asarray(injection @ trace).ravel()
        return CompactTrace(complex_vector(trace), lifted, residual, degree, free)

    def verify_fields(self, trace: CompactTrace, fields: Iterable[np.ndarray]) -> float:
        """Check original weak continuity from physical fields in the requested space.

        Each original equation is ``sum(B.T u)=g`` on free lower-degree trace
        coordinates. The prescribed Neumann coordinates are excluded, while
        the Dirichlet functional ``g`` is retained. No Schur blocks or response
        coefficients enter this check. The scale accumulates absolute physical
        coupling actions before cancellation. Both inputs stream in cell order.
        """
        return verify_helmholtz_trace_fields(
            self.skeleton,
            trace.degree,
            trace.free_dofs,
            (
                (item.coupling, item.dofs, item.boundary, real_vector(coefficients))
                for item, coefficients in zip(self.local, fields, strict=True)
            ),
            dtype=np.result_type(self.matrix.dtype, self.rhs.dtype),
        )

    def reconstruct(
        self, prepared_coordinates: np.ndarray
    ) -> Iterator[tuple[np.ndarray, complex, float]]:
        """Yield one physical field and its original-equation checks in cell order.

        Coordinates are real and interleaved in the prepared trace basis, rather
        than complex coefficients in a possibly smaller requested subspace.
        A repeatable disk store releases each local response after its yield.
        """
        trace = np.asarray(prepared_coordinates)
        if (
            np.iscomplexobj(trace)
            or trace.shape != (self.skeleton.size,)
            or not np.isfinite(trace).all()
        ):
            raise ValueError("prepared trace must contain finite real interleaved coordinates")
        for item in self.local:
            yield item.reconstruct(trace)

    def solve(self, degree: int, *, solver: str = "scipy") -> CompactField:
        """Restrict the original Schur form and reuse unmodified source/trace lifts."""
        trace = self.solve_trace(degree, solver=solver)
        pressure, balance, local_residual = [], [], []
        for values, conservation, error in self.reconstruct(trace.prepared_coordinates):
            pressure.append(values)
            balance.append(conservation)
            local_residual.append(error)
        return CompactField(
            tuple(pressure),
            trace.trace,
            trace.residual,
            max(local_residual),
            np.asarray(balance),
            self.verify_fields(trace, pressure),
        )
