"""Retain exact Helmholtz local responses with sparse boundary couplings.

This example adapter consumes the package's local assembler and condensation.
It changes storage and global COO construction, not quadrature, variational
operators, response coefficients or polynomial trace subspaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from examples.helmholtz_trace_family import solve_restricted_coordinates
from pymhm.helmholtz_forms import complex_vector
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
        boundary_action = self.coupling @ local_trace
        defect = self.matrix @ values + boundary_action - self.load
        scale = max(
            np.linalg.norm(abs(self.matrix) @ abs(values)),
            np.linalg.norm(boundary_action),
            np.linalg.norm(self.load),
            np.finfo(float).tiny,
        )
        residual = float(np.linalg.norm(defect) / scale)
        if residual > 1e-10:
            raise ValueError("compact reconstruction fails original local equations")
        return complex_vector(values), complex(np.sum(complex_vector(defect))), residual

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


@dataclass(frozen=True)
class CompactField:
    """Physical fields and original-equation diagnostics in one restricted space."""

    pressure: tuple[np.ndarray, ...]
    trace: np.ndarray
    residual: float
    local_residual_max: float
    balance: np.ndarray


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
    ) -> CompactFamily:
        """Assemble original local blocks and accumulate COO arrays in cell order."""
        local = tuple(
            map_local(
                CompactFactory(factory, local_solver, local_refinement_precision),
                range(len(factory.mesh.cells)),
                backend=backend,
                workers=workers,
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
        count = sum(len(item.dofs) ** 2 for item in local)
        rows = np.empty(count, dtype=np.int64)
        columns = np.empty_like(rows)
        entries = np.empty(count)
        rhs = np.zeros(skeleton.size)
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

    def solve(self, degree: int, *, solver: str = "scipy") -> CompactField:
        """Restrict the original Schur form and reuse unmodified source/trace lifts."""
        fixed = {dof: value for item in self.local for dof, value in item.fixed.items()}
        _, injection, _, trace, residual = solve_restricted_coordinates(
            self.skeleton, self.matrix, self.rhs, fixed, degree, solver=solver
        )
        lifted = np.asarray(injection @ trace).ravel()
        pressure, balance, local_residual = [], [], []
        for item in self.local:
            values, conservation, error = item.reconstruct(lifted)
            pressure.append(values)
            balance.append(conservation)
            local_residual.append(error)
        return CompactField(
            tuple(pressure),
            complex_vector(trace),
            residual,
            max(local_residual),
            np.asarray(balance),
        )
