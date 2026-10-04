"""Distributed MHM assembly and direct solution on an MPI communicator.

Each rank owns its input cells, local matrices, responses and reconstructed
fields. PETSc distributes the global matrix and vector by rows; off-rank
contributions are assembled collectively. No rank gathers the global matrix or
all local responses. MUMPS performs distributed pivoted LU of the saddle system.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import LocalAssembly, LocalProblem, LocalResponse
from pymhm.core.validation import FloatArray, IntArray
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError, _optional, _tolerances


@dataclass(frozen=True)
class DistributedHybridSolution:
    """Owned global coefficients and fields of the cells assigned to this rank.

    ``ownership`` is PETSc's half-open row interval, including retained modes and
    any gauge multipliers. ``local_trace`` contains exactly the skeleton values
    needed by local cells. The residual uses the collective original physical
    equations after excluding prescribed rows; gauges cannot repair incompatible
    physical data beyond the componentwise evaluation bound for a declared
    kernel. General retained modes receive no such allowance. ``linear_residual``
    also checks the augmented boundary system.
    """

    ownership: tuple[int, int]
    owned_coefficients: FloatArray
    local_trace_indices: IntArray
    local_trace: FloatArray
    coarse: tuple[FloatArray, ...]
    fields: tuple[FloatArray, ...]
    local_metadata: tuple[Any, ...]
    residual: float
    linear_residual: float
    global_size: int
    raw_residual: float | None = None
    raw_residual_norm: float | None = None


def _collective_error(comm: Any, error: Exception | None) -> None:
    """Raise on every rank before entering native collectives after a local failure."""
    failures = comm.allgather(None if error is None else f"{type(error).__name__}: {error}")
    if any(value is not None for value in failures):
        details = "; ".join(f"rank {rank}: {value}" for rank, value in enumerate(failures) if value)
        raise ValueError(f"distributed preparation failed: {details}")


def _indices(indices: Any, size: int, name: str) -> IntArray:
    """Validate potentially repeated global indices of additive contributions."""
    raw = np.asarray(indices)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError(f"{name} must be integer indices")
    values = raw.astype(np.int64)
    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(f"{name} index outside global system")
    return values


def _values(values: Any, size: int, name: str) -> FloatArray:
    """Require a real finite vector for a local physical contribution."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or raw.shape != (size,) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a real finite vector of length {size}")
    return np.asarray(raw, dtype=float)


def _extract(vector: Any, indices: IntArray, petsc: Any, resources: ExitStack) -> FloatArray:
    """Communicate only requested global coefficients to a rank-local sequential vector."""
    source = petsc.IS().createGeneral(indices.astype(petsc.IntType), comm=petsc.COMM_SELF)
    resources.callback(source.destroy)
    target = petsc.Vec().createSeq(len(indices), comm=petsc.COMM_SELF)
    resources.callback(target.destroy)
    destination = petsc.IS().createStride(len(indices), step=1, comm=petsc.COMM_SELF)
    resources.callback(destination.destroy)
    scatter = petsc.Scatter().create(vector, source, target, destination)
    resources.callback(scatter.destroy)
    scatter.scatter(vector, target, addv=petsc.InsertMode.INSERT_VALUES)
    return np.array(target.getArray(readonly=True), dtype=float, copy=True)


def solve_distributed(
    factory: Callable[[Any], LocalProblem | LocalAssembly],
    local_items: Iterable[Any],
    *,
    trace_size: int,
    comm: Any,
    boundary_load: tuple[Any, Any] | None = None,
    fixed: dict[int, float] | None = None,
    moments: Sequence[tuple[Sequence[FloatArray], float]] | None = None,
    local_solver: str = "scipy",
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> DistributedHybridSolution:
    """Assemble and solve MHM collectively from nonreplicated rank-owned cells.

    ``comm`` is an mpi4py communicator. Invoke this function on every rank,
    including ranks with no cells. ``trace_size`` and ``fixed`` are replicated
    small metadata and must agree between ranks. Skeleton numbering must cover
    ``range(trace_size)``. Retained modes are numbered after traces, first by
    rank, then by local input order. A distributed mesh partitioner is not
    inferred: the application assigns each physical cell to exactly one rank.

    ``boundary_load=(indices, values)`` is an additive *owned* boundary-moment
    contribution, not a replicated full load. Contributions are subtracted as in
    ``HybridSystem``. Every physical moment supplies this rank's weight vectors
    and the same global target on all ranks. Fixed values are trace coefficients;
    they are eliminated through PETSc row/column elimination with RHS lifting.

    PETSc/MUMPS is mandatory; there is no sequential or iterative fallback. The
    numerical-library factorization detects singular pivots and the original
    equations are independently checked. This does not prove an inf-sup bound.
    """
    petsc = _optional("petsc4py.PETSc", "Install MPI-enabled petsc4py with MUMPS.")
    if not petsc.Sys.hasExternalPackage("mumps"):
        raise SolverUnavailableError("distributed MHM requires PETSc with MUMPS")
    responses: list[LocalResponse] = []
    metadata: list[Any] = []
    error = None
    fixed = {} if fixed is None else dict(fixed)
    moments = tuple(() if moments is None else moments)
    boundary_indices, boundary_values = np.empty(0, dtype=np.int64), np.empty(0)
    try:
        _tolerances(rtol, atol)
        if isinstance(trace_size, bool) or not isinstance(trace_size, int) or trace_size < 0:
            raise ValueError("trace_size must be a nonnegative integer")
        _indices(list(fixed), trace_size, "fixed")
        _values(list(fixed.values()), len(fixed), "fixed values")
        for item in local_items:
            built = factory(item)
            problem = built.problem if isinstance(built, LocalAssembly) else built
            if not isinstance(problem, LocalProblem):
                raise TypeError("factory must return LocalProblem or LocalAssembly")
            _indices(problem.trace_dofs, trace_size, "trace")
            responses.append(problem.condense(local_solver))
            metadata.append(built.metadata if isinstance(built, LocalAssembly) else None)
        if boundary_load is not None:
            boundary_indices = _indices(boundary_load[0], trace_size, "boundary")
            boundary_values = _values(boundary_load[1], len(boundary_indices), "boundary values")
        for weights, target in moments:
            if len(weights) != len(responses) or not np.isfinite(target):
                raise ValueError("moments require one weight per owned cell and finite targets")
            for weight, response in zip(weights, responses, strict=True):
                _values(weight, len(response.source), "moment weights")
    except Exception as exc:
        error = exc
    _collective_error(comm, error)
    configuration = (
        trace_size,
        sorted(fixed.items()),
        [target for _, target in moments],
        rtol,
        atol,
    )
    if any(value != configuration for value in comm.allgather(configuration)):
        raise ValueError(
            "trace size, fixed values, moment targets and tolerances must agree across ranks"
        )
    counts = comm.allgather(sum(r.problem.coarse_basis.shape[1] for r in responses))
    cell_count = comm.allreduce(len(responses))
    if cell_count == 0:
        raise ValueError("at least one global local problem is required")
    coarse_start = trace_size + sum(counts[: comm.rank])
    offsets = (
        coarse_start + np.r_[0, np.cumsum([r.problem.coarse_basis.shape[1] for r in responses])]
    )
    physical_size = trace_size + sum(counts)
    size = physical_size + len(moments)
    if not size:
        raise ValueError("distributed global system must contain at least one unknown")
    with ExitStack() as resources:
        matrix = petsc.Mat().createAIJ(size=(size, size), nnz=16, comm=comm)
        resources.callback(matrix.destroy)
        matrix.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        rhs = matrix.createVecLeft()
        resources.callback(rhs.destroy)
        coverage = rhs.duplicate()
        resources.callback(coverage.destroy)
        load_scale = rhs.duplicate()
        resources.callback(load_scale.destroy)
        for index, response in enumerate(responses):
            dofs, block, load = response.global_contribution(
                np.arange(offsets[index], offsets[index + 1])
            )
            dofs = dofs.astype(petsc.IntType)
            matrix.setValues(dofs, dofs, block, addv=petsc.InsertMode.ADD_VALUES)
            rhs.setValues(dofs, load, addv=petsc.InsertMode.ADD_VALUES)
            load_scale.setValues(dofs, np.abs(load), addv=petsc.InsertMode.ADD_VALUES)
            coverage.setValues(dofs, np.ones(len(dofs)), addv=petsc.InsertMode.ADD_VALUES)
        rhs.setValues(
            boundary_indices.astype(petsc.IntType),
            -boundary_values,
            addv=petsc.InsertMode.ADD_VALUES,
        )
        load_scale.setValues(
            boundary_indices.astype(petsc.IntType),
            np.abs(boundary_values),
            addv=petsc.InsertMode.ADD_VALUES,
        )
        matrix.assemble()
        rhs.assemble()
        coverage.assemble()
        load_scale.assemble()
        start, stop = rhs.getOwnershipRange()
        owned = np.arange(start, stop)
        missing = np.any(coverage.getArray(readonly=True)[owned < physical_size] == 0)
        if comm.allreduce(int(missing)):
            raise ValueError("global trace numbering contains unowned indices")
        # Keep physical equations separately, before adding gauge multipliers.
        physical_matrix, physical_rhs = matrix.copy(), rhs.copy()
        resources.callback(physical_matrix.destroy)
        resources.callback(physical_rhs.destroy)
        for index, (weights, target) in enumerate(moments):
            gauge = physical_size + index
            source_integral = 0.0
            for cell, (weight, response) in enumerate(zip(weights, responses, strict=True)):
                indices = np.r_[
                    response.problem.trace_dofs, np.arange(offsets[cell], offsets[cell + 1])
                ].astype(petsc.IntType)
                row = np.r_[-response.lifts.T @ weight, response.retained_basis.T @ weight]
                matrix.setValues([gauge], indices, row[None, :], addv=petsc.InsertMode.ADD_VALUES)
                matrix.setValues(indices, [gauge], row[:, None], addv=petsc.InsertMode.ADD_VALUES)
                source_integral += float(weight @ response.source)
            rhs.setValue(
                gauge,
                (target if comm.rank == 0 else 0.0) - source_integral,
                addv=petsc.InsertMode.ADD_VALUES,
            )
        matrix.assemble()
        rhs.assemble()
        imposed = rhs.duplicate()
        resources.callback(imposed.destroy)
        fixed_owned = np.array(
            [index for index in fixed if start <= index < stop], dtype=petsc.IntType
        )
        imposed.setValues(fixed_owned, [fixed[int(index)] for index in fixed_owned])
        imposed.assemble()
        matrix.zeroRowsColumns(fixed_owned, diag=1.0, x=imposed, b=rhs)
        ksp = petsc.KSP().create(comm=comm)
        resources.callback(ksp.destroy)
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setErrorIfNotConverged(True)
        solution = rhs.duplicate()
        resources.callback(solution.destroy)
        try:
            ksp.solve(rhs, solution)
        except petsc.Error as exc:
            raise LinearSolveError(f"distributed MUMPS factorization/solve failed: {exc}") from exc
        difference = rhs.duplicate()
        resources.callback(difference.destroy)
        matrix.mult(solution, difference)
        difference.axpy(-1.0, rhs)
        absolute_residual, rhs_norm = difference.norm(), rhs.norm()
        if not np.isfinite(absolute_residual) or absolute_residual > max(atol, rtol * rhs_norm):
            raise LinearSolveError("distributed augmented residual criterion failed")
        excluded = np.isin(owned, list(fixed)) | (owned >= physical_size)
        load_scale.getArray()[excluded] = 0
        # Remove prescribed columns before evaluating the free physical equations.
        physical_rhs.getArray()[excluded] = 0
        original_rhs_norm = physical_rhs.norm()
        physical_matrix.mult(imposed, difference)
        difference.getArray()[excluded] = 0
        prescribed_action_norm = difference.norm()
        physical_rhs.axpy(-1.0, difference)
        free_solution = solution.copy()
        resources.callback(free_solution.destroy)
        free_solution.getArray()[excluded] = 0
        physical_matrix.mult(free_solution, difference)
        difference.axpy(-1.0, physical_rhs)
        difference.getArray()[excluded] = 0
        physical_rhs.getArray()[excluded] = 0
        absolute_matrix = physical_matrix.copy()
        resources.callback(absolute_matrix.destroy)
        indptr, indices, entries = absolute_matrix.getValuesCSR()
        absolute_matrix.setValuesCSR(indptr, indices, np.abs(entries))
        absolute_matrix.assemble()
        absolute_prescribed = imposed.copy()
        resources.callback(absolute_prescribed.destroy)
        absolute_prescribed.getArray()[:] = np.abs(absolute_prescribed.getArray())
        prescribed_roundoff = rhs.duplicate()
        resources.callback(prescribed_roundoff.destroy)
        absolute_matrix.mult(absolute_prescribed, prescribed_roundoff)
        fixed_entries = np.isin(indices, list(fixed))
        row_indices = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        terms = np.bincount(row_indices, weights=fixed_entries, minlength=len(indptr) - 1)
        rounding_units = (terms + 1) * np.finfo(float).eps
        prescribed_roundoff.getArray()[:] *= rounding_units / (1 - rounding_units)
        prescribed_roundoff.getArray()[excluded] = 0
        free_solution.getArray()[:] = np.abs(free_solution.getArray())
        absolute_matrix.mult(free_solution, imposed)
        imposed.getArray()[excluded] = 0
        raw_residual_norm = difference.norm()
        physical_scale = max(
            imposed.norm(),
            physical_rhs.norm(),
            original_rhs_norm,
            prescribed_action_norm,
            load_scale.norm(),
            prescribed_roundoff.norm() / 1e-8,
            np.finfo(float).tiny,
        )
        raw_residual = raw_residual_norm / physical_scale
        residual = raw_residual
        if residual > 1e-8 and moments:
            kernel_indices = (
                np.unique(np.concatenate([r.problem.trace_dofs for r in responses]))
                if responses
                else np.empty(0, dtype=np.int64)
            )
            requested_kernel = np.r_[kernel_indices, np.arange(coarse_start, offsets[-1])]
            kernel_coefficients = _extract(solution, requested_kernel, petsc, resources)
            kernel_allowance = rhs.duplicate()
            resources.callback(kernel_allowance.destroy)
            kernel_allowance.getArray()[:] = 0
            for cell, response in enumerate(responses):
                first = len(kernel_indices) + offsets[cell] - coarse_start
                last = len(kernel_indices) + offsets[cell + 1] - coarse_start
                kernel_field = response.reconstruct(
                    kernel_coefficients[
                        np.searchsorted(kernel_indices, response.problem.trace_dofs)
                    ],
                    kernel_coefficients[first:last],
                )
                bound = response.kernel_roundoff_bound(kernel_field)
                kernel_allowance.setValues(
                    np.arange(offsets[cell], offsets[cell + 1], dtype=petsc.IntType),
                    bound,
                    addv=petsc.InsertMode.ADD_VALUES,
                )
            kernel_allowance.assemble()
            certified = rhs.duplicate()
            resources.callback(certified.destroy)
            certified.getArray()[:] = np.maximum(
                np.abs(difference.getArray(readonly=True))
                - kernel_allowance.getArray(readonly=True),
                0,
            )
            residual = certified.norm() / physical_scale
        if residual > 1e-8:
            raise ValueError("incompatible data: distributed gauge changed physical equations")
        trace_indices = (
            np.unique(np.concatenate([r.problem.trace_dofs for r in responses]))
            if responses
            else np.empty(0, dtype=np.int64)
        )
        requested = np.r_[trace_indices, np.arange(coarse_start, offsets[-1])].astype(np.int64)
        coefficients = _extract(solution, requested, petsc, resources)
        traces = coefficients[: len(trace_indices)]
        coarse = tuple(
            coefficients[
                len(trace_indices) + offsets[i] - coarse_start : len(trace_indices)
                + offsets[i + 1]
                - coarse_start
            ].copy()
            for i in range(len(responses))
        )
        fields = tuple(
            response.reconstruct(
                traces[np.searchsorted(trace_indices, response.problem.trace_dofs)], mode
            )
            for response, mode in zip(responses, coarse, strict=True)
        )
        return DistributedHybridSolution(
            (start, stop),
            np.array(solution.getArray(readonly=True), copy=True),
            trace_indices,
            traces,
            coarse,
            fields,
            tuple(metadata),
            residual,
            absolute_residual / max(rhs_norm, np.finfo(float).tiny),
            size,
            raw_residual,
            raw_residual_norm,
        )
