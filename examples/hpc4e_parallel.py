"""Distributed native algebra for the original HPC4E reference example.

Only rows and finite-element cells are partitioned. The operator, five Ruiz
congruences and original-equation tolerance are independent of the MPI layout.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np


def check_symmetry(
    matrix: Any, pointer: np.ndarray, indices: np.ndarray, values: np.ndarray
) -> None:
    """Check all coefficients without extracting the complement of each MPI partition.

    Diagonal ownership blocks are checked locally. Only nonzero coefficients
    coupling different ranks are gathered for their exact transpose comparison.
    The bound is the same global maximum-entry relative criterion as the serial
    factorizer, including any coefficient whose transpose is structurally zero.
    """
    from mpi4py import MPI
    from scipy import sparse

    comm = matrix.comm.tompi4py()
    scale = comm.allreduce(float(np.max(abs(values), initial=0)), op=MPI.MAX)
    diagonal = matrix.getDiagonalBlock()
    try:
        local_symmetric = diagonal.isSymmetric(tol=1e-13 * scale)
    finally:
        diagonal.destroy()
    if not comm.allreduce(local_symmetric, op=MPI.LAND):
        raise ValueError("the original UFL matrix must be symmetric")
    begin, end = matrix.getOwnershipRange()
    selected = np.flatnonzero(((indices < begin) | (indices >= end)) & (values != 0))
    rows = np.searchsorted(pointer, selected, side="right") - 1 + begin
    pieces = comm.gather((rows, indices[selected], values[selected]), root=0)
    symmetric = None
    if comm.rank == 0:
        row, column, value = (np.concatenate([piece[i] for piece in pieces]) for i in range(3))
        offdiagonal = sparse.coo_matrix((value, (row, column)), shape=matrix.getSize()).tocsr()
        difference = offdiagonal - offdiagonal.T
        symmetric = bool(np.max(abs(difference.data), initial=0) <= 1e-13 * scale)
    if not comm.bcast(symmetric, root=0):
        raise ValueError("the original UFL matrix must be symmetric")


def compact_matrix(matrix: Any, resources: ExitStack) -> tuple[Any, dict[str, int]]:
    """Discard exactly zero stored entries without changing any matrix coefficient."""
    from mpi4py import MPI
    from petsc4py import PETSc
    from scipy import sparse

    comm = matrix.comm.tompi4py()
    pointer, indices, values = matrix.getValuesCSR()
    check_symmetry(matrix, pointer, indices, values)
    before = int(comm.allreduce(len(values), op=MPI.SUM))
    local = sparse.csr_matrix(
        (values, indices, pointer), shape=(matrix.getLocalSize()[0], matrix.getSize()[1])
    )
    local.eliminate_zeros()
    # Indefinite pressure/rotation rows need explicit structural zero diagonals.
    offset = matrix.getOwnershipRange()[0]
    local.setdiag(local.diagonal(k=offset), k=offset)
    local.sort_indices()
    after = int(comm.allreduce(local.nnz, op=MPI.SUM))
    result = PETSc.Mat().createAIJ(
        size=tuple(zip(matrix.getLocalSize(), matrix.getSize(), strict=True)),
        csr=(local.indptr, local.indices, local.data),
        comm=matrix.comm,
    )
    resources.callback(result.destroy)
    result.assemble()
    # Removing numerical zeros can make the stored graph asymmetric, although
    # the coefficient-wise symmetry check above is unchanged by their removal.
    result.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    matrix.destroy()
    return result, {"structural_nonzeros": before, "nonzero_entries": after}


def symmetric_equilibration(matrix: Any, resources: ExitStack) -> np.ndarray:
    """Apply five infinity-row Ruiz congruences in place and return the owned diagonal."""
    from mpi4py import MPI

    comm = matrix.comm.tompi4py()
    if matrix.isSymmetricKnown() != (True, True):
        raise ValueError("equilibration requires the verified symmetric UFL operator")
    vector = matrix.createVecLeft()
    resources.callback(vector.destroy)
    diagonal = np.ones(vector.getLocalSize())
    for _ in range(5):
        pointer, _, values = matrix.getValuesCSR()
        empty = np.any(np.diff(pointer) == 0)
        if comm.allreduce(bool(empty), op=MPI.LOR):
            raise ValueError("symmetric equilibration failed: matrix has a zero row")
        norms = np.maximum.reduceat(abs(values), pointer[:-1])
        if comm.allreduce(bool(np.any(norms == 0)), op=MPI.LOR):
            raise ValueError("symmetric equilibration failed: matrix has a zero row")
        vector.array[:] = 1.0 / np.sqrt(norms)
        diagonal *= vector.array
        del pointer, values, norms
        matrix.diagonalScale(vector, vector)
    from petsc4py import PETSc

    # Applying the same positive diagonal on both sides preserves symmetry.
    matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    return diagonal


def original_residual(matrix: Any, forcing: np.ndarray, solution: np.ndarray) -> np.ndarray:
    """Evaluate owned rows of b-Ax with extended products and MPI-gathered coordinates.

    Only the solution is replicated. Sparse rows remain distributed, and their
    extended-precision products are evaluated in bounded batches of 8192 rows.
    """
    from mpi4py import MPI
    from scipy import sparse

    comm = matrix.comm.tompi4py()
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise RuntimeError("precise distributed residual requires wider long double")
    counts = np.asarray(comm.allgather(len(solution)), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts[:-1])))
    global_state = np.empty(int(counts.sum()), dtype=np.longdouble)
    comm.Allgatherv(
        np.asarray(solution, dtype=np.longdouble), [global_state, counts, offsets, MPI.LONG_DOUBLE]
    )
    result = np.asarray(forcing, dtype=np.longdouble).copy()
    # PETSc returns a CSR copy. Stagger that temporary allocation by rank so
    # precision checks do not replicate another full matrix while factors live.
    for owner in range(comm.size):
        if comm.rank == owner:
            pointer, indices, values = matrix.getValuesCSR()
            for begin in range(0, len(forcing), 8192):
                end = min(begin + 8192, len(forcing))
                first, last = pointer[begin], pointer[end]
                block = sparse.csr_matrix(
                    (
                        values[first:last].astype(np.longdouble),
                        indices[first:last],
                        pointer[begin : end + 1] - first,
                    ),
                    shape=(end - begin, len(global_state)),
                )
                result[begin:end] -= block @ global_state
            del pointer, indices, values, block
        comm.barrier()
    return result


def checked_solve(
    ksp: Any,
    original: Any,
    rhs: Any,
    forcing: Any,
    prescribed: np.ndarray,
    boundary: Any,
    diagonal: np.ndarray,
    resources: ExitStack,
    *,
    refinement_precision: str = "double",
    rtol: float = 1e-11,
) -> tuple[Any, list[float]]:
    """Reuse native MUMPS factors for at most two original-equation defect corrections.

    The denominator is the Euclidean norm of the unscaled boundary-eliminated
    right-hand side, exactly as for the serial direct factorization. Physical
    free rows use the original UFL operator; prescribed rows use x=g. An
    explicitly extended accumulator still undergoes a float64 physical check
    in the calling driver before any field is accepted.
    """
    from mpi4py import MPI

    comm = original.comm.tompi4py()
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise RuntimeError("extended refinement requires wider long double")
    vectors = [original.createVecRight() for _ in range(3)]
    for vector in vectors:
        resources.callback(vector.destroy)
    scaled_rhs, state, output = vectors

    def solve_local(load: np.ndarray) -> np.ndarray:
        """Transform owned loads and coordinates through the declared congruence."""
        scaled_rhs.array[:] = diagonal * load
        ksp.solve(scaled_rhs, state)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"MUMPS failed with reason {ksp.getConvergedReason()}")
        return diagonal * state.array.copy()

    dtype = np.longdouble if refinement_precision == "extended" else np.float64
    solution = np.asarray(solve_local(rhs.array), dtype=dtype)
    denominator = rhs.norm()
    history = []
    for iteration in range(3):
        # Known Dirichlet coordinates belong to the affine trial space exactly.
        # This also makes defect corrections consistent with zeroRowsColumns.
        solution[prescribed] = boundary.array[prescribed]
        defect = original_residual(original, forcing.array[: len(solution)], solution)
        defect[prescribed] = boundary.array[prescribed] - solution[prescribed]
        squared = comm.allreduce(np.sum(defect * defect, dtype=np.longdouble), op=MPI.SUM)
        absolute = float(np.sqrt(squared))
        relative = absolute / denominator if denominator else absolute
        history.append(relative)
        if absolute <= rtol * denominator:
            output.array[:] = solution
            return output, history
        if iteration < 2:
            solution += solve_local(np.asarray(defect, dtype=float))
    raise ValueError(f"original distributed residual criterion failed: {history}, allowed={rtol}")
