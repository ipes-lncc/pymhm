"""Ordered sparse reduction of local blocks with shared trace indices."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core._assembly import accumulate_load, contribution_coordinates
from pymhm.core.contracts import LocalResponse, _array, _preserved_array
from pymhm.core.validation import FloatArray, IntArray


def local_global_contribution(
    response: LocalResponse,
    coarse_dofs: Any,
    *,
    direct_matrix: Any = None,
    direct_load: Any = None,
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Return indices, matrix and RHS of this cell's reduced Petrov equations.

    This is the common assembly contract for serial, MPI and accelerator
    backends. The trace rows enforce C.T u=g; retained test rows enforce
    W.T (A u+B lambda-f)=0. The negative signs preserve the symmetric
    saddle convention whenever C=B and test/trial data coincide.
    ``direct_matrix`` and ``direct_load`` add explicitly declared local D/g
    terms to the trace rows, in the same local trace ordering. Their defaults
    are zero. Wider real floating digits are preserved. The same operation
    therefore owns four-block variational forms in serial, CPU workers and MPI.
    """
    p = response.problem
    coarse_dofs = np.asarray(coarse_dofs)
    if (
        coarse_dofs.shape != (p.coarse_basis.shape[1],)
        or not np.issubdtype(coarse_dofs.dtype, np.integer)
        or np.any(coarse_dofs < 0)
    ):
        raise ValueError("coarse_dofs must contain one nonnegative integer per retained mode")
    g = p.test_coupling.T @ response.retained_basis
    if response.coarse_vectors is None:
        coarse_trace = p.left_kernel.T @ p.coupling
        coarse_matrix = np.zeros((len(coarse_dofs), len(coarse_dofs)))
    else:
        coarse_trace = p.test_basis.T @ p.coupling - p._test_action.T @ response.lifts
        coarse_matrix = p._test_action.T @ response.retained_basis
    block = np.block([[p.test_coupling.T @ response.lifts, -g], [-coarse_trace, -coarse_matrix]])
    load = response.global_load()
    count = len(p.trace_dofs)
    direct = (
        None
        if direct_matrix is None
        else _preserved_array(direct_matrix, (count, count), "direct_matrix")
    )
    forcing = (
        None if direct_load is None else _preserved_array(direct_load, (count,), "direct_load")
    )
    if direct is not None or forcing is not None:
        dtype = np.result_type(
            block,
            load,
            *([] if direct is None else [direct]),
            *([] if forcing is None else [forcing]),
        )
        block, load = block.astype(dtype, copy=False), load.astype(dtype, copy=False)
    if direct is not None:
        block[:count, :count] += direct
    if forcing is not None:
        load[:count] += forcing
    return (
        np.r_[p.trace_dofs, coarse_dofs].astype(np.int64),
        block,
        load,
    )


def assemble_hybrid_contributions(
    contributions: Iterable[tuple[IntArray, FloatArray, FloatArray]],
    *,
    trace_size: int,
    kernel_offsets: IntArray,
    boundary_load: Any = None,
    require_local_trace_coverage: bool = True,
) -> tuple[Any, FloatArray, FloatArray]:
    """Reduce ordered cell contributions to a CSC operator, RHS and load scale.

    ``kernel_offsets`` starts at ``trace_size`` and bounds each cell's retained
    global coordinates. Exactly one contribution per interval is consumed in
    iterable order, without retaining a tuple of contributions. Shared trace
    entries are added in that order; wider real matrix and load digits survive.
    The COO matrix entries remain stored until the final CSC conversion.
    ``load_scale`` sums absolute loads before cancellation. The weak Dirichlet
    ``boundary_load`` is subtracted once from the trace RHS and its absolute
    value added to that scale. No trace orientation, gauge or local basis is
    inferred. The caller supplies the executed, consistently numbered blocks.
    ``require_local_trace_coverage=False`` permits coordinates belonging only
    to an explicitly assembled global form; retained cell intervals are always
    checked. The complete global solver still rejects a singular operator.
    A single kernel offset and no contributions represent a global-only form;
    uncovered trace coordinates require require_local_trace_coverage=False.
    """
    if (
        isinstance(trace_size, (bool, np.bool_))
        or not isinstance(trace_size, (int, np.integer))
        or trace_size < 0
    ):
        raise ValueError("trace_size must be a nonnegative integer")
    kernel_offsets = np.asarray(kernel_offsets)
    if (
        kernel_offsets.ndim != 1
        or len(kernel_offsets) < (2 if require_local_trace_coverage else 1)
        or not np.issubdtype(kernel_offsets.dtype, np.integer)
        or kernel_offsets[0] != trace_size
        or np.any(kernel_offsets[1:] < kernel_offsets[:-1])
    ):
        raise ValueError(
            "kernel_offsets must be an ordered integer partition starting at trace_size"
        )
    size = int(kernel_offsets[-1])
    rhs = np.zeros(size)
    load_scale = np.zeros(size)
    rows: list[IntArray] = []
    columns: list[IntArray] = []
    entries: list[FloatArray] = []
    all_indices = []
    count = 0
    for cell, (indices, block, local_rhs) in enumerate(contributions):
        if cell >= len(kernel_offsets) - 1:
            raise ValueError("one contribution per coarse cell partition is required")
        count += 1
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= size)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("contribution indices must be distinct valid global integers")
        if not np.array_equal(
            np.sort(indices[indices >= trace_size]),
            np.arange(kernel_offsets[cell], kernel_offsets[cell + 1]),
        ):
            raise ValueError("contribution coarse indices must match its ordered cell partition")
        # Native kernels consume native-endian indices; persisted arrays can
        # carry another byte order or an unsigned integer representation.
        indices = np.asarray(indices, dtype=np.int64)
        checked_block = _array(block, (len(indices), len(indices)), "contribution matrix")
        checked_rhs = _array(local_rhs, (len(indices),), "contribution rhs")
        # Real floating inputs may contain explicitly retained correction digits.
        # Validation remains identical for lists and other convertible inputs.
        block, local_rhs = np.asarray(block), np.asarray(local_rhs)
        block = block if block.dtype.kind == "f" else checked_block
        local_rhs = local_rhs if local_rhs.dtype.kind == "f" else checked_rhs
        dtype = np.result_type(rhs.dtype, local_rhs.dtype)
        rhs, load_scale = rhs.astype(dtype, copy=False), load_scale.astype(dtype, copy=False)
        all_indices.append(indices.copy())
        row, column = contribution_coordinates(indices)
        rows.append(row)
        columns.append(column)
        # Providers may reuse a scratch block after yielding their contribution.
        entries.append(block.ravel().copy())
        if rhs.dtype == np.dtype(float):
            accumulate_load(indices, checked_rhs, rhs, load_scale)
        else:
            # Explicit wider inputs preserve their data contract outside JIT.
            np.add.at(rhs, indices, local_rhs)
            np.add.at(load_scale, indices, np.abs(local_rhs))
    if count != len(kernel_offsets) - 1:
        raise ValueError("one contribution per coarse cell partition is required")
    if require_local_trace_coverage and not np.array_equal(
        np.unique(np.concatenate(all_indices)), np.arange(size)
    ):
        raise ValueError("contributions must cover the contiguous global numbering")
    matrix = sparse.coo_matrix(
        (
            np.concatenate(entries) if entries else np.empty(0),
            (
                np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
                np.concatenate(columns) if columns else np.empty(0, dtype=np.int64),
            ),
        ),
        shape=(size, size),
    ).tocsc()
    matrix.eliminate_zeros()
    if boundary_load is not None:
        boundary = _array(boundary_load, (trace_size,), "boundary_load")
        raw_boundary = np.asarray(boundary_load)
        if raw_boundary.dtype.kind == "f":
            boundary = raw_boundary
        dtype = np.result_type(rhs.dtype, boundary.dtype)
        rhs, load_scale = rhs.astype(dtype, copy=False), load_scale.astype(dtype, copy=False)
        rhs[:trace_size] -= boundary
        load_scale[:trace_size] += np.abs(boundary)
    return matrix, rhs, load_scale
