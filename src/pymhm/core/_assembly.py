"""Native sparse coordinates and ordered, race-free load accumulation."""

import numpy as np
from numba import njit

from pymhm.core.validation import FloatArray, IntArray


@njit(cache=True, nogil=True)
def contribution_coordinates(indices: IntArray) -> tuple[IntArray, IntArray]:
    """Expand distinct global indices in the block's original row-major order.

    The caller validates bounds and uniqueness. No sorting or duplicate
    reduction occurs here: the sparse backend combines shared macroface entries
    after the coordinator has consumed the complete ordered contribution stream.
    """
    count = len(indices)
    rows = np.empty(count * count, dtype=np.int64)
    columns = np.empty(count * count, dtype=np.int64)
    for i in range(count):
        for j in range(count):
            slot = i * count + j
            rows[slot] = indices[i]
            columns[slot] = indices[j]
    return rows, columns


@njit(cache=True, nogil=True)
def accumulate_load(
    indices: IntArray, local_rhs: FloatArray, rhs: FloatArray, load_scale: FloatArray
) -> None:
    """Add one validated binary64 load and its absolute scale in place.

    This serial kernel executes only in the ordered assembly coordinator.
    Releasing the GIL does not introduce concurrent writes to shared faces.
    Each coordinate receives the same additions, in the same order, as the
    original sparse reduction; cancellation and source scales stay distinct.
    """
    for i in range(len(indices)):
        slot = indices[i]
        rhs[slot] += local_rhs[i]
        load_scale[slot] += abs(local_rhs[i])
