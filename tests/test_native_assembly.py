"""Independent sparse scatter checks for shared faces and streaming providers."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pymhm.core._assembly import accumulate_load, contribution_coordinates
from pymhm.core.contributions import assemble_hybrid_contributions


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("indices", [[], [3], [4, 0, 2]])
def test_native_scatter_preserves_declared_slots_and_cancellation(interpreted, indices):
    """Native and Python executions match independently indexed face loads."""
    indices = np.asarray(indices, dtype=np.int64)
    coordinates = contribution_coordinates.py_func if interpreted else contribution_coordinates
    scatter = accumulate_load.py_func if interpreted else accumulate_load
    rows, columns = coordinates(indices)
    expected_pairs = [(int(i), int(j)) for i in indices for j in indices]
    assert_array_equal(rows, [i for i, _ in expected_pairs])
    assert_array_equal(columns, [j for _, j in expected_pairs])
    rhs = np.full(5, 2.0)
    scale = np.full(5, 3.0)
    loads = np.arange(len(indices), dtype=float) - 1.0
    expected_rhs, expected_scale = rhs.copy(), scale.copy()
    for index, value in zip(indices, loads, strict=True):
        expected_rhs[index] += value
        expected_scale[index] += abs(value)
    scatter(indices, loads, rhs, scale)
    assert_array_equal(rhs, expected_rhs)
    assert_array_equal(scale, expected_scale)


def test_streamed_scratch_blocks_and_shared_face_loads_are_captured_once():
    """Reused provider buffers cannot overwrite previously consumed operators."""
    scratch = np.empty((2, 2))
    loads = np.empty(2)
    indices = np.empty(2, dtype=np.int64)

    def contributions():
        """Yield two cells sharing trace 1 and reuse both numerical buffers."""
        scratch[:] = [[2.0, -1.0], [-1.0, 2.0]]
        loads[:] = [1.0, -2.0]
        indices[:] = [0, 1]
        yield indices, scratch, loads
        scratch[:] = [[3.0, -0.5], [-0.5, 3.0]]
        loads[:] = [2.0, 4.0]
        indices[:] = [1, 2]
        yield indices, scratch, loads

    matrix, rhs, scale = assemble_hybrid_contributions(
        contributions(), trace_size=3, kernel_offsets=np.array([3, 3, 3])
    )
    assert_array_equal(matrix.toarray(), [[2, -1, 0], [-1, 5, -0.5], [0, -0.5, 3]])
    assert_array_equal(rhs, [1, 0, 4])
    assert_array_equal(scale, [1, 4, 4])


@pytest.mark.parametrize("dtype", [np.int32, np.uint64, ">i8"])
def test_native_assembly_accepts_valid_persisted_integer_representations(dtype):
    """Archive byte order and integer width cannot change the declared numbering."""
    matrix, rhs, scale = assemble_hybrid_contributions(
        [(np.array([1, 0], dtype=dtype), np.diag([2.0, 3.0]), [-2.0, 1.0])],
        trace_size=2,
        kernel_offsets=np.array([2, 2]),
    )
    assert_array_equal(matrix.toarray(), np.diag([3.0, 2.0]))
    assert_array_equal(rhs, [1, -2])
    assert_array_equal(scale, [1, 2])
