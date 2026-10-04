"""Sparse assembly of explicitly indexed element blocks."""

from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray


def assemble_element_blocks(
    blocks: FloatArray, rows: IntArray, columns: IntArray, shape: tuple[int, int]
) -> Any:
    """Scatter cell-major dense blocks into a CSC operator in the declared DOF order.

    ``blocks[cell,row,column]``, ``rows[cell,row]`` and ``columns[cell,column]``
    describe independent row and column spaces. COO entries use C-order ravel;
    conversion to CSC sums all contributions to repeated global coordinates.
    """
    return sparse.coo_matrix(
        (
            blocks.ravel(),
            (
                np.repeat(rows, columns.shape[1], axis=1).ravel(),
                np.tile(columns, (1, rows.shape[1])).ravel(),
            ),
        ),
        shape=shape,
    ).tocsc()
