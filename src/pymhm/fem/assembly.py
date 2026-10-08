"""Sparse scatter of user-integrated element forms in declared coefficient maps."""

from typing import Any

import numpy as np
from scipy import sparse


def assemble_element_blocks(
    blocks: Any,
    rows: Any,
    columns: Any,
    shape: tuple[int, int],
) -> Any:
    """Scatter element matrices to a CSC operator without selecting a PDE.

    ``blocks[e,i,j]`` pairs test coefficient ``rows[e,i]`` with trial
    coefficient ``columns[e,j]``. Repeated global coordinates sum, including
    repeated coordinates within an element. Square matrices declare identical
    trial/test maps and equal dimensions explicitly. Real/complex entries
    keep their numeric dtype. The caller owns integration, orientation signs
    and local basis transformations; none are inferred during this scatter.
    """
    values = np.asarray(blocks)
    tests = np.asarray(rows)
    trials = np.asarray(columns)
    dimensions = tuple(shape)
    if (
        len(dimensions) != 2
        or any(not isinstance(size, (int, np.integer)) or size < 0 for size in dimensions)
        or tests.ndim != 2
        or trials.ndim != 2
        or tests.dtype.kind not in "iu"
        or trials.dtype.kind not in "iu"
        or tests.shape[0] != trials.shape[0]
        or values.shape != (tests.shape[0], tests.shape[1], trials.shape[1])
        or values.dtype.kind not in "biufc"
        or not np.isfinite(values).all()
        or np.any(tests < 0)
        or np.any(trials < 0)
        or np.any(tests >= dimensions[0])
        or np.any(trials >= dimensions[1])
    ):
        raise ValueError("element blocks require finite entries and matching in-range integer maps")
    rows = np.repeat(tests, trials.shape[1], axis=1).ravel()
    columns = np.tile(trials, (1, tests.shape[1])).ravel()
    return sparse.coo_matrix((values.ravel(), (rows, columns)), shape=dimensions).tocsc()
