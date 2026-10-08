"""Finite symmetric nonnegative Brinkman resistance evaluation in two dimensions."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray


def resistance_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate a symmetric nonnegative scalar or tensor Brinkman resistance."""
    raw = field(points) if callable(field) else field
    if np.iscomplexobj(raw):
        raise ValueError("drag must be real")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("drag must be finite")
    if array.ndim == 0 or array.shape == (len(points),):
        result = np.broadcast_to(array, (len(points),))[:, None, None] * np.eye(2)
    else:
        result = np.broadcast_to(array, (len(points), 2, 2)).copy()
    if (
        not np.isfinite(result).all()
        or not np.allclose(result, result.swapaxes(-1, -2), rtol=1e-12, atol=1e-14)
        or np.any(np.linalg.eigvalsh(result) < 0)
    ):
        raise ValueError("drag must be finite, symmetric and nonnegative")
    return result
