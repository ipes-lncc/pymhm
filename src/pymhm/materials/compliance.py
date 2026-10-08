"""Positive finite or infinite Lamé compressibility with explicit reciprocal convention."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray


def compressibility_values(lame_lambda: Any, points: FloatArray) -> FloatArray:
    """Evaluate inverse first Lame modulus, allowing positive infinity pointwise."""
    raw = lame_lambda(points) if callable(lame_lambda) else lame_lambda
    if np.iscomplexobj(raw):
        raise ValueError("Lame lambda must be real")
    coefficient = np.broadcast_to(np.asarray(raw, dtype=float), (len(points),))
    if np.isnan(coefficient).any() or np.any(coefficient <= 0):
        raise ValueError("Lame lambda must be positive or infinity")
    return 1 / coefficient
