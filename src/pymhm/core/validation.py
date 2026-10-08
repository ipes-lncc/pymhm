"""Shared real array aliases and exact discrete parameter validation."""

from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def positive_int(value: Any, name: str, minimum: int = 1) -> int:
    """Validate a discrete parameter without silently truncating real numbers."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def real_array(value: Any, name: str) -> FloatArray:
    """Copy a finite real array into binary64 without discarding imaginary parts."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite and real")
    return np.array(raw, dtype=float, copy=True)


def dyadic_refinement(value: Any, name: str = "refinement") -> int:
    """Validate a positive power of two for nested simplex subdivisions.

    This is a geometric alignment condition, not a polynomial degree or
    mathematical stability certificate. The returned integer is unchanged.
    """
    result = positive_int(value, name)
    if result & (result - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return result
