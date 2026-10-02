"""Portable float64-component storage for coefficients accumulated in wider precision."""

from __future__ import annotations

import numpy as np


def precision_fields(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """Name the portable high, correction and tail arrays for an NPZ field."""
    high, low, tail = split_precision(values)
    return {name: high, f"{name}_correction": low, f"{name}_tail": tail}


def split_precision(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Store a finite real array as three portable float64 arrays.

    Successive arrays store rounding remainders. Three components cover both
    the 64-bit and 113-bit significands used by native extended formats, within
    the float64 exponent range. This avoids a platform-specific ``longdouble``
    representation in NPZ. It does not increase the accuracy of the input.
    """
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError("precision archives require finite real arrays")
    remaining = np.asarray(array, dtype=np.longdouble).copy()
    parts = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(3):
            part = np.array(remaining, dtype=np.float64, copy=True)
            parts.append(part)
            remaining -= part
    if any(not np.isfinite(part).all() for part in parts) or np.any(remaining != 0):
        raise ValueError("precision archives require values in the float64 exponent range")
    return parts[0], parts[1], parts[2]


def restore_precision(
    high: np.ndarray, low: np.ndarray, tail: np.ndarray | None = None
) -> np.ndarray:
    """Add portable high/remainder arrays in the widest native NumPy real precision.

    On platforms where long double equals double, both arrays remain readable,
    but their native summed value has only double precision. The archived
    components retain the separate remainders for a wider-precision consumer.
    """
    first, remainder = np.asarray(high), np.asarray(low)
    final = np.zeros_like(first) if tail is None else np.asarray(tail)
    if (
        first.shape != remainder.shape
        or first.shape != final.shape
        or np.iscomplexobj(first)
        or np.iscomplexobj(remainder)
        or np.iscomplexobj(final)
        or not np.isfinite(first).all()
        or not np.isfinite(remainder).all()
        or not np.isfinite(final).all()
    ):
        raise ValueError("precision archive components must be finite real arrays of equal shape")
    return (
        first.astype(np.longdouble) + remainder.astype(np.longdouble) + final.astype(np.longdouble)
    )
