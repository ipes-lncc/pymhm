"""Finite coefficient evaluation with explicit two- and three-dimensional conventions.

Scalar, vector and SPD tensor inputs use the existing point-major layouts.
The dimension-specific symmetry checks and returned storage conventions stay
separate so material validation preserves each formulation's numerical contract.
"""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.core.validation import real_array as _real


def scalar_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate a finite real scalar coefficient on points shaped ``(n, 2)``."""
    value = field(points) if callable(field) else field
    if np.iscomplexobj(value):
        raise ValueError("scalar coefficient must be real")
    result = np.broadcast_to(np.asarray(value, dtype=float), (len(points),)).copy()
    if not np.isfinite(result).all():
        raise ValueError("scalar coefficient must be finite")
    return result


def vector_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate a finite real two-component coefficient with point-major storage."""
    value = field(points) if callable(field) else field
    if np.iscomplexobj(value):
        raise ValueError("vector coefficient must be real")
    result = np.broadcast_to(np.asarray(value, dtype=float), (len(points), 2)).copy()
    if not np.isfinite(result).all():
        raise ValueError("vector coefficient must be finite")
    return result


def tensor_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate an isotropic scalar or symmetric positive-definite 2x2 tensor."""
    raw = field(points) if callable(field) else field
    if np.iscomplexobj(raw):
        raise ValueError("diffusion tensor must be real")
    value = np.asarray(raw, dtype=float)
    if value.ndim == 0 or value.shape == (len(points),):
        result = np.broadcast_to(value, (len(points),))[:, None, None] * np.eye(2)
    else:
        result = np.broadcast_to(value, (len(points), 2, 2)).copy()
    if (
        not np.isfinite(result).all()
        or not np.allclose(result, result.swapaxes(1, 2), rtol=1e-12, atol=1e-14)
        or np.any(np.linalg.eigvalsh(result) <= 0)
    ):
        raise ValueError("diffusion tensor must be finite, symmetric and positive definite")
    return result


def scalar_values_3d(coefficient: Any, points: FloatArray) -> FloatArray:
    """Evaluate finite real scalar data at points of shape (n, 3)."""
    values = _real(
        coefficient(points) if callable(coefficient) else coefficient, "scalar coefficient"
    )
    try:
        return np.broadcast_to(values, (len(points),))
    except ValueError as exc:
        raise ValueError("scalar coefficient must return one value per point") from exc


def tensor_values_3d(coefficient: Any, points: FloatArray) -> FloatArray:
    """Evaluate positive scalar or symmetric positive-definite 3 by 3 diffusion."""
    values = _real(coefficient(points) if callable(coefficient) else coefficient, "diffusion")
    if values.ndim == 0 or values.shape == (len(points),):
        values = np.broadcast_to(values, (len(points),))[:, None, None] * np.eye(3)
    try:
        values = np.broadcast_to(values, (len(points), 3, 3))
    except ValueError as exc:
        raise ValueError("diffusion must return scalar values or 3 by 3 tensors") from exc
    scale = np.max(np.abs(values), axis=(1, 2))
    if np.any(
        np.max(np.abs(values - values.transpose(0, 2, 1)), axis=(1, 2)) > 1e-12 * scale
    ) or np.any(np.linalg.eigvalsh(values) <= 0):
        raise ValueError("diffusion must be symmetric positive definite")
    return values


def vector_values_3d(coefficient: Any, points: FloatArray) -> FloatArray:
    """Evaluate finite real vectors with exactly three Cartesian components."""
    values = _real(coefficient(points) if callable(coefficient) else coefficient, "vector data")
    try:
        return np.broadcast_to(values, (len(points), 3))
    except ValueError as exc:
        raise ValueError("vector data must have three components per point") from exc
