"""Finite separated scalar fields without rank fitting or coefficient averaging."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray


def factor_values(value: Any, points: FloatArray) -> FloatArray:
    """Evaluate a real one-coordinate factor with a finite scalar/vector contract."""
    raw = np.asarray(value(points) if callable(value) else value)
    if np.iscomplexobj(raw) or raw.shape not in ((), points.shape) or not np.isfinite(raw).all():
        raise ValueError(
            "separable factors must return finite real scalars or coordinate-shaped arrays"
        )
    return np.broadcast_to(np.asarray(raw, dtype=float), points.shape)


@dataclass(frozen=True)
class SeparableField:
    """Represent ``sum(a(x)*b(y) for a,b in terms)`` without approximation.

    Each factor is a finite real scalar or a callback accepting a one-dimensional
    coordinate array. An empty sum represents zero. Signed terms are allowed;
    a diffusion assembler separately verifies positivity at its Gauss points.
    """

    terms: tuple[tuple[Any, Any], ...]

    def __post_init__(self) -> None:
        """Normalize term pairs and reject nonscalar, nonfinite constant factors."""
        try:
            terms = tuple(tuple(term) for term in self.terms)
        except TypeError as exc:
            raise ValueError("separable terms must be pairs of one-coordinate factors") from exc
        if any(len(term) != 2 for term in terms):
            raise ValueError("separable terms must be pairs of one-coordinate factors")
        for term in terms:
            for factor in term:
                if not callable(factor):
                    if np.asarray(factor).ndim != 0:
                        raise ValueError("constant separable factors must be scalar")
                    factor_values(factor, np.array([0.0]))
        object.__setattr__(self, "terms", terms)

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate the declared scalar field at physical XY point pairs."""
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
            raise ValueError("separable field points must be finite real XY pairs")
        result = np.zeros(len(raw))
        for x, y in self.terms:
            result += factor_values(x, raw[:, 0]) * factor_values(y, raw[:, 1])
        if not np.isfinite(result).all():
            raise ValueError("separable field evaluation must remain finite")
        return result
