"""Explicit two-component refinement for independently assembled native references.

A coefficient is the unevaluated sum of two binary64 values. Error-free products
and sums evaluate the original real CSR equations; native factors remain double
precision. No matrix entry, physical scale or residual tolerance is modified.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse


def two_sum(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a binary64 sum into its rounded value and exact rounding remainder."""
    total = first + second
    virtual = total - first
    return total, (first - (total - virtual)) + (second - virtual)


def two_product(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a finite binary64 product by Dekker's 27-bit significand partition."""
    product = first * second
    scaled = (2**27 + 1) * first
    high_first = scaled - (scaled - first)
    low_first = first - high_first
    scaled = (2**27 + 1) * second
    high_second = scaled - (scaled - second)
    low_second = second - high_second
    remainder = (
        ((high_first * high_second - product) + high_first * low_second) + low_first * high_second
    ) + low_first * low_second
    if not np.all(np.isfinite(product)) or not np.all(np.isfinite(remainder)):
        raise ArithmeticError("compensated products exceeded their finite arithmetic range")
    return product, remainder


def component_residual(
    matrix: sparse.csr_matrix, rhs: np.ndarray, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    """Return the original residual with both coefficient components and compensated rows.

    The real finite input is restricted to values whose Dekker splitting does
    not overflow or underflow materially. This is checked on the native reference
    fixtures, independently against Decimal arithmetic. The returned residual is
    binary64; rounding that already-small residual does not discard cancellation
    digits from the much larger operator products.
    """
    result = np.empty(len(rhs))
    lengths = np.diff(matrix.indptr)
    for start in range(0, len(rhs), 16384):
        rows = np.arange(start, min(start + 16384, len(rhs)))
        begins, count = matrix.indptr[rows], lengths[rows]
        total = np.asarray(rhs[rows], dtype=float).copy()
        remainder = np.zeros(len(rows))
        for column in range(count.max(initial=0)):
            active = np.flatnonzero(count > column)
            indices = begins[active] + column
            owners = matrix.indices[indices]
            product, error = two_product(-matrix.data[indices], high[owners])
            correction, tail = two_product(-matrix.data[indices], low[owners])
            error, carry = two_sum(error, correction)
            carry += tail
            summed, dropped = two_sum(total[active], product)
            small, final = two_sum(remainder[active], error)
            dropped += small
            total[active], remainder[active] = two_sum(summed, dropped)
            remainder[active] += final + carry
        result[rows] = total + remainder
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("compensated residual is nonfinite")
    return result


def refine_components(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    solve: Callable[[np.ndarray], np.ndarray],
    *,
    rtol: float = 1e-10,
    iterations: int = 6,
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    residual_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Reuse a native factorization while retaining both correction components.

    By default the target is the original CSR equation. An explicit ``residual``
    may instead evaluate the intended physical weak form; then the factorization
    is a preconditioner for that target, and its own residual is a separate
    diagnostic. ``residual_scale`` specifies the target's RHS norm when different
    from the initial sparse RHS. No operator entries or tolerance are adjusted.
    """
    high = np.array(solve(rhs), dtype=float, copy=True)
    low = np.zeros_like(high)
    scale = np.linalg.norm(rhs) if residual_scale is None else residual_scale
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("residual scale must be finite and nonnegative")
    history = []
    for step in range(iterations + 1):
        defect = (
            component_residual(matrix, rhs, high, low) if residual is None else residual(high, low)
        )
        error = float(np.linalg.norm(defect))
        history.append(error / scale if scale else error)
        if error <= rtol * scale:
            return high, low, history
        if step != iterations:
            correction = np.asarray(solve(np.asarray(defect, dtype=float)), dtype=float)
            high, carry = two_sum(high, correction)
            low += carry
            high, low = two_sum(high, low)
    raise ArithmeticError(f"target residual criterion failed after refinement: {history[-1]}")
