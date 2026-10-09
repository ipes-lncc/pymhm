"""Compiled binary64 leaves for planar ownership and affine coordinates.

The containing Python owner retains the centroid search, numerical validation
and exact rational predicates. These leaves fuse the finite-precision filter
without changing its error bound or moving physical query coordinates. They
release the GIL, use Numba's disk cache and create no additional worker pool.
Fast-math reassociation is deliberately disabled.
"""

import numpy as np
from numba import njit

from pymhm.core.validation import FloatArray, IntArray

_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny


@njit(cache=True, nogil=True)
def containment_filter(
    points: FloatArray,
    candidates: IntArray,
    vertices: FloatArray,
    areas: FloatArray,
    errors: FloatArray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return filtered membership and entries requiring exact physical predicates.

    ``points`` has shape ``(query,2)`` and ``candidates`` has shape
    ``(query,slot)``. ``vertices`` retains each candidate triangle's literal
    coordinates and orientation. Signed twice-areas and their conservative
    binary64 bounds are supplied by the mesh owner. Nonfinite and underflowed
    determinant terms remain ambiguous; an exact stored vertex is accepted.
    Both returned boolean arrays retain the query and candidate-slot axes.
    """
    inside = np.empty(candidates.shape, dtype=np.bool_)
    pending = np.empty(candidates.shape, dtype=np.bool_)
    for query in range(candidates.shape[0]):
        for slot in range(candidates.shape[1]):
            cell = candidates[query, slot]
            area = areas[cell]
            area_error = errors[cell]
            sign = -1 if area < 0 else 1
            threshold = tolerance * abs(area)
            uncertain_area = (abs(area) <= area_error) or not np.isfinite(area)
            uncertain = (
                (tolerance != 0) and (area != 0) and (abs(threshold) < _TINY)
            ) or uncertain_area
            outside = False
            enclosed = True
            vertex = False
            for edge in range(3):
                vertex = vertex or (
                    vertices[cell, edge, 0] == points[query, 0]
                    and vertices[cell, edge, 1] == points[query, 1]
                )
                first, second = (edge + 1) % 3, (edge + 2) % 3
                ax = vertices[cell, first, 0] - points[query, 0]
                ay = vertices[cell, first, 1] - points[query, 1]
                bx = vertices[cell, second, 0] - points[query, 0]
                by = vertices[cell, second, 1] - points[query, 1]
                product1, product2 = ax * by, ay * bx
                numerator = product1 - product2
                error = 8 * _EPS * (abs(product1) + abs(product2))
                underflow = (ax != 0 and by != 0 and abs(product1) < _TINY) or (
                    ay != 0 and bx != 0 and abs(product2) < _TINY
                )
                if underflow or not np.isfinite(numerator) or not np.isfinite(error):
                    error = np.inf
                score = sign * numerator + threshold
                bound = error + tolerance * area_error
                bound += 8 * _EPS * (abs(numerator) + abs(threshold))
                uncertain = uncertain or abs(score) <= bound or not np.isfinite(bound + score)
                outside = outside or score < -bound
                enclosed = enclosed and score >= 0
            inside[query, slot] = enclosed or vertex
            pending[query, slot] = uncertain and not (outside and not uncertain_area) and not vertex
    return inside, pending


@njit(cache=True, nogil=True)
def candidate_barycentric(
    points: FloatArray, candidates: IntArray, origins: FloatArray, inverse: FloatArray
) -> FloatArray:
    """Apply declared affine maps without gathered cross-candidate tensors.

    Query coordinates are neither clamped nor reconciled across incident
    triangles. The output has shape ``(query,slot,3)`` and preserves the
    original two-term dot-product order and ``1-(lambda1+lambda2)`` convention.
    Map arrays may be read-only and candidate arrays may have arbitrary strides.
    """
    output = np.empty((candidates.shape[0], candidates.shape[1], 3), dtype=np.float64)
    for query in range(candidates.shape[0]):
        for slot in range(candidates.shape[1]):
            cell = candidates[query, slot]
            x = points[query, 0] - origins[cell, 0]
            y = points[query, 1] - origins[cell, 1]
            first = inverse[cell, 0, 0] * x + inverse[cell, 0, 1] * y
            second = inverse[cell, 1, 0] * x + inverse[cell, 1, 1] * y
            output[query, slot, 0] = 1 - (first + second)
            output[query, slot, 1] = first
            output[query, slot, 2] = second
    return output
