"""Compiled binary64 quadrature leaves with compensated real accumulation.

The leaves consume already validated, cellwise arrays. They retain every
represented quadrature term without clipping, kernel projection or reassociation.
Compilation releases the GIL and uses the process-local Numba disk cache; no
additional worker pool or fast-math transformations are enabled.
"""

import numpy as np
from numba import njit

from pymhm.core.validation import FloatArray


@njit(cache=True, nogil=True)
def ordinary_product_range(values: FloatArray) -> bool:
    """Bound nonzero operands so five-factor products retain binary64 range.

    The conservative interval ``[2**-180,2**180]`` leaves exponent headroom
    for Cartesian and quadrature sums. Zero coefficients remain exact zeros.
    Values outside this interval require the owner's exceptional range path;
    this check does not clip, reject or normalize their represented values.
    """
    lower, upper = 2.0**-180, 2.0**180
    for value in values.flat:
        magnitude = abs(value)
        if magnitude != 0.0 and not lower <= magnitude <= upper:
            return False
    return True


@njit(cache=True, nogil=True, inline="always")
def _compensated_add(total: float, correction: float, value: float) -> tuple[float, float]:
    """Accumulate one real term using Neumaier's magnitude-aware compensation."""
    combined = total + value
    if abs(total) >= abs(value):
        correction += (total - combined) + value
    else:
        correction += (value - combined) + total
    return combined, correction


@njit(cache=True, nogil=True)
def diffusion_blocks(
    weights: FloatArray,
    gradients: FloatArray,
    tensors: FloatArray,
    measures: FloatArray,
) -> FloatArray:
    """Integrate cellwise ``grad(v).K.grad(u)`` with compensated contractions.

    Array shapes are ``(cells,q)``, ``(cells,q,basis,dimension)``,
    ``(cells,q,dimension,dimension)`` and ``(cells,)``. The tensor action is
    evaluated once per quadrature point and basis function. Both Cartesian
    contractions and the quadrature sum are compensated in binary64. The output
    preserves the literal basis order and has shape ``(cells,basis,basis)``.
    """
    cells, count, width, dimension = gradients.shape
    result = np.zeros((cells, width, width))
    transformed = np.empty((count, width, dimension))
    for cell in range(cells):
        for q in range(count):
            for j in range(width):
                for a in range(dimension):
                    total = 0.0
                    correction = 0.0
                    for b in range(dimension):
                        value = tensors[cell, q, a, b] * gradients[cell, q, j, b]
                        total, correction = _compensated_add(total, correction, value)
                    transformed[q, j, a] = total + correction
        for i in range(width):
            for j in range(width):
                total = 0.0
                correction = 0.0
                for q in range(count):
                    contraction = 0.0
                    contraction_correction = 0.0
                    for a in range(dimension):
                        value = gradients[cell, q, i, a] * transformed[q, j, a]
                        contraction, contraction_correction = _compensated_add(
                            contraction, contraction_correction, value
                        )
                    value = (
                        (contraction + contraction_correction) * weights[cell, q] * measures[cell]
                    )
                    total, correction = _compensated_add(total, correction, value)
                result[cell, i, j] = total + correction
    return result


@njit(cache=True, nogil=True)
def boundary_moments(basis: FloatArray, weights: FloatArray, values: FloatArray) -> FloatArray:
    """Integrate real trace moments without altering the represented quadrature.

    Inputs have shapes ``(q,basis)``, ``(q,)`` and ``(q,components)``. Terms
    retain their left-to-right product convention; Neumaier accumulation
    prevents large opposite moments from erasing a small represented term.
    """
    result = np.zeros((basis.shape[1], values.shape[1]))
    for i in range(basis.shape[1]):
        for j in range(values.shape[1]):
            total = 0.0
            correction = 0.0
            for q in range(len(weights)):
                value = basis[q, i] * weights[q] * values[q, j]
                total, correction = _compensated_add(total, correction, value)
            result[i, j] = total + correction
    return result
