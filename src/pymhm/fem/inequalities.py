"""Discrete inverse constants in physical finite-element coordinates."""

import numpy as np
from scipy import linalg

from pymhm.core.validation import FloatArray


def laplacian_inverse_bound(
    gradient: FloatArray, hessian: FloatArray, weights: FloatArray, diameters: FloatArray
) -> FloatArray:
    """Return cellwise ``m=min(1/3,C)`` with ``C*h²*||Delta v||²<=||grad v||²``.

    Gradients have axes ``(cell, point, basis, physical_direction)`` and
    Hessians append a second physical direction. Positive quadrature weights
    integrate the reference cell; its physical volume cancels in the energy
    quotient. ``diameters`` are physical cell diameters. The basis must span
    the constants with exactly one resolved stiffness kernel per cell.

    The spectral quotient applies in two and three dimensions. It describes
    this supplied polynomial space and quadrature, rather than a mesh-uniform
    inf-sup certificate. The quadrature must integrate both Gram matrices
    exactly. Unresolved additional stiffness modes are rejected using the
    declared relative spectral threshold ``1e-12``. Piecewise-affine functions
    have zero Laplacian and return ``1/3``.
    """
    energy = np.einsum("q,tqia,tqja->tij", weights, gradient, gradient)
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)
    strong = np.einsum("q,tqi,tqj->tij", weights, laplacian, laplacian)
    constants = np.full(len(gradient), 1 / 3)
    for cell, (stiffness, residual) in enumerate(zip(energy, strong, strict=True)):
        values, vectors = linalg.eigh(stiffness)
        selected = values > 1e-12 * values[-1]
        if np.count_nonzero(selected) != len(values) - 1:
            raise ValueError("inverse inequality requires exactly 1 resolved kernel modes")
        basis = vectors[:, selected] / np.sqrt(values[selected])
        maximum = max(0.0, float(linalg.eigvalsh(basis.T @ residual @ basis)[-1]))
        if maximum > 0:
            constants[cell] = min(1 / 3, 1 / (diameters[cell] ** 2 * maximum))
    return constants
