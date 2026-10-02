"""Analytical data for independently checkable Darcy and Stokes benchmarks."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def darcy_pressure(points: FloatArray) -> FloatArray:
    """Return the cosine pressure of the mixed Darcy benchmark (unit K)."""
    return np.cos(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])


def darcy_source(points: FloatArray) -> FloatArray:
    """Return -Delta(p) for the cosine benchmark."""
    return 2 * np.pi**2 * darcy_pressure(points)


def darcy_flux(points: FloatArray) -> FloatArray:
    """Return -grad(p), with the physical Darcy flux convention."""
    x, y = points.T
    return np.pi * np.column_stack(
        (np.sin(np.pi * x) * np.cos(np.pi * y), np.cos(np.pi * x) * np.sin(np.pi * y))
    )


def stokes_velocity(points: FloatArray) -> FloatArray:
    """Return curl(psi) for psi=-128*x²*(x-1)²*y²*(y-1)².

    Differentiating one streamfunction fixes the relative signs and guarantees
    analytical incompressibility instead of relying on a transcribed component.
    """
    a = Polynomial([0, 0, 1, -2, 1])
    x, y = points.T
    return 128 * np.column_stack((-a(x) * a.deriv()(y), a.deriv()(x) * a(y)))


def stokes_pressure(points: FloatArray) -> FloatArray:
    """Return the zero-mean bilinear pressure 150*(x-1/2)*(y-1/2)."""
    x, y = points.T
    return 150 * (x - 0.5) * (y - 0.5)


def stokes_source(points: FloatArray, viscosity: float = 1.0, drag: float = 0.0) -> FloatArray:
    """Evaluate -nu*Delta(u)+drag*u+grad(p) by polynomial differentiation."""
    a = Polynomial([0, 0, 1, -2, 1])
    x, y = points.T
    laplacian = 128 * np.column_stack(
        (
            -(a.deriv(2)(x) * a.deriv()(y) + a(x) * a.deriv(3)(y)),
            a.deriv(3)(x) * a(y) + a.deriv()(x) * a.deriv(2)(y),
        )
    )
    pressure_gradient = 150 * np.column_stack((y - 0.5, x - 0.5))
    return -viscosity * laplacian + drag * stokes_velocity(points) + pressure_gradient
