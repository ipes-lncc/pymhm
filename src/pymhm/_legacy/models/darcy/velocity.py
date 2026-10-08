"""Compatibility imports for the public physical/vector field owner."""

from pymhm.postprocessing.velocity import (
    PolynomialDarcyVelocity,
    PrimalDarcyVelocity,
    polynomial_darcy_velocity,
)
from pymhm.postprocessing.velocity import _TriangleLocator as _compat_TriangleLocator

__all__ = ["PolynomialDarcyVelocity", "PrimalDarcyVelocity", "polynomial_darcy_velocity"]

_TriangleLocator = _compat_TriangleLocator
