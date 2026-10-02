"""Analytical pressure and physical flux fields used by the Darcy figures."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Field = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def darcy_fields(name: str) -> tuple[Field, Field]:
    """Return pressure and physical Darcy flux callables for a named exact case.

    Input coordinates have shape (n, 2). Pressure has shape (n,) and flux
    has shape (n, 2). These are closed-form fields only; no reference solver,
    finite element assembly or comparison runner is involved.
    """
    if name == "aff":

        def pressure(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate the affine pressure with gradients one and two."""
            return 1 + points[:, 0] + 2 * points[:, 1]

        def flux(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate the constant physical flux for unit permeability."""
            return np.tile([-1.0, -2.0], (len(points), 1))

        return pressure, flux

    if name == "poly":

        def pressure(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate the quadratic polynomial pressure."""
            x, y = points.T
            return x**2 + y**2 + x * y

        def flux(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate minus the quadratic pressure gradient."""
            x, y = points.T
            return np.column_stack((-2 * x - y, -x - 2 * y))

        return pressure, flux

    if name == "cos":

        def pressure(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate the unit-frequency cosine pressure."""
            x, y = (np.pi * points).T
            return np.cos(x) * np.cos(y)

        def flux(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate minus the cosine pressure gradient."""
            x, y = (np.pi * points).T
            return np.pi * np.column_stack((np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)))

        return pressure, flux

    if name in ("layer", "layer1000"):
        contrast = 10.0 if name == "layer" else 1000.0

        def pressure(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate continuous pressure across the fitted material interface."""
            x, y = points.T
            return 1 + np.minimum(x, 0.5) + np.maximum(x - 0.5, 0) / contrast + 2 * y

        def flux(points: NDArray[np.float64]) -> NDArray[np.float64]:
            """Keep continuous normal flux and the physical tangential jump."""
            permeability = np.where(points[:, 0] < 0.5, 1.0, contrast)
            return np.column_stack((-np.ones(len(points)), -2 * permeability))

        return pressure, flux

    raise ValueError(f"Unknown analytical Darcy case: {name}")
