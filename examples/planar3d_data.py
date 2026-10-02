"""Exact transmission data for an oblique planar tensor interface in three dimensions."""

from dataclasses import dataclass

import numpy as np

from pymhm.planar_material import PlanarMaterial, PlanarRegion


@dataclass(frozen=True)
class Planar3DData:
    """Continuous piecewise-quadratic pressure with continuous affine physical flux.

    For t=n.x-b and K=k*A, p=(t+0.3*t**2)/k and q=-(1+0.6*t)*A*n.
    The same value p=0 and the same flux occur on both sides of the interface.
    The interior halfspace t<=0 has k=contrast; the background has k=1.
    """

    contrast: float = 25.0

    @property
    def normal(self) -> np.ndarray:
        """Return the unnormalized interface normal used in the analytical coordinate."""
        return np.array([1.0, 0.4, 0.2])

    @property
    def tensor(self) -> np.ndarray:
        """Return an SPD tensor with off-diagonal physical coupling."""
        return np.array([[2.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.5]])

    @property
    def material(self) -> PlanarMaterial:
        """Describe the exact halfspace independently of the local mesh."""
        return PlanarMaterial(
            self.tensor, (PlanarRegion([self.normal], [0.63], self.contrast * self.tensor),)
        )

    @property
    def source(self) -> float:
        """Return div(q)=-0.6*n.T*A*n, with no interface distribution."""
        return float(-0.6 * self.normal @ self.tensor @ self.normal)

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Evaluate continuous pressure, selecting the stated closed material side."""
        coordinate = points @ self.normal - 0.63
        scale = np.where(coordinate <= 0, self.contrast, 1.0)
        return (coordinate + 0.3 * coordinate**2) / scale

    def flux(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the physical flux, equal on both sides of the interface."""
        coordinate = points @ self.normal - 0.63
        return -(1 + 0.6 * coordinate[:, None]) * (self.tensor @ self.normal)

    def normal_flux(self, points: np.ndarray, *, normal: np.ndarray) -> np.ndarray:
        """Evaluate outward scalar flux for a specified original macroface normal."""
        return self.flux(points) @ normal
