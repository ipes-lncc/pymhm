"""Analytical PEC cavity modes for the published TM and a full vector 3D verification."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CavityMode:
    """Divergence-free cavity mode with unit permittivity and permeability."""

    dimension: int = 2
    wavenumber: float = 2 * np.pi

    @property
    def omega(self) -> float:
        """Return the physical temporal eigenfrequency sqrt(2)*wavenumber."""
        return float(np.sqrt(2) * self.wavenumber)

    def electric_shape(self, points: np.ndarray) -> np.ndarray:
        """Return a scalar TM field or three nonzero vector-mode components."""
        sine = np.sin(self.wavenumber * points)
        if self.dimension == 2:
            return (sine[:, 0] * sine[:, 1])[:, None]
        return np.column_stack(
            (
                sine[:, 1] * sine[:, 2],
                0.7 * sine[:, 2] * sine[:, 0],
                0.4 * sine[:, 0] * sine[:, 1],
            )
        )

    def magnetic_shape(self, points: np.ndarray) -> np.ndarray:
        """Return -curl(E_shape)/omega, differentiated analytically."""
        sine, cosine = np.sin(self.wavenumber * points), np.cos(self.wavenumber * points)
        if self.dimension == 2:
            return np.column_stack(
                (-sine[:, 0] * cosine[:, 1], cosine[:, 0] * sine[:, 1])
            ) / np.sqrt(2)
        curl = np.column_stack(
            (
                sine[:, 0] * (0.4 * cosine[:, 1] - 0.7 * cosine[:, 2]),
                sine[:, 1] * (cosine[:, 2] - 0.4 * cosine[:, 0]),
                sine[:, 2] * (0.7 * cosine[:, 0] - cosine[:, 1]),
            )
        )
        return -curl / np.sqrt(2)

    def electric(self, time: float, points: np.ndarray) -> np.ndarray:
        """Evaluate E at its actual time, without replacing staggered sampling by averaging."""
        return np.cos(self.omega * time) * self.electric_shape(points)

    def magnetic(self, time: float, points: np.ndarray) -> np.ndarray:
        """Evaluate H at its actual integer leapfrog time."""
        return np.sin(self.omega * time) * self.magnetic_shape(points)
