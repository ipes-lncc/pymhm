"""Independent divergence-free analytical data for three-dimensional weak-symmetry elasticity."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolenoidalElasticity3D:
    """Curl of (0,0,prod sin²(pi*x_i)); force is independent of the Lamé bulk modulus.

    Displacement vanishes on every cube face. The exact mean pressure is zero
    for finite lambda and at incompressibility. Complex arguments remain
    analytic, enabling independent complex-step derivative checks.
    """

    mu: float = 1.0

    def fields(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return displacement, Cartesian gradient and independently differentiated body force."""
        p = np.asarray(points)
        p = p.astype(np.result_type(p.dtype, np.float64), copy=False)
        f, d, dd, ddd = (
            np.sin(np.pi * p) ** 2,
            np.pi * np.sin(2 * np.pi * p),
            2 * np.pi**2 * np.cos(2 * np.pi * p),
            -4 * np.pi**3 * np.sin(2 * np.pi * p),
        )
        u = np.zeros(p.shape, dtype=p.dtype)
        u[:, 0], u[:, 1] = f[:, 0] * d[:, 1] * f[:, 2], -d[:, 0] * f[:, 1] * f[:, 2]
        gradient = np.zeros((len(p), 3, 3), dtype=p.dtype)
        gradient[:, 0, 0] = d[:, 0] * d[:, 1] * f[:, 2]
        gradient[:, 0, 1] = f[:, 0] * dd[:, 1] * f[:, 2]
        gradient[:, 0, 2] = f[:, 0] * d[:, 1] * d[:, 2]
        gradient[:, 1, 0] = -dd[:, 0] * f[:, 1] * f[:, 2]
        gradient[:, 1, 1] = -d[:, 0] * d[:, 1] * f[:, 2]
        gradient[:, 1, 2] = -d[:, 0] * f[:, 1] * d[:, 2]
        force = np.zeros_like(p)
        force[:, 0] = -self.mu * (
            dd[:, 0] * d[:, 1] * f[:, 2]
            + f[:, 0] * ddd[:, 1] * f[:, 2]
            + f[:, 0] * d[:, 1] * dd[:, 2]
        )
        force[:, 1] = self.mu * (
            ddd[:, 0] * f[:, 1] * f[:, 2]
            + d[:, 0] * dd[:, 1] * f[:, 2]
            + d[:, 0] * f[:, 1] * dd[:, 2]
        )
        return u, gradient, force

    def displacement(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the homogeneous-boundary solenoidal displacement."""
        return self.fields(points)[0]

    def source(self, points: np.ndarray) -> np.ndarray:
        """Evaluate -mu*Laplacian(u); the grad(div u) term vanishes identically."""
        return self.fields(points)[2]

    def stress(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the exact Cauchy stress, including zero hydrostatic pressure."""
        gradient = self.fields(points)[1]
        return self.mu * (gradient + gradient.swapaxes(-1, -2))

    def rotation(self, points: np.ndarray) -> np.ndarray:
        """Evaluate axial skew-gradient coordinates, equal to minus one half curl(u)."""
        g = self.fields(points)[1]
        return (
            np.column_stack(
                (g[:, 1, 2] - g[:, 2, 1], g[:, 2, 0] - g[:, 0, 2], g[:, 0, 1] - g[:, 1, 0])
            )
            / 2
        )
