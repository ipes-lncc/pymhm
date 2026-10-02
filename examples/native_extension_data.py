"""Exact smooth and boundary-layer data for enriched native PDE formulations."""

from dataclasses import dataclass

import numpy as np
from manufactured import stokes_source, stokes_velocity
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sine(points: FloatArray) -> FloatArray:
    """Evaluate the unit-square sine eigenfunction with zero boundary values."""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def sine_gradient(points: FloatArray) -> FloatArray:
    """Evaluate the exact gradient of the sine eigenfunction."""
    x, y = points.T
    return np.pi * np.column_stack(
        (np.cos(np.pi * x) * np.sin(np.pi * y), np.sin(np.pi * x) * np.cos(np.pi * y))
    )


def resistance(points: FloatArray) -> FloatArray:
    """Return the variable SPD Brinkman tensor (1+x+2y)*[[20,3],[3,1]]."""
    factor = 1 + points[:, 0] + 2 * points[:, 1]
    return factor[:, None, None] * np.array([[20.0, 3.0], [3.0, 1.0]])


def flow_source(points: FloatArray) -> FloatArray:
    """Add the full tensor resistance action to the manufactured Stokes force."""
    return stokes_source(points) + np.einsum(
        "nij,nj->ni", resistance(points), stokes_velocity(points)
    )


def rad_diffusion(points: FloatArray) -> FloatArray:
    """Return a variable SPD diffusion tensor with off-diagonal coupling."""
    factor = 0.05 * (1 + points[:, 0] + points[:, 1])
    return factor[:, None, None] * np.array([[2.0, 0.3], [0.3, 1.0]])


def rad_velocity(points: FloatArray) -> FloatArray:
    """Return a compressible affine velocity with divergence two."""
    return points + [1.0, 0.5]


def rad_reaction(points: FloatArray) -> FloatArray:
    """Return a nonnegative spatially variable reaction coefficient."""
    return 0.5 + points[:, 0] * points[:, 1]


def rad_source(points: FloatArray) -> FloatArray:
    """Evaluate -div(K grad(u))+div(beta*u)+c*u for the sine field."""
    x, y = points.T
    value = sine(points)
    cross = np.cos(np.pi * x) * np.cos(np.pi * y)
    diffusion = 0.05 * (1 + x + y) * np.pi**2 * (3 * value - 0.6 * cross)
    transport = np.sum((rad_velocity(points) - [0.115, 0.065]) * sine_gradient(points), axis=1)
    return diffusion + transport + (2.5 + x * y) * value


def heat_exact(points: FloatArray, time: float) -> FloatArray:
    """Return the smoothly growing heat field (1+t)*sin(pi*x)*sin(pi*y)."""
    return (1 + time) * sine(points)


def heat_source(points: FloatArray, time: float) -> FloatArray:
    """Evaluate u_t-Delta(u), with time-linear data to avoid temporal truncation."""
    return (1 + 2 * np.pi**2 * (1 + time)) * sine(points)


@dataclass(frozen=True)
class BoundaryLayer:
    """Exact conservative transport with unit velocity with outflow layer of width epsilon."""

    epsilon: float

    def __post_init__(self) -> None:
        """Require strictly positive, finite diffusivity."""
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    def value(self, points: FloatArray) -> FloatArray:
        """Evaluate x minus the normalized outflow exponential on the whole boundary."""
        x = points[:, 0]
        return x - (np.exp((x - 1) / self.epsilon) - np.exp(-1 / self.epsilon)) / (
            -np.expm1(-1 / self.epsilon)
        )

    def gradient(self, points: FloatArray) -> FloatArray:
        """Evaluate the axial derivative and an identically zero transverse derivative."""
        exponential = np.exp((points[:, 0] - 1) / self.epsilon)
        dx = 1 - exponential / (-self.epsilon * np.expm1(-1 / self.epsilon))
        return np.column_stack((dx, np.zeros_like(dx)))

    @property
    def maximum(self) -> float:
        """Return the exact positive interior maximum, not the upper bound one."""
        x = 1 + self.epsilon * np.log(-self.epsilon * np.expm1(-1 / self.epsilon))
        return float(self.value(np.array([[x, 0.5]]))[0])
