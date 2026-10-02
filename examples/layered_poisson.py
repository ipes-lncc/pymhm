"""An analytical sine-series reference for two-layer unit-square Poisson data."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayeredPoissonSeries:
    """Solve -div(a grad(p))=1, p=0, with a=contrast below y=1/2 and one above.

    The infinite sine series in x is an analytical solution. ``modes`` truncates
    its odd Fourier modes; reported numerical errors must distinguish this
    truncation from discretization and integration errors. Stable exponential
    ratios avoid overflowing hyperbolic functions at high mode numbers.
    """

    contrast: float = 10.0
    modes: int = 511

    def _coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return wave numbers, forcing amplitudes and continuous interface values."""
        wave = np.pi * np.arange(1, 2 * self.modes, 2)
        forcing = 4 / wave
        interface = (
            2 * forcing / wave**2 * np.tanh(wave / 4) * np.tanh(wave / 2) / (self.contrast + 1)
        )
        return wave, forcing, interface

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate truncated pressure and its one-sided analytical gradient."""
        wave, forcing, interface = self._coefficients()
        pressure = np.empty(len(points))
        gradient = np.empty((len(points), 2))
        for start in range(0, len(points), 2048):
            stop = min(start + 2048, len(points))
            x, y = points[start:stop].T
            below = y < 0.5
            distance = np.where(below, y, 1 - y)[:, None]
            coefficient = np.where(below, self.contrast, 1)[:, None]
            growing = np.exp(wave * (distance - 0.5))
            decaying = np.exp(-wave * distance)
            reverse = np.exp(-wave * (distance + 0.5))
            source_part = 1 - (growing + decaying) / (1 + np.exp(-wave / 2))
            trace_part = (growing - reverse) / (1 - np.exp(-wave))
            amplitudes = forcing / (coefficient * wave**2) * source_part + interface * trace_part
            derivative = -forcing / (coefficient * wave) * (growing - decaying) / (
                1 + np.exp(-wave / 2)
            ) + interface * wave * (growing + reverse) / (1 - np.exp(-wave))
            sine, cosine = np.sin(x[:, None] * wave), np.cos(x[:, None] * wave)
            pressure[start:stop] = np.sum(sine * amplitudes, axis=1)
            gradient[start:stop, 0] = np.sum(wave * cosine * amplitudes, axis=1)
            gradient[start:stop, 1] = np.where(below, 1, -1) * np.sum(sine * derivative, axis=1)
        return pressure, gradient

    def energy_squared(self) -> float:
        """Integrate f*p analytically for the same truncated sine series."""
        wave, forcing, interface = self._coefficients()
        hyperbolic = np.tanh(wave / 4)
        integral_y = (
            forcing / wave**2 * (1 / self.contrast + 1) * (0.5 - 2 * hyperbolic / wave)
            + 2 * interface * hyperbolic / wave
        )
        return float(np.sum(2 * integral_y / wave))
