"""Smooth three-dimensional elasticity with bounded forcing as lambda tends to infinity."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaLS3DData:
    """A polynomial solenoidal displacement plus an exactly compressible pressure potential."""

    lame_lambda: float = 1e8
    variable_shear: bool = True

    @staticmethod
    def _bubble(t: np.ndarray, derivative: int) -> np.ndarray:
        """Return derivatives through order three of t(1-t)."""
        coefficients = ([0, 1, -1], [1, -2], [-2], [0])
        return np.polynomial.polynomial.polyval(t, coefficients[derivative])

    @property
    def inverse_lambda(self) -> float:
        """Return zero compliance at exact incompressibility."""
        return 0.0 if np.isinf(self.lame_lambda) else 1 / self.lame_lambda

    def shear(self, points: np.ndarray) -> np.ndarray:
        """Evaluate mu=1+x/4+z/8, or constant one for the primal control."""
        return 1 + (
            0.25 * points[:, 0] + 0.125 * points[:, 2]
            if self.variable_shear
            else np.zeros(len(points))
        )

    def shear_gradient(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the exact shear gradient."""
        return np.broadcast_to(
            [0.25, 0.0, 0.125] if self.variable_shear else [0.0, 0.0, 0.0], points.shape
        )

    @property
    def shear_bounds(self) -> tuple[float, float, float]:
        """Return certified value and gradient bounds throughout the unit cube."""
        return (1.0, 1.375, np.sqrt(0.25**2 + 0.125**2)) if self.variable_shear else (1.0, 1.0, 0.0)

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Return the zero-mean pressure cos(pi*x) cos(pi*y) cos(pi*z)."""
        return np.prod(np.cos(np.pi * points), axis=1)

    def pressure_gradient(self, points: np.ndarray) -> np.ndarray:
        """Differentiate the trigonometric pressure independently in each component."""
        cosine, sine = np.cos(np.pi * points), np.sin(np.pi * points)
        return np.column_stack(
            [
                -np.pi * sine[:, i] * np.prod(cosine[:, [j for j in range(3) if j != i]], axis=1)
                for i in range(3)
            ]
        )

    def _pressure_hessian(self, points: np.ndarray) -> np.ndarray:
        """Return the symmetric physical pressure Hessian."""
        result = np.zeros((len(points), 3, 3))
        for i in range(3):
            result[:, i, i] = -(np.pi**2) * self.pressure(points)
            for j in range(i):
                k = 3 - i - j
                result[:, i, j] = result[:, j, i] = (
                    np.pi**2
                    * np.sin(np.pi * points[:, i])
                    * np.sin(np.pi * points[:, j])
                    * np.cos(np.pi * points[:, k])
                )
        return result

    def _solenoidal(self, points: np.ndarray, derivative: int | None = None) -> np.ndarray:
        """Differentiate curl(0,0,8*b(x)b(y)b(z)) by a chosen Cartesian coordinate."""
        result = np.zeros((len(points), 3))
        for component, powers, sign in [(0, [0, 1, 0], 1.0), (1, [1, 0, 0], -1.0)]:
            orders = np.array(powers)
            if derivative is not None:
                orders[derivative] += 1
            result[:, component] = (
                8
                * sign
                * np.prod([self._bubble(points[:, i], int(orders[i])) for i in range(3)], axis=0)
            )
        return result

    def displacement(self, points: np.ndarray) -> np.ndarray:
        """Return u0+grad(p)/(3*pi²*lambda), with div(u)=-p/lambda."""
        return self._solenoidal(points) + self.inverse_lambda * self.pressure_gradient(points) / (
            3 * np.pi**2
        )

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Return the exact displacement gradient, with component before derivative."""
        return np.stack(
            [self._solenoidal(points, i) for i in range(3)], axis=-1
        ) + self.inverse_lambda * self._pressure_hessian(points) / (3 * np.pi**2)

    def source(self, points: np.ndarray) -> np.ndarray:
        """Evaluate -div(2mu*epsilon(u))+grad(p), with a lambda-uniform body force."""
        laplacian = np.zeros((len(points), 3))
        for component, powers, sign in [(0, [0, 1, 0], 1.0), (1, [1, 0, 0], -1.0)]:
            for derivative in range(3):
                orders = np.array(powers)
                orders[derivative] += 2
                laplacian[:, component] += (
                    8
                    * sign
                    * np.prod(
                        [self._bubble(points[:, i], int(orders[i])) for i in range(3)], axis=0
                    )
                )
        gradient = self.gradient(points)
        return (
            -self.shear(points)[:, None] * laplacian
            + (1 + 2 * self.shear(points) * self.inverse_lambda)[:, None]
            * self.pressure_gradient(points)
            - np.einsum(
                "nij,nj->ni", gradient + gradient.swapaxes(1, 2), self.shear_gradient(points)
            )
        )

    def stress(self, points: np.ndarray) -> np.ndarray:
        """Evaluate full Cauchy stress without subtracting nearly equal lambda terms."""
        gradient = self.gradient(points)
        return self.shear(points)[:, None, None] * (
            gradient + gradient.swapaxes(1, 2)
        ) - self.pressure(points)[:, None, None] * np.eye(3)
