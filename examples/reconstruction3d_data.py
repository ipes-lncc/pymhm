"""Independent smooth and localized manufactured data for three-dimensional Darcy flow."""

import numpy as np


def fields(
    points: np.ndarray, localized: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pressure, physical flux and negative Laplacian on the unit cube."""
    x = np.asarray(points)
    if localized:
        alpha = 30.0
        distance = x - np.array([0.3, 0.4, 0.6])
        exponential = np.exp(-alpha * distance**2)
        polynomial = x * (1 - x)
        value = polynomial * exponential
        first = (1 - 2 * x - 2 * alpha * distance * polynomial) * exponential
        second = (
            -2
            - 4 * alpha * distance * (1 - 2 * x)
            + (4 * alpha**2 * distance**2 - 2 * alpha) * polynomial
        ) * exponential
        amplitude = 100.0
    else:
        value = np.sin(np.pi * x)
        first = np.pi * np.cos(np.pi * x)
        second = -(np.pi**2) * value
        amplitude = 1.0
    gradient = np.column_stack(
        [first[:, i] * np.prod(value[:, [j for j in range(3) if j != i]], axis=1) for i in range(3)]
    )
    laplacian = sum(
        second[:, i] * np.prod(value[:, [j for j in range(3) if j != i]], axis=1) for i in range(3)
    )
    return amplitude * value.prod(axis=1), -amplitude * gradient, -amplitude * laplacian
