"""Analytical data for anisotropic elasticity and three-dimensional elliptic checks."""

import numpy as np


def compliance() -> np.ndarray:
    """Return a full Cartesian compliance with anisotropic symmetric and positive skew blocks."""
    kelvin = np.array([[6.0, 2, 0.7], [2, 4, -0.4], [0.7, -0.4, 3]])
    transform = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [0, 1, 0, 0]], float)
    transform[:, 2:] /= np.sqrt(2)
    block = np.zeros((4, 4))
    block[:3, :3], block[3, 3] = np.linalg.inv(kelvin), 0.5
    return transform @ block @ transform.T


def displacement(points: np.ndarray) -> np.ndarray:
    """Evaluate u=(x^2 y^2, x^3 y), including its nonzero displacement boundary."""
    x, y = points.T
    return np.column_stack((x * x * y * y, x**3 * y))


def stress(points: np.ndarray) -> np.ndarray:
    """Invert the constitutive law on the independently differentiated symmetric gradient."""
    x, y = points.T
    strain = np.column_stack((2 * x * y * y, 2.5 * x * x * y, 2.5 * x * x * y, x**3))
    return np.linalg.solve(compliance(), strain.T).T.reshape(-1, 2, 2)


def force(points: np.ndarray) -> np.ndarray:
    """Evaluate minus stress divergence analytically, without differentiating a discrete field."""
    x, y = points.T
    dx = np.column_stack((2 * y * y, 5 * x * y, 5 * x * y, 3 * x * x))
    dy = np.column_stack((4 * x * y, 2.5 * x * x, 2.5 * x * x, np.zeros(len(x))))
    sx = np.linalg.solve(compliance(), dx.T).T
    sy = np.linalg.solve(compliance(), dy.T).T
    return -np.column_stack((sx[:, 0] + sy[:, 1], sx[:, 2] + sy[:, 3]))


def rotation(points: np.ndarray) -> np.ndarray:
    """Return half the antisymmetric displacement gradient in the mixed convention."""
    return -0.5 * points[:, 0] ** 2 * points[:, 1]


def pressure3d(points: np.ndarray) -> np.ndarray:
    """Smooth unit-cube pressure with homogeneous boundary values."""
    return np.prod(np.sin(np.pi * points), axis=1)


def flux3d(points: np.ndarray) -> np.ndarray:
    """Exact physical Darcy flux for identity permeability."""
    return -np.pi * np.column_stack(
        [
            np.cos(np.pi * points[:, i])
            * np.prod(np.sin(np.pi * points[:, [j for j in range(3) if j != i]]), axis=1)
            for i in range(3)
        ]
    )


def source3d(points: np.ndarray) -> np.ndarray:
    """Evaluate minus Laplacian of the prescribed pressure."""
    return 3 * np.pi**2 * pressure3d(points)
