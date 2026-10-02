"""Shared geometry and material data for the square-obstacle Darcy benchmark.

The unit square contains a centered square of area 0.25. Material interfaces
cross the uniform macrogrid and align with even local refinements.
Finite-well supports remain fixed, preserving their integrated strengths.
"""

from __future__ import annotations

import numpy as np

from pymhm import TriangleMesh

OBSTACLE_AREA = 0.25
OBSTACLE_SIDE = 0.5
OBSTACLE_LOWER = 0.25
OBSTACLE_UPPER = 0.75


def macro_mesh() -> TriangleMesh:
    """Return 200 uniform macrotriangles crossed by the square's material boundary."""
    return TriangleMesh.unit_square(10)


def coefficient(points: np.ndarray) -> np.ndarray:
    """Evaluate isotropic permeability 1e-4 inside the square and 1 outside."""
    inside = np.all((points > OBSTACLE_LOWER) & (points < OBSTACLE_UPPER), axis=-1)
    return np.where(inside, 1e-4, 1.0)


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate fixed finite wells with total extraction/injection rates -1/+1."""
    extraction = np.all(points < 0.1, axis=-1)
    injection = np.all(points > 0.9, axis=-1)
    return 100.0 * (injection.astype(float) - extraction)
