"""Tetrahedral rigid modes and independent Cartesian traction projections.

Physical outward traction fixes the negative hybrid traction multiplier. Weak
Dirichlet moments retain the same signed scalar macroface basis conventions.
"""

from functools import partial
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary
from pymhm.materials.evaluation import vector_values_3d


def rigid_modes_3d(points: FloatArray, center: FloatArray) -> FloatArray:
    """Return three translations followed by e_x/e_y/e_z cross (x-center)."""
    points = np.asarray(points) - np.asarray(center)
    result = np.zeros((*points.shape[:-1], 3, 6))
    result[..., :3] = np.eye(3)
    for axis in range(3):
        result[..., 3 + axis] = np.cross(np.eye(3)[axis], points)
    return result


def vector_boundary_data_3d(
    skeleton: TriangularSkeleton, dirichlet: Any, traction: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Reuse signed scalar face quadrature independently for each physical vector component."""
    load = np.zeros((skeleton.size, 3))
    fixed = {}
    for component in range(3):
        prescribed = {
            face: partial(_component, datum=value, component=component)
            for face, value in traction.items()
        }
        load[:, component], scalar_fixed = _boundary(
            skeleton, partial(_component, datum=dirichlet, component=component), prescribed, order
        )
        fixed.update({3 * index + component: -value for index, value in scalar_fixed.items()})
    return load.ravel(), fixed


def _component(points: FloatArray, *, datum: Any, component: int) -> FloatArray:
    """Evaluate one validated physical-vector component for a scalar trace operation."""
    return vector_values_3d(datum, points)[:, component]


KELVIN_BASIS_3D = np.zeros((6, 3, 3))
KELVIN_BASIS_3D[:3] = np.einsum("ai,aj->aij", np.eye(3), np.eye(3))
for _mode, (_a, _b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
    KELVIN_BASIS_3D[_mode, _a, _b] = KELVIN_BASIS_3D[_mode, _b, _a] = 1 / np.sqrt(2)
