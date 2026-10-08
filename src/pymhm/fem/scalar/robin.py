"""Affine Robin boundary forms and physical nodal volume moments in two dimensions.

These integrate supplied finite-element bases; they do not select a hybrid
method, solve a PDE, or infer a multiplier convention. Robin sigma is explicitly
parameter*(x-origin)/2 and normals point outward from the supplied fine mesh.
"""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.boundary_forms import boundary_bilinear, boundary_quadrature
from pymhm.meshes.triangle import TriangleMesh


def nodal_volume_moments(fine: TriangleMesh, degree: int) -> FloatArray:
    """Integrate every local nodal basis function for the physical pressure mean."""
    bary, weights = triangle_quadrature(degree + 1)
    dofs, points, basis, _, _ = tabulate(fine, degree, bary)
    result = np.zeros(len(points))
    local = np.einsum("q,qi,t->ti", weights, basis, fine.areas)
    np.add.at(result, dofs.ravel(), local.ravel())
    return result


def robin_boundary_operator(
    fine: TriangleMesh, degree: int, parameter: float, origin: FloatArray, order: int
) -> Any:
    """Integrate sigma.n*u*v using the general boundary owner, sigma=parameter*(x-origin)/2."""
    size, rules = boundary_quadrature(fine, degree, order=order)
    return boundary_bilinear(
        size, rules, lambda points, normal: parameter * ((points - origin) @ normal) / 2
    )
