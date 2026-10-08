"""Physical normal moments of already assembled vector boundary functionals."""

from math import fsum
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.traces.interval import SkeletonSpace


def boundary_normal_integral(
    skeleton: Any, boundary: FloatArray, *, absolute: bool = False
) -> float:
    """Apply the hydrostatic trace to the assembled displacement boundary moments.

    This uses exactly the same quadrature as the global equations; a separate
    integration rule would make the finite-compressibility identity inconsistent.
    Compensated accumulation reduces cancellation between opposite boundaries
    on platforms with or without a wider native real type.
    ``absolute`` sums uncancelled moments for scale-invariant compatibility.
    """
    terms = []
    for face in skeleton.mesh.boundary_faces:
        if isinstance(skeleton, SkeletonSpace):
            start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
            tangent = end - start
            normal = np.array([tangent[1], -tangent[0]]) / skeleton.mesh.lengths[face]
            coefficients = skeleton.faces[face].constant_coefficients()[:, None] * normal
            products = boundary[skeleton.dofs(int(face))] * coefficients.ravel()
        else:
            indices = skeleton.dofs(int(face))
            normal = skeleton.mesh.normals[face]
            coefficients = skeleton.constant_coefficients[indices, None] * normal
            products = boundary.reshape(-1, len(normal))[indices] * coefficients
            products = products.ravel()
        terms.extend(abs(products) if absolute else products)
    return fsum(terms)


def require_balanced_moment(moment: float, absolute_terms: float, *, rtol: float = 1e-10) -> None:
    """Require a signed integral vanish relative to its uncancelled physical terms.

    This condition is independent of amplitude and physical units. Both inputs
    must come from the same integration rule. It supports flux compatibility,
    force balance and moment constraints without an absolute zero threshold.
    """
    if not np.isfinite(moment) or not np.isfinite(absolute_terms) or absolute_terms < 0:
        raise ValueError("moment and its nonnegative cancellation scale must be finite")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("moment comparison rtol must be finite and nonnegative")
    if abs(moment) > rtol * absolute_terms:
        raise ValueError("incompatible prescribed integral moment")


def require_compatible_displacement_flux(flux: float, absolute_moments: float) -> None:
    """Require volume displacement balance in incompressible elasticity."""
    try:
        require_balanced_moment(flux, absolute_moments)
    except ValueError as error:
        raise ValueError("incompatible incompressible displacement boundary data") from error
