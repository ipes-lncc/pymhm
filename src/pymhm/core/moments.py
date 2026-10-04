"""Operator-independent reconstruction from physical coefficient moments."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray
from pymhm.linalg.moments import solve_moment_system


def energy_reconstruction(
    matrix: Any,
    moments: FloatArray,
    refinement_precision: Literal["double", "extended"] = "double",
) -> tuple[FloatArray, FloatArray]:
    """Construct ``R`` with ``A R + C mu = 0`` and ``C.T R = I``.

    ``matrix`` is the finite real square coefficient operator A. ``moments``
    contains a nonempty set of independent real moment columns C in the original coefficient
    basis; they retain their physical units. The augmented A/C operator must
    be invertible. Moment independence alone does not prove that condition,
    stability of a discretization, or a coercivity estimate.

    Return the reconstruction R and the represented reduction ``R.T A R``.
    For a symmetric positive semidefinite A coercive on the moment-zero space,
    R minimizes its energy under the declared moments. For a nonsymmetric A,
    the same original saddle defines a constrained Galerkin reconstruction;
    its reduction keeps the full bilinear operator, including antisymmetry.
    Neither operator nor reduction is projected onto its symmetric part.
    Wider arithmetic follows ``solve_moment_system`` without changing rows,
    moment units or the original 1e-10 residual criterion.
    """
    scale = np.linalg.norm(moments, axis=0)
    if np.any(scale == 0) or np.linalg.matrix_rank(moments / scale) != moments.shape[1]:
        raise ValueError("local space cannot represent all independent cell and face moments")
    # Native equilibration acts only inside the fixed factorization. Acceptance
    # and corrections use the original physical A/C rows and integral units.
    augmented = sparse.bmat([[matrix, moments], [moments.T, None]], format="csc")
    rhs = np.vstack((np.zeros((matrix.shape[0], len(scale))), np.eye(len(scale))))
    reconstruction = solve_moment_system(augmented, rhs, refinement_precision=refinement_precision)[
        : matrix.shape[0]
    ]
    energy = reconstruction.T @ (matrix @ reconstruction)
    return reconstruction, energy
