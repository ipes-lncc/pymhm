"""Shared solves of represented moment-coordinate equations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray
from pymhm.linalg.linear import _accurate_residual, _checked, factorize


def solve_moment_system(
    matrix: Any,
    rhs: Any,
    *,
    solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> FloatArray:
    """Solve ``matrix @ result = rhs`` without discarding represented wider digits.

    ``matrix`` is a finite real square operator; ``rhs`` is a real vector or a
    matrix of independent right-hand sides. The named factorization validates
    solvability and owns its native resources within this call. No symmetry,
    boundary convention or physical meaning of a coefficient is inferred.

    Factors use binary64. ``refinement_precision='extended'`` retains the
    original operator and forcing in wider storage for two additional defect
    corrections and the unchanged 1e-10 check of every original RHS column.
    It requires a genuinely wider platform long-double representation and
    returns wider coefficients; it never substitutes binary64 silently.
    Local reconstructions, condensed operators and gauged moment equations in
    all supported spatial dimensions use this same operation.
    """
    original = sparse.csr_matrix(matrix)
    dtype = np.longdouble if refinement_precision == "extended" else float
    forcing = np.asarray(rhs, dtype=dtype)
    with factorize(original, solver=solver) as factor:
        result = factor.solve(forcing, refinement_precision=refinement_precision)
        if refinement_precision == "extended":
            for _ in range(2):
                defect = _accurate_residual(original, forcing, result)
                if np.any(defect):
                    result += factor.solve(defect, refinement_precision="extended")
        return _checked(original, forcing, result, 1e-10, 0.0)
