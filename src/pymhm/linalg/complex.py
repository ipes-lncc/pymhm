"""Exact real coordinates for user-defined complex coefficient equations.

The scalar ordering is ``(Re z0, Im z0, Re z1, Im z1, ...)``. No physical
operator, conjugation convention, damping or boundary condition is inferred.
"""

from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray


def realify_vector(value: Any) -> FloatArray:
    """Interleave a finite complex vector's real and imaginary coordinates.

    Real input has zero imaginary coordinates. The input must be rank one;
    the returned real array owns its data and preserves floating precision.
    This map represents coefficients rather than changing the definition of
    a complex variational form.
    """
    data = np.asarray(value)
    if data.ndim != 1 or data.dtype.kind not in "biufc" or not np.isfinite(data).all():
        raise ValueError("complex coefficients must be a finite numeric vector")
    paired = np.column_stack((data.real, data.imag))
    return paired.ravel() if paired.dtype.kind == "f" else paired.astype(float).ravel()


def complexify_vector(value: Any) -> Any:
    """Decode an even-length finite real vector in interleaved Re/Im order."""
    data = np.asarray(value)
    if (
        data.ndim != 1
        or data.size % 2
        or data.dtype.kind not in "biuf"
        or not np.isfinite(data).all()
    ):
        raise ValueError("realified coefficients must be a finite even-length real vector")
    paired = data.reshape(-1, 2)
    return paired[:, 0] + 1j * paired[:, 1]


def realify_operator(value: Any) -> Any:
    """Represent complex multiplication by blocks ``[[Re,-Im],[Im,Re]]``.

    Rectangular dense or sparse operators are accepted. The returned CSC
    matrix has twice the row/column count and acts on :func:`realify_vector`
    coordinates. For a sesquilinear form the caller first assembles its complex
    matrix with the desired conjugated test convention; this function embeds
    that matrix exactly and does not transpose or conjugate it.
    """
    if not sparse.issparse(value):
        data = np.asarray(value)
        if data.ndim != 2 or data.dtype.kind not in "biufc":
            raise ValueError("complex operator must be a numeric matrix")
        value = data
    matrix = sparse.csc_matrix(value)
    if not np.isfinite(matrix.data).all():
        raise ValueError("complex operator must have finite entries")
    return (
        sparse.kron(matrix.real, sparse.eye(2))
        + sparse.kron(matrix.imag, [[0.0, -1.0], [1.0, 0.0]])
    ).tocsc()
