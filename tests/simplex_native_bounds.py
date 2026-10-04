"""Forward-error bounds for dualized native polynomial accumulation in tests."""

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.reference import orthogonal_polynomial_tabulation, simplex_lagrange_basis


def gamma(operations: int) -> float:
    """Return the standard binary64 accumulation bound gamma_n=n*eps/(1-n*eps)."""
    factor = operations * np.finfo(float).eps
    return float(factor / (1 - factor))


def reference_roundoff_bounds(
    cell: str, degree: int, bary: FloatArray, nodes: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Bound each native value/derivative by its absolute coefficient action.

    The evaluation contracts n polynomial modes; gamma_n scales the sum of
    absolute executed coefficient-times-polynomial contributions. Dualization
    also perturbs the coefficient rows: its normwise forward bound is
    gamma_n*kappa(V)/(1-gamma_n*kappa(V)), with the nodal polynomial matrix V.
    Multiply that bound by ||coefficient_row||_2 ||polynomial_vector||_2.
    This includes structural zero coefficients without assuming a componentwise
    relative inverse error. No field solve or equation tolerance is changed.
    """
    import basix

    basis = simplex_lagrange_basis(cell, degree, nodes=nodes)
    polynomials = orthogonal_polynomial_tabulation(cell, degree, bary[:, 1:], nderiv=2)
    absolute = np.einsum("bk,dqk->dqb", abs(basis.basis_matrix), abs(polynomials))
    accumulation = gamma(basis.basis_matrix.shape[1])
    nodal_polynomials = orthogonal_polynomial_tabulation(cell, degree, nodes[:, 1:], nderiv=0)[0]
    inverse_uncertainty = accumulation * np.linalg.cond(nodal_polynomials)
    assert inverse_uncertainty < 1
    sensitivity = (
        np.linalg.norm(basis.basis_matrix, axis=1)[None, None, :]
        * np.linalg.norm(polynomials, axis=2)[..., None]
    )
    bounds = accumulation * absolute + inverse_uncertainty / (1 - inverse_uncertainty) * sensitivity
    dimension = bary.shape[1] - 1
    first = np.empty((*bounds[0].shape, dimension))
    second = np.empty((*bounds[0].shape, dimension, dimension))
    for axis in range(dimension):
        direction = np.eye(dimension, dtype=int)[axis]
        first[..., axis] = bounds[basix.index(*direction)]
        for other in range(dimension):
            second[..., axis, other] = bounds[
                basix.index(*(direction + np.eye(dimension, dtype=int)[other]))
            ]
    return bounds[0], first, second
