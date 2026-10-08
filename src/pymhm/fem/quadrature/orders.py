"""Explicit degree-dependent Gaussian point-count policies for nodal forms."""

from pymhm.core.validation import positive_int


def nodal_quadrature_order(degree: int, requested: int, *, increment: int = 2) -> int:
    """Return max(requested,degree+increment) as a declared Gaussian point count.

    This policy is a count in each tensor/Duffy coordinate, not a polynomial
    exactness degree or a certificate for nonpolynomial coefficients. The caller
    declares the nonnegative increment and independently verifies quadrature
    adequacy for its form, material, source and error norm.
    """
    return max(
        positive_int(requested, "quadrature_order"),
        positive_int(degree, "degree") + positive_int(increment, "increment", 0),
    )
