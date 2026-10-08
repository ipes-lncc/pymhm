"""Dimension-dependent polynomial hypotheses for the MHM flux estimator.

The sufficient degree condition follows Theorem 5.2 of
[Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073).
"""

from collections.abc import Iterable

from pymhm.core.validation import positive_int


def validate_estimator_face_partitions(
    continuous: Iterable[bool], subface_counts: Iterable[int]
) -> None:
    """Require independent subface polynomials for the cited estimator theorem.

    Equation (3.1) and Theorem 5.2 use tests supported on each individual
    skeletal subface in dimensions two and three. C0 over several subfaces
    excludes these tests. C0 on one subface coincides with its DG polynomial
    space; canonical RT reconstruction is not restricted by this condition.
    """
    for is_continuous, count in zip(continuous, subface_counts, strict=True):
        if is_continuous and positive_int(count, "subface count") > 1:
            raise ValueError("published estimator requires independent polynomials on each subface")


def minimum_estimator_degree(trace_degree: int, dimension: int) -> int:
    """Return ell+d from Theorem 5.2 for simplicial MHM in dimension two or three.

    This is the sufficient polynomial condition used by the estimator. It is
    not a prerequisite for merely constructing canonical RT moments or for
    solving every primal MHM discretization.
    """
    ell = positive_int(trace_degree, "trace degree", 0)
    if dimension not in (2, 3):
        raise ValueError("estimator dimension must be two or three")
    return ell + dimension


def validate_estimator_spaces(
    local_degree: int, trace_degree: int, reconstruction_degree: int, dimension: int
) -> None:
    """Require k>=ell+d and ell<=m<=k without conflating RT existence with error estimates."""
    minimum = minimum_estimator_degree(trace_degree, dimension)
    k = positive_int(local_degree, "local degree")
    m = positive_int(reconstruction_degree, "RT degree", 0)
    if k < minimum or not trace_degree <= m <= k:
        raise ValueError(f"estimator spaces require k>=ell+{dimension} and ell<=m<=k")
