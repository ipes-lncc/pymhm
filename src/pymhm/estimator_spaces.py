"""Dimension-dependent polynomial hypotheses for the L09 MHM flux estimator."""

from pymhm.mesh import positive_int


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
