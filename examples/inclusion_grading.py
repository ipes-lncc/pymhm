"""Declared material-interface grading for the square-annulus CG2 reference."""

from __future__ import annotations

import numpy as np

from examples.pgmhm_inclusion_data import axis
from examples.solve_unusual_spe10_reference import subdivide_axis


def graded_axis(level: int) -> np.ndarray:
    """Refine symmetrically toward exact material interfaces, then bisect uniformly.

    The first level divides every material interval in the proportions
    1:2:4:8:4:2:1. Subsequent levels bisect every existing interval in both
    directions, retaining the diagonal triangulation as an actual nested
    finite-element sequence. The original uniform sequence is separate.
    """
    if not isinstance(level, int) or isinstance(level, bool) or level < 1:
        raise ValueError("grading level must be a positive integer")
    base = axis(1)
    weights = np.array([1, 2, 4, 8, 4, 2, 1], dtype=float)
    fraction = np.r_[0.0, weights.cumsum() / weights.sum()]
    intervals = base[:-1, None] + np.diff(base)[:, None] * fraction
    first = np.r_[intervals[:, :-1].ravel(), base[-1]]
    return subdivide_axis(first, 2 ** (level - 1))
