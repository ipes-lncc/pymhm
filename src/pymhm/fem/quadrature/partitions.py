"""Geometric compatibility of independently declared simplicial integration pieces."""

from typing import Any

import numpy as np

from pymhm.core.validation import real_array


def simplex_partition_refines(fine: Any, coarse: Any, *, tolerance: float = 1e-12) -> bool:
    """Check that each fine simplex lies in one coarse simplex of a partition.

    Arrays have axes (piece,vertex,ambient_coordinate); codimension is allowed.
    Vertices suffice because the coarse pieces are convex simplices. This tests
    the common-integration hypothesis, rather than equality of partition counts.
    The nonnegative tolerance applies to barycentric coordinates and geometric
    residual divided by the coarse piece diameter. Partition coverage and
    nonoverlap remain the declaring geometry owner's separate responsibility.
    """
    fine, coarse = real_array(fine, "fine partition"), real_array(coarse, "coarse partition")
    if (
        fine.ndim != 3
        or coarse.ndim != 3
        or fine.shape[1:] != coarse.shape[1:]
        or fine.shape[1] < 2
        or fine.shape[1] > fine.shape[2] + 1
        or not len(fine)
        or not len(coarse)
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("simplex partitions need finite compatible nonempty vertex arrays")
    for vertices in fine:
        contained = False
        for outer in coarse:
            edges = outer[1:] - outer[0]
            if np.linalg.matrix_rank(edges) != len(edges):
                raise ValueError("coarse integration pieces must be nondegenerate simplices")
            coordinates = (vertices - outer[0]) @ np.linalg.pinv(edges)
            residual = vertices - outer[0] - coordinates @ edges
            if (
                np.min(coordinates) >= -tolerance
                and np.max(coordinates.sum(axis=1)) <= 1 + tolerance
                and np.max(abs(residual)) <= tolerance * np.linalg.norm(edges)
            ):
                contained = True
                break
        if not contained:
            return False
    return True
