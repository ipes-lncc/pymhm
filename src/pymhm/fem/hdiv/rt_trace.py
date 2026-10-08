"""Physical oriented RT normal-moment pairings on planar macrofaces."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray
from pymhm.fem.hdiv.rt import rt_degree
from pymhm.fem.reference import legendre_values
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def rt_trace_map(
    coarse: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace, degree: int
) -> FloatArray:
    """Integrate canonical RT normal moments of the oriented macroface flux density."""
    m = rt_degree(degree)
    count = m + 1
    result = np.zeros((count * len(fine.boundary_faces), len(skeleton.cell_dofs(cell))))
    x, w = leggauss(m + 3)
    parameter = (x + 1) / 2
    basis = legendre_values(x, m)
    offset = 0
    for side, face in enumerate(coarse.cell_faces[cell]):
        space = skeleton.faces[face]
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        for index, edge in enumerate(fine.boundary_faces):
            vertices = fine.points[fine.faces[edge]]
            t = (vertices - start) @ tangent / (tangent @ tangent)
            if (
                not np.allclose(vertices, start + t[:, None] * tangent, atol=1e-12, rtol=0)
                or min(t) < -1e-12
                or max(t) > 1 + 1e-12
            ):
                continue
            evaluation = np.clip(t[0] + parameter * (t[1] - t[0]), 0, 1)
            result[count * index : count * (index + 1), offset : offset + space.size] = (
                fine.lengths[edge]
                * coarse.signs[cell, side]
                * (basis.T @ (w[:, None] / 2 * space.evaluate(evaluation)))
            )
        offset += space.size
    return result
