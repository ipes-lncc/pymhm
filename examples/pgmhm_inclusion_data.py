"""Square-annulus data for the heterogeneous PGMHM experiment.

The 27-by-27 geometry scales the three-by-three construction of Barrenechea
et al. (2020), Figure 7, keeping radii/period equal to 3/32 and 3/16. These
ratios are consistent with the embedded raster of Fernando et al. (2023),
Figure 9. They are stated explicitly because its Figure 8 period is inconsistent
with that three-by-three construction.
"""

from __future__ import annotations

import numpy as np

from examples.mh_campaign import l_mesh
from examples.solve_unusual_spe10_reference import subdivide_axis
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh

COUNT = 27
PERIOD = 1 / COUNT
INNER_RADIUS = 3 * PERIOD / 32
OUTER_RADIUS = 3 * PERIOD / 16
CONTRAST = 1e5


def coefficient(points: np.ndarray) -> np.ndarray:
    """Return K=1e5 on square annuli and K=1 in their cores and exterior."""
    points = np.asarray(points)
    distance = abs(np.mod(points, PERIOD) - PERIOD / 2).max(axis=-1)
    return np.where((distance > INNER_RADIUS) & (distance < OUTER_RADIUS), CONTRAST, 1.0)


def material_array() -> np.ndarray:
    """Encode all exact interfaces on a uniform 864-by-864 Cartesian material grid."""
    axis = (np.arange(32 * COUNT) + 0.5) / (32 * COUNT)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return coefficient(np.stack((x, y), axis=-1))


def axis(factor: int, *, macro: bool = False) -> np.ndarray:
    """Include every annulus interface, then subdivide each interval isotropically."""
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("factor must be a positive integer")
    centers = (np.arange(COUNT) + 0.5) / COUNT
    breaks = [0.0, 1.0]
    for radius in (-OUTER_RADIUS, -INNER_RADIUS, INNER_RADIUS, OUTER_RADIUS):
        breaks.extend(centers + radius)
    if macro:
        # Both coordinate directions retain the union of the L-mesh grid axes.
        breaks.extend(np.linspace(0, 1, 9))
        breaks.extend(np.linspace(0, 1, 13))
    return subdivide_axis(np.unique(np.asarray(breaks)), factor)


def rectangle_mesh(coordinates: np.ndarray) -> TriangleMesh:
    """Split every material-resolved rectangle southwest to northeast."""
    x, y = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((x.ravel(), y.ravel()))
    n = len(coordinates)
    lower = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)).ravel()
    cells = np.concatenate(
        (
            np.column_stack((lower, lower + 1, lower + n + 1)),
            np.column_stack((lower, lower + n + 1, lower + n)),
        )
    )
    return TriangleMesh(points, cells)


def local_meshes(factor: int) -> tuple[PolygonMesh, tuple[TriangleMesh, ...]]:
    """Partition a common fitted triangular grid into the 32 stated L-shaped macros."""
    macro = l_mesh(4)
    global_mesh = rectangle_mesh(axis(factor, macro=True))
    centers = global_mesh.points[global_mesh.cells].mean(axis=1)
    counts = np.zeros(len(centers), dtype=int)
    local = []
    for cell in macro.cells:
        vertices = macro.points[cell]
        inside = np.zeros(len(centers), dtype=bool)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            if a[1] == b[1]:
                continue
            crossing = ((a[1] > centers[:, 1]) != (b[1] > centers[:, 1])) & (
                centers[:, 0] < a[0] + (centers[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            )
            inside ^= crossing
        counts += inside
        nodes, inverse = np.unique(global_mesh.cells[inside], return_inverse=True)
        local.append(TriangleMesh(global_mesh.points[nodes], inverse.reshape(-1, 3)))
    if not np.all(counts == 1):
        raise ArithmeticError("the fitted local partitions must cover each fine cell exactly once")
    return macro, tuple(local)
