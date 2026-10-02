"""Disconnected polynomial display grids on physical horizontal tetrahedral sections."""

from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from pymhm import TriangleMesh


def section_grid(mesh: Any, height: float = 0.37, refinement: int = 4) -> dict[str, np.ndarray]:
    """Triangulate each nondegenerate section separately, preserving fine-cell ownership.

    Returned points are physical 3D coordinates; barycentric coordinates refer
    to the owning tetrahedron. Coincident points are not merged across cells.
    Boundary segments describe the original section polygons, independently
    of the display refinement. Empty and point/line intersections are omitted.
    """
    display = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]]).submesh(0, refinement)
    rule = np.column_stack((1 - display.points.sum(axis=1), display.points))
    points, triangles, parents, barycentric, segments = [], [], [], [], []
    offset = 0
    for cell, vertices in enumerate(mesh.points[mesh.cells]):
        cut = [p for p in vertices if p[2] == height]
        for a, b in combinations(vertices, 2):
            if (a[2] - height) * (b[2] - height) < 0:
                cut.append(a + (height - a[2]) / (b[2] - a[2]) * (b - a))
        if len(cut) < 3:
            continue
        polygon = np.unique(cut, axis=0)
        if len(polygon) < 3 or np.linalg.matrix_rank(polygon[:, :2] - polygon[0, :2]) < 2:
            continue
        polygon = polygon[ConvexHull(polygon[:, :2]).vertices]
        segments.extend(zip(polygon[:, :2], np.roll(polygon[:, :2], -1, axis=0), strict=True))
        inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
        for i in range(1, len(polygon) - 1):
            samples = rule @ polygon[[0, i, i + 1]]
            xi = (samples - vertices[0]) @ inverse.T
            points.append(samples)
            barycentric.append(np.column_stack((1 - xi.sum(axis=1), xi)))
            parents.extend([cell] * len(samples))
            triangles.append(display.cells + offset)
            offset += len(samples)
    return dict(
        points=np.concatenate(points) if points else np.empty((0, 3)),
        cells=np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=int),
        parents=np.asarray(parents, dtype=int),
        barycentric=np.concatenate(barycentric) if barycentric else np.empty((0, 4)),
        segments=np.asarray(segments).reshape(-1, 2, 2),
    )
