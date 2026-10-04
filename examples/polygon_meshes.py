"""Reproducible polygon partitions for the generalized RAD literature case."""

import numpy as np
from scipy.spatial import Voronoi

from pymhm import TriangleMesh
from pymhm.meshes.polygonal import PolygonMesh


def rhombus_partition(n: int) -> PolygonMesh:
    """Clip an exact integer diamond lattice to the unit square, preserving shared nodes."""
    points, cells, lookup = [], [], {}
    for i in range(2 * n):
        for j in range(-n, n):
            polygon = np.array(
                [
                    (i + j, i - j),
                    (i + j + 1, i - j + 1),
                    (i + j + 2, i - j),
                    (i + j + 1, i - j - 1),
                ],
                dtype=float,
            )
            for axis, boundary, sign in ((0, 0, 1), (0, 2 * n, -1), (1, 0, 1), (1, 2 * n, -1)):
                clipped = []
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                    da, db = sign * (a[axis] - boundary), sign * (b[axis] - boundary)
                    if da >= 0:
                        clipped.append(a)
                    if (da < 0 < db) or (db < 0 < da):
                        clipped.append(a + (b - a) * da / (da - db))
                polygon = np.array(clipped).reshape(-1, 2)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            assert np.array_equal(polygon, np.rint(polygon))
            ids = []
            for point in polygon.astype(int):
                key = tuple(point)
                if key not in lookup:
                    lookup[key] = len(points)
                    points.append(point / (2 * n))
                ids.append(lookup[key])
            cells.append(np.array(ids))
    return PolygonMesh(np.asarray(points), tuple(cells), local_triangulation="boundary")


def polygon_partition(n: int, family: str) -> PolygonMesh:
    """Construct square, nonconvex L/square, or clipped hexagonal macro partitions.

    Hexagons are Voronoi cells of staggered seeds. Reflected seeds impose the
    four domain boundaries without independently clipping shared macrofaces.
    Boundary cells need not be regular hexagons.
    """
    if family == "triangle":
        mesh = TriangleMesh.unit_square(n)
        return PolygonMesh(mesh.points, tuple(mesh.cells), local_triangulation="boundary")
    if family == "rhombus":
        return rhombus_partition(n)
    if family == "hexagon":
        rows = int(np.ceil(2 * n / np.sqrt(3)))
        seeds = np.array(
            [
                ((i + 0.5 + 0.25 * (-1) ** j) / n, (j + 0.5) / rows)
                for j in range(rows)
                for i in range(n)
            ]
        )
        copies = [seeds]
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                if x == y == 0:
                    continue
                shifted = seeds.copy()
                if x:
                    shifted[:, 0] = -seeds[:, 0] + (2 if x == 1 else 0)
                if y:
                    shifted[:, 1] = -seeds[:, 1] + (2 if y == 1 else 0)
                copies.append(shifted)
        voronoi = Voronoi(np.vstack(copies))
        regions = [np.asarray(voronoi.regions[r]) for r in voronoi.point_region[: len(seeds)]]
        used = np.unique(np.concatenate(regions))
        if np.any(used < 0):
            raise RuntimeError("reflected Voronoi seeds must give bounded original cells")
        remap = np.full(len(voronoi.vertices), -1)
        remap[used] = np.arange(len(used))
        coordinates = np.clip(voronoi.vertices[used], 0, 1)
        # Reflected seeds prescribe exact coordinate planes. Restore those
        # planes after the floating-point Voronoi construction.
        for boundary in (0.0, 1.0):
            coordinates[np.abs(coordinates - boundary) <= 64 * np.finfo(float).eps] = boundary
        return PolygonMesh(
            coordinates, tuple(remap[c] for c in regions), local_triangulation="boundary"
        )
    points, cells, lookup = [], [], {}
    square = ((0, 0), (2, 0), (2, 2), (0, 2))
    elbow = ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1))
    corner = ((1, 1), (2, 1), (2, 2), (1, 2))
    shapes = (square,) if family == "square" else (elbow, corner)
    if family not in ("square", "L"):
        raise ValueError("family must be triangle, square, rhombus, L or hexagon")
    for j in range(n):
        for i in range(n):
            for shape in shapes:
                indices = []
                for a, b in shape:
                    key = (2 * i + a, 2 * j + b)
                    if key not in lookup:
                        lookup[key] = len(points)
                        points.append(np.array(key) / (2 * n))
                    indices.append(lookup[key])
                cells.append(np.asarray(indices))
    return PolygonMesh(np.asarray(points), tuple(cells), local_triangulation="boundary")
