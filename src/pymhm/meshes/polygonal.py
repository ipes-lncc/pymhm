"""Conforming polygonal macro meshes with independent triangular local meshes.

Macro cells are simple straight-sided polygons, including nonconvex cells.
Ear triangulation is refined using topological barycentric keys, so internal
edges share nodes without coordinate rounding. The global trace remains on
the original polygon faces and is never replaced by the triangulation edges.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.meshes.triangle import TriangleMesh


class _CellRows(tuple):
    """Immutable ragged incidence supporting both row and (cell, side) indexing."""

    def __getitem__(self, index: Any) -> Any:
        """Preserve the rectangular mesh indexing contract for variable-sized cells."""
        if isinstance(index, tuple):
            return super().__getitem__(index[0])[index[1:]]
        return super().__getitem__(index)


def _cross(a: FloatArray, b: FloatArray) -> Any:
    """Return the scalar two-dimensional cross product, allowing leading axes."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _triangulate(points: FloatArray) -> IntArray:
    """Ear-clip one simple counterclockwise polygon, preserving all boundary nodes."""
    remaining = list(range(len(points)))
    triangles = []
    scale = float(np.max(np.ptp(points, axis=0)))
    tolerance = 64 * np.finfo(float).eps * scale * scale
    while len(remaining) > 3:
        for j, b in enumerate(remaining):
            a, c = remaining[j - 1], remaining[(j + 1) % len(remaining)]
            if _cross(points[b] - points[a], points[c] - points[a]) <= tolerance:
                continue
            other = [v for v in remaining if v not in (a, b, c)]
            vertices = points[[a, b, c]]
            differences = points[other, None, :] - vertices[None, :, :]
            signs = _cross(np.roll(vertices, -1, axis=0) - vertices, differences)
            if np.any(np.all(signs >= -tolerance, axis=1)):
                continue
            triangles.append((a, b, c))
            remaining.pop(j)
            break
        else:
            raise ValueError("polygon must be simple and have resolvable nondegenerate ears")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return np.asarray(triangles, dtype=np.int64)


@dataclass(frozen=True)
class PolygonMesh:
    """Planar polygon macrocells with oriented edges and arbitrary vertex counts.

    ``cells`` lists cyclic vertex indices for each macrocell. Coordinates must
    use common vertex IDs at shared faces; cells are reoriented counterclockwise.
    Consecutive collinear boundary vertices are permitted and remain separate
    macrofaces. Each polygon must be simple, without holes. Adjacent macrofaces
    must match geometrically; hanging vertices must be inserted on both cells.
    ``local_triangulation='boundary'`` supplies distinct adjacent triangles for
    distinct polygon faces. It uses an interior centroid fan when visible from
    every edge, otherwise centroid fans within the ear triangulation. The default
    ``'ears'`` gives the triangulation with no additional interior vertices.
    """

    points: FloatArray
    cells: tuple[IntArray, ...]
    local_triangulation: Literal["ears", "boundary"] = "ears"
    faces: IntArray = field(init=False, repr=False)
    cell_faces: tuple[IntArray, ...] = field(init=False, repr=False)
    signs: tuple[IntArray, ...] = field(init=False, repr=False)
    face_cells: IntArray = field(init=False, repr=False)
    _triangles: tuple[IntArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate polygon topology and construct signed macroface incidence."""
        if self.local_triangulation not in ("ears", "boundary"):
            raise ValueError("local_triangulation must be ears or boundary")
        if np.iscomplexobj(self.points):
            raise ValueError("polygon points must be real")
        points = np.array(self.points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("polygon points must be finite XY pairs")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("polygon points require unique coordinates and shared vertex IDs")
        if not len(self.cells):
            raise ValueError("at least one polygon is required")
        cells, triangles = [], []
        faces: list[tuple[int, int]] = []
        neighbors, cell_faces, signs = [], [], []
        lookup: dict[tuple[int, int], int] = {}
        for cell, raw in enumerate(self.cells):
            indices = np.asarray(raw)
            if (
                indices.ndim != 1
                or len(indices) < 3
                or not np.issubdtype(indices.dtype, np.integer)
            ):
                raise ValueError("each polygon needs at least three integer vertex indices")
            indices = indices.astype(np.int64, copy=True)
            if (
                indices.min() < 0
                or indices.max() >= len(points)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError("polygon indices must be distinct and within the point array")
            xy = points[indices]
            area = float(np.sum(_cross(xy - xy[0], np.roll(xy, -1, axis=0) - xy[0])) / 2)
            if abs(area) <= 64 * np.finfo(float).eps * float(np.prod(np.ptp(xy, axis=0))):
                raise ValueError("polygon has zero or unresolved area")
            if area < 0:
                indices = indices[::-1].copy()
                xy = points[indices]
            # Nonadjacent edges of a simple polygon cannot cross or overlap.
            for i in range(len(xy)):
                a, b = xy[i], xy[(i + 1) % len(xy)]
                for j in range(i + 2, len(xy)):
                    if i == 0 and j == len(xy) - 1:
                        continue
                    c, d = xy[j], xy[(j + 1) % len(xy)]
                    ca, da = _cross(b - a, c - a), _cross(b - a, d - a)
                    ac, bc = _cross(d - c, a - c), _cross(d - c, b - c)
                    if (
                        ca * da <= 0
                        and ac * bc <= 0
                        and np.all(
                            np.maximum(np.minimum(a, b), np.minimum(c, d))
                            <= np.minimum(np.maximum(a, b), np.maximum(c, d))
                        )
                    ):
                        raise ValueError("polygon boundary self-intersects")
            triangles.append(_triangulate(xy))
            local_faces, local_signs = [], []
            for a, b in zip(indices, np.roll(indices, -1), strict=True):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                if key not in lookup:
                    lookup[key] = len(faces)
                    faces.append((int(a), int(b)))
                    neighbors.append([cell, -1])
                    sign = 1
                else:
                    face = lookup[key]
                    if neighbors[face][1] != -1 or faces[face] == (a, b):
                        raise ValueError("nonmanifold macroface or overlapping cell orientation")
                    neighbors[face][1] = cell
                    sign = -1
                local_faces.append(lookup[key])
                local_signs.append(sign)
            cells.append(indices)
            cell_faces.append(np.asarray(local_faces, dtype=np.int64))
            signs.append(np.asarray(local_signs, dtype=np.int64))
        for a, b in faces:
            vector = points[b] - points[a]
            parameter = (points - points[a]) @ vector / (vector @ vector)
            distance = np.abs(_cross(vector, points - points[a])) / np.linalg.norm(vector)
            tolerance = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(points)))
            if np.any(
                (distance < tolerance) & (parameter > tolerance) & (parameter < 1 - tolerance)
            ):
                raise ValueError("hanging macro vertex: split the edge in every adjacent polygon")
        for name, value in (
            ("points", points),
            ("cells", tuple(cells)),
            ("faces", np.asarray(faces, dtype=np.int64)),
            ("cell_faces", _CellRows(cell_faces)),
            ("signs", _CellRows(signs)),
            ("face_cells", np.asarray(neighbors, dtype=np.int64)),
            ("_triangles", tuple(triangles)),
        ):
            if isinstance(value, tuple):
                for array in value:
                    array.setflags(write=False)
            else:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def areas(self) -> FloatArray:
        """Return positive polygon areas from oriented boundary integrals."""
        return np.array(
            [
                np.sum(
                    _cross(
                        self.points[c] - self.points[c[0]],
                        np.roll(self.points[c], -1, axis=0) - self.points[c[0]],
                    )
                )
                / 2
                for c in self.cells
            ]
        )

    @property
    def lengths(self) -> FloatArray:
        """Return physical lengths of macrofaces, independent of local triangulations."""
        return np.linalg.norm(self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]], axis=1)

    @property
    def normals(self) -> FloatArray:
        """Return unit normals pointing out of each face's first adjacent macrocell."""
        tangent = self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]]
        return np.column_stack((tangent[:, 1], -tangent[:, 0])) / self.lengths[:, None]

    @property
    def boundary_faces(self) -> IntArray:
        """Return macrofaces with a single adjacent polygon."""
        return np.flatnonzero(self.face_cells[:, 1] < 0)

    def submesh(self, cell: int, subdivisions: int) -> TriangleMesh:
        """Refine the ear triangulation conformingly with exact topological node keys."""
        cell = positive_int(cell, "cell", 0)
        r = positive_int(subdivisions, "subdivisions")
        if cell >= len(self.cells):
            raise ValueError("cell index outside polygon mesh")
        vertices = self.points[self.cells[cell]]
        base = self._triangles[cell]
        if self.local_triangulation == "boundary":
            corners = vertices[base]
            areas = _cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) / 2
            center = areas @ corners.mean(axis=1) / areas.sum()
            n = len(vertices)
            tolerance = 64 * np.finfo(float).eps * np.max(np.ptp(vertices, axis=0)) ** 2
            if np.all(
                _cross(np.roll(vertices, -1, axis=0) - vertices, center - vertices) > tolerance
            ):
                vertices = np.vstack((vertices, center))
                base = np.column_stack((np.arange(n), np.roll(np.arange(n), -1), np.full(n, n)))
            else:
                vertices = np.vstack((vertices, corners.mean(axis=1)))
                base = np.array(
                    [
                        (int(a), int(b), n + cell)
                        for cell, tri in enumerate(base)
                        for a, b in zip(tri, np.roll(tri, -1), strict=True)
                    ]
                )
        lookup: dict[tuple[tuple[int, int], ...], int] = {}
        points: list[FloatArray] = []
        triangles = []
        for triangle in base:
            local = {}
            for i in range(r + 1):
                for j in range(r + 1 - i):
                    weights = (r - i - j, i, j)
                    key = tuple(
                        sorted((int(v), w) for v, w in zip(triangle, weights, strict=True) if w)
                    )
                    if key not in lookup:
                        lookup[key] = len(points)
                        points.append(np.asarray(weights) @ vertices[triangle] / r)
                    local[i, j] = lookup[key]
            for i in range(r):
                for j in range(r - i):
                    triangles.append((local[i, j], local[i + 1, j], local[i, j + 1]))
                    if i + j < r - 1:
                        triangles.append((local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]))
        return TriangleMesh(np.asarray(points), np.asarray(triangles, dtype=np.int64))
