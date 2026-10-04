"""Oriented triangular geometry and signed cell-to-face incidence."""

from dataclasses import dataclass, field

import numpy as np

from pymhm.core.validation import FloatArray as FloatArray
from pymhm.core.validation import IntArray as IntArray
from pymhm.core.validation import positive_int as positive_int


@dataclass(frozen=True)
class TriangleMesh:
    """Conforming, nondegenerate planar triangles with consistent face normals.

    Points have shape ``(n, 2)`` and cells ``(m, 3)``. Clockwise input cells are
    reoriented. Every face normal points out of its first adjacent cell; boundary
    normals therefore point out of the domain. Geometry arrays are copied.
    """

    points: FloatArray
    cells: IntArray
    faces: IntArray = field(init=False, repr=False)
    cell_faces: IntArray = field(init=False, repr=False)
    signs: IntArray = field(init=False, repr=False)
    face_cells: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate geometry and construct the signed cell-to-face incidence."""
        points = np.array(self.points, dtype=float, copy=True)
        raw = np.asarray(self.cells)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("points must be finite with shape (n, 2)")
        if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
            raise ValueError("cells must have nonempty shape (m, 3)")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("cell indices must be integers")
        cells = raw.astype(np.int64, copy=True)
        if cells.min() < 0 or cells.max() >= len(points):
            raise ValueError("cell index outside point array")
        vertices = points[cells]
        a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if np.any(
            np.abs(determinant)
            <= np.finfo(float).eps * np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        ):
            raise ValueError("degenerate triangle")
        reverse = determinant < 0
        cells[reverse] = cells[reverse][:, [0, 2, 1]]
        if len(np.unique(np.sort(cells, axis=1), axis=0)) != len(cells):
            raise ValueError("duplicate triangle")
        lookup: dict[tuple[int, int], int] = {}
        faces: list[tuple[int, int]] = []
        neighbors: list[list[int]] = []
        cell_faces = np.empty_like(cells)
        signs = np.empty_like(cells)
        for cell, nodes in enumerate(cells):
            for side in range(3):
                i, j = int(nodes[side]), int(nodes[(side + 1) % 3])
                key = min(i, j), max(i, j)
                if key not in lookup:
                    lookup[key] = len(faces)
                    faces.append((i, j))
                    neighbors.append([cell, -1])
                    sign = 1
                else:
                    face = lookup[key]
                    if neighbors[face][1] != -1 or faces[face] == (i, j):
                        raise ValueError("nonmanifold face or overlapping cell orientation")
                    neighbors[face][1] = cell
                    sign = -1
                cell_faces[cell, side] = lookup[key]
                signs[cell, side] = sign
        for name, array in (
            ("points", points),
            ("cells", cells),
            ("faces", np.array(faces, dtype=np.int64)),
            ("cell_faces", cell_faces),
            ("signs", signs),
            ("face_cells", np.array(neighbors, dtype=np.int64)),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def unit_square(cls, nx: int = 1, ny: int | None = None) -> "TriangleMesh":
        """Split a Cartesian unit-square grid along each southwest diagonal."""
        nx = positive_int(nx, "nx")
        ny = nx if ny is None else positive_int(ny, "ny")
        points = np.array([(x / nx, y / ny) for y in range(ny + 1) for x in range(nx + 1)])
        cells: list[tuple[int, int, int]] = []
        for y in range(ny):
            for x in range(nx):
                a = y * (nx + 1) + x
                cells.extend(((a, a + 1, a + nx + 2), (a, a + nx + 2, a + nx + 1)))
        return cls(points, np.array(cells, dtype=np.int64))

    @property
    def boundary_faces(self) -> IntArray:
        """Return indices of faces with exactly one adjacent cell."""
        return np.flatnonzero(self.face_cells[:, 1] == -1)

    @property
    def normals(self) -> FloatArray:
        """Return unit normals in the globally fixed face orientations."""
        tangent = self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]]
        return np.column_stack((tangent[:, 1], -tangent[:, 0])) / self.lengths[:, None]

    @property
    def lengths(self) -> FloatArray:
        """Return face lengths."""
        return np.linalg.norm(self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]], axis=1)

    @property
    def areas(self) -> FloatArray:
        """Return positive triangle areas."""
        vertices = self.points[self.cells]
        a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2

    def submesh(self, cell: int, subdivisions: int) -> "TriangleMesh":
        """Uniformly subdivide one macrotriangle into ``subdivisions**2`` cells."""
        n = positive_int(subdivisions, "subdivisions")
        positive_int(cell, "cell", 0)
        if cell >= len(self.cells):
            raise ValueError("cell index outside mesh")
        vertices = self.points[self.cells[cell]]
        indices = {
            (i, j): k
            for k, (i, j) in enumerate((i, j) for i in range(n + 1) for j in range(n + 1 - i))
        }
        points = np.array(
            [
                vertices[0]
                + i / n * (vertices[1] - vertices[0])
                + j / n * (vertices[2] - vertices[0])
                for i, j in indices
            ]
        )
        cells: list[tuple[int, int, int]] = []
        for i in range(n):
            for j in range(n - i):
                cells.append((indices[i, j], indices[i + 1, j], indices[i, j + 1]))
                if i + j < n - 1:
                    cells.append((indices[i + 1, j], indices[i + 1, j + 1], indices[i, j + 1]))
        return TriangleMesh(points, np.array(cells, dtype=np.int64))
