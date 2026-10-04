"""Oriented triangular meshes and independently enriched polynomial skeletons."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def positive_int(value: Any, name: str, minimum: int = 1) -> int:
    """Validate a discrete parameter without silently truncating real numbers."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


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


@dataclass(frozen=True)
class FaceSpace:
    """Discontinuous Legendre or continuous nodal polynomials on one macroface.

    Breaks parameterize the oriented face on [0, 1]. Each segment has its own
    degree. The default is one constant multiplier on the entire face.
    ``continuous=True`` shares endpoints between segments on this macroface,
    with no continuity imposed across different macrofaces. Degrees must then
    be positive; interior interpolation nodes are equally spaced.
    """

    breaks: tuple[float, ...] = (0.0, 1.0)
    degrees: tuple[int, ...] = (0,)
    continuous: bool = False

    def __post_init__(self) -> None:
        """Reject gaps, reversed intervals and inconsistent polynomial degrees."""
        breaks = tuple(float(x) for x in self.breaks)
        degrees = tuple(positive_int(p, "degree", 0) for p in self.degrees)
        if len(breaks) != len(degrees) + 1 or not degrees:
            raise ValueError("one degree is required per face segment")
        if breaks[0] != 0 or breaks[-1] != 1 or not np.all(np.diff(breaks) > 0):
            raise ValueError("face breaks must increase strictly from 0 to 1")
        if not isinstance(self.continuous, bool) or (self.continuous and min(degrees) < 1):
            raise ValueError("continuous faces require positive degrees and a boolean flag")
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "degrees", degrees)

    @classmethod
    def uniform(
        cls, degree: int = 0, subdivisions: int = 1, *, continuous: bool = False
    ) -> "FaceSpace":
        """Construct an equally partitioned face with the same degree per segment."""
        n = positive_int(subdivisions, "subdivisions")
        return cls(tuple(np.linspace(0, 1, n + 1)), (degree,) * n, continuous)

    @property
    def size(self) -> int:
        """Return the number of scalar face unknowns."""
        return 1 + sum(self.degrees) if self.continuous else sum(p + 1 for p in self.degrees)

    def evaluate(self, parameter: Any) -> FloatArray:
        """Evaluate Basix scalar bases at oriented coordinates in [0, 1].

        Continuous segment nodes are ordered left endpoint, right endpoint,
        then increasing interior nodes. Discontinuous modes use conventional
        unnormalized Legendre polynomials, whose degree-j squared mass on a
        segment of length L is L/(2j+1). Internal breaks belong to the segment
        on their right; a face endpoint belongs to its adjacent segment.
        """
        from pymhm.element_backends import legendre_values, simplex_lagrange_tabulation

        t = np.atleast_1d(np.asarray(parameter, dtype=float))
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face coordinates must be a finite vector in [0, 1]")
        values = np.zeros((len(t), self.size))
        offset = 0
        for i, degree in enumerate(self.degrees):
            lo, hi = self.breaks[i : i + 2]
            mask = (t >= lo) & ((t < hi) | ((i == len(self.degrees) - 1) & (t <= hi)))
            local = (t[mask] - lo) / (hi - lo)
            if self.continuous:
                nodes = np.r_[0.0, 1.0, np.arange(1, degree) / degree]
                basis, _, _ = simplex_lagrange_tabulation(
                    "interval",
                    degree,
                    np.column_stack((1 - local, local)),
                    nodes=np.column_stack((1 - nodes, nodes)),
                    nderiv=0,
                )
                ids = np.r_[i, i + 1, len(self.breaks) + offset + np.arange(degree - 1)]
                values[np.ix_(mask, ids)] = basis
                offset += degree - 1
            else:
                values[mask, offset : offset + degree + 1] = legendre_values(2 * local - 1, degree)
                offset += degree + 1
        return values

    def constant_coefficients(self) -> FloatArray:
        """Represent the constant one exactly in this face's declared basis."""
        if self.continuous:
            return np.ones(self.size)
        coefficients = np.zeros(self.size)
        coefficients[np.r_[0, np.cumsum(np.asarray(self.degrees[:-1]) + 1)].astype(int)] = 1
        return coefficients

    def quadrature(self, order: int = 4) -> tuple[FloatArray, FloatArray]:
        """Return Gauss points and weights, separately on every segment.

        Normalize the reference rule to its exact mass two before mapping
        each subinterval. Extended accumulation prevents a rounded weight-sum
        defect from becoming an artificial volume change in nearly
        incompressible displacement boundary data. No field value is clipped.
        Returned coordinates and weights retain binary64 storage.
        """
        x, w = leggauss(positive_int(order, "quadrature order"))
        w = np.asarray(w, dtype=np.longdouble)
        w *= np.longdouble(2) / np.sum(w)
        points: list[float] = []
        weights: list[float] = []
        for lo, hi in zip(self.breaks[:-1], self.breaks[1:], strict=True):
            points.extend(lo + (x + 1) * (hi - lo) / 2)
            weights.extend(w * (hi - lo) / 2)
        return np.asarray(points), np.asarray(weights, dtype=float)


@dataclass(frozen=True, init=False)
class SkeletonSpace:
    """Independent face partitions and degrees with component-interleaved DOFs."""

    mesh: TriangleMesh
    faces: tuple[FaceSpace, ...]
    components: int = 1
    offsets: IntArray = field(init=False, repr=False)

    def __init__(
        self, mesh: TriangleMesh, faces: tuple[FaceSpace, ...] | None = None, components: int = 1
    ) -> None:
        """Assign unique face numbers and validate the geometry-to-space association."""
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "components", positive_int(components, "components"))
        faces = tuple(FaceSpace() for _ in mesh.faces) if faces is None else tuple(faces)
        if len(faces) != len(self.mesh.faces) or not all(
            isinstance(face, FaceSpace) for face in faces
        ):
            raise ValueError("provide exactly one FaceSpace per mesh face")
        offsets = np.r_[0, np.cumsum([face.size * components for face in faces])].astype(np.int64)
        offsets.setflags(write=False)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "offsets", offsets)

    @property
    def size(self) -> int:
        """Return the total number of trace coefficients."""
        return int(self.offsets[-1])

    def dofs(self, face: int) -> IntArray:
        """Return the global DOFs belonging to a face."""
        positive_int(face, "face", 0)
        if face >= len(self.mesh.faces):
            raise ValueError("face index outside mesh")
        return np.arange(self.offsets[face], self.offsets[face + 1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate trace DOFs in the cell's counterclockwise side order."""
        positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cells):
            raise ValueError("cell index outside mesh")
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])
