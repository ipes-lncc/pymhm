"""Uniform rectangular topology, geometry and exact local subdivision numbering."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int


@dataclass(frozen=True)
class CartesianMacroMesh:
    """Uniform rectangular mesh with counterclockwise cells and oriented faces.

    ``bounds=(xmin,xmax,ymin,ymax)`` and ``nx,ny`` describe the physical domain.
    The first adjacent cell defines each face normal. A missing ``ny`` uses
    ``nx``. Arrays are immutable; cell and point numbering run fastest in x.
    The geometric face interface is compatible with ``SkeletonSpace``.
    """

    nx: int = 1
    ny: int | None = None
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    points: FloatArray = field(init=False, repr=False)
    cells: IntArray = field(init=False, repr=False)
    faces: IntArray = field(init=False, repr=False)
    cell_faces: IntArray = field(init=False, repr=False)
    signs: IntArray = field(init=False, repr=False)
    face_cells: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the rectangle and build its signed cell-face incidence."""
        nx = positive_int(self.nx, "nx")
        ny = nx if self.ny is None else positive_int(self.ny, "ny")
        if np.iscomplexobj(self.bounds):
            raise ValueError("bounds must be real")
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("bounds must contain four finite coordinates")
        x0, x1, y0, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bounds must have positive width and height")
        x, y = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
        points = np.column_stack((np.tile(x, ny + 1), np.repeat(y, nx + 1)))
        if len(np.unique(points[:, 0])) != nx + 1 or len(np.unique(points[:, 1])) != ny + 1:
            raise ValueError("cell spacing is not resolvable at the coordinate scale")
        i, j = np.meshgrid(np.arange(nx), np.arange(ny))
        a = (j * (nx + 1) + i).ravel()
        cells = np.column_stack((a, a + 1, a + nx + 2, a + nx + 1))
        # Face IDs follow first encounter in counterclockwise row-major cells.
        # Closed formulas preserve that public numbering without Python dictionaries.
        horizontal = np.empty((ny + 1, nx), dtype=np.int64)
        vertical = np.empty((ny, nx + 1), dtype=np.int64)
        columns = np.arange(nx)
        horizontal[0] = np.where(columns == 0, 0, 3 * columns + 1)
        horizontal[1] = np.where(columns == 0, 2, 3 * columns + 3)
        vertical[0, 0] = 3
        vertical[0, 1:] = np.where(columns == 0, 1, 3 * columns + 2)
        starts = 3 * nx + 1 + np.arange(ny - 1) * (2 * nx + 1)
        horizontal[2:] = starts[:, None] + np.where(columns == 0, 1, 2 * columns + 2)
        vertical[1:, 0] = starts + 2
        vertical[1:, 1:] = starts[:, None] + np.where(columns == 0, 0, 2 * columns + 1)
        count = (ny + 1) * nx + (nx + 1) * ny
        faces = np.empty((count, 2), dtype=np.int64)
        neighbors = np.full((count, 2), -1, dtype=np.int64)
        grid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        hstart, hend = grid[:, :-1].copy(), grid[:, 1:].copy()
        hstart[1:], hend[1:] = hend[1:].copy(), hstart[1:].copy()
        faces[horizontal.ravel()] = np.column_stack((hstart.ravel(), hend.ravel()))
        vstart, vend = grid[:-1].copy(), grid[1:].copy()
        vstart[:, 0], vend[:, 0] = vend[:, 0].copy(), vstart[:, 0].copy()
        faces[vertical.ravel()] = np.column_stack((vstart.ravel(), vend.ravel()))
        ids = np.arange(nx * ny).reshape(ny, nx)
        neighbors[horizontal[0], 0] = ids[0]
        neighbors[horizontal[1:].ravel(), 0] = ids.ravel()
        neighbors[horizontal[1:-1].ravel(), 1] = ids[1:].ravel()
        neighbors[vertical[:, 0], 0] = ids[:, 0]
        neighbors[vertical[:, 1:].ravel(), 0] = ids.ravel()
        neighbors[vertical[:, 1:-1].ravel(), 1] = ids[:, 1:].ravel()
        cell_faces = np.column_stack(
            (
                horizontal[:-1].ravel(),
                vertical[:, 1:].ravel(),
                horizontal[1:].ravel(),
                vertical[:, :-1].ravel(),
            )
        )
        signs = np.column_stack(
            (
                np.where(j.ravel() == 0, 1, -1),
                np.ones(nx * ny, dtype=np.int64),
                np.ones(nx * ny, dtype=np.int64),
                np.where(i.ravel() == 0, 1, -1),
            )
        )
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "bounds", tuple(float(v) for v in bounds))
        for name, value in (
            ("points", points),
            ("cells", cells),
            ("faces", np.asarray(faces, dtype=np.int64)),
            ("cell_faces", cell_faces),
            ("signs", signs),
            ("face_cells", np.asarray(neighbors, dtype=np.int64)),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def spacing(self) -> FloatArray:
        """Return the physical widths of one rectangular cell."""
        x0, x1, y0, y1 = self.bounds
        return np.array([(x1 - x0) / self.nx, (y1 - y0) / cast(int, self.ny)])

    @property
    def areas(self) -> FloatArray:
        """Return one positive physical area per cell."""
        return np.full(len(self.cells), np.prod(self.spacing))

    @property
    def lengths(self) -> FloatArray:
        """Return the physical lengths of the oriented mesh faces."""
        return np.linalg.norm(self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]], axis=1)

    @property
    def normals(self) -> FloatArray:
        """Return unit normals pointing out of each face's first adjacent cell."""
        tangent = self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]]
        return np.column_stack((tangent[:, 1], -tangent[:, 0])) / self.lengths[:, None]

    @property
    def boundary_faces(self) -> IntArray:
        """Return faces with a single neighboring rectangle."""
        return np.flatnonzero(self.face_cells[:, 1] == -1)

    def submesh(self, cell: int, subdivisions: int | tuple[int, int]) -> "CartesianMacroMesh":
        """Refine one rectangle independently in its two coordinate directions."""
        positive_int(cell, "cell", 0)
        if cell >= len(self.cells):
            raise ValueError("cell index outside mesh")
        nx, ny = _refinement(subdivisions)
        lower, upper = self.points[self.cells[cell, [0, 2]]]
        return CartesianMacroMesh(nx, ny, (lower[0], upper[0], lower[1], upper[1]))


def cartesian_refinement(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a scalar or rectangular pair of local subdivision counts."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("local_refinement requires two integers")
        return positive_int(value[0], "local_refinement"), positive_int(
            value[1], "local_refinement"
        )
    n = positive_int(value, "local_refinement")
    return n, n


_refinement = cartesian_refinement
