"""Affine crisscross partitions inside triangular macroelements."""

import numpy as np

from pymhm.mesh import TriangleMesh, positive_int
from pymhm.refinement import validate_submesh


def crisscross_submesh(
    mesh: TriangleMesh,
    cell: int,
    subdivisions: int,
    *,
    diagonal_side: int | None = None,
) -> TriangleMesh:
    """Partition one macrotriangle into 2*subdivisions**2 triangles.

    The chosen macro side is the diagonal of an affine Cartesian square.
    Its opposite vertex and two endpoints define the reference right triangle.
    Interior little squares have both diagonals; squares intersected by the
    macro diagonal retain two triangles. The diagonal therefore has 2*r
    fine edges, and each other macro side has r. By default the longest
    physical macro side is chosen, with the first side resolving equal lengths.

    This is the geometry depicted in Barros (2022), Figure 4, when the macro
    triangles are halves of Cartesian squares. On general triangles it is its
    explicitly specified affine image. The parameter is a subdivision count,
    not the maximum physical diameter of the macrotriangle.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("crisscross partitions require a TriangleMesh")
    index = positive_int(cell, "cell", 0)
    if index >= len(mesh.cells):
        raise ValueError("cell index outside mesh")
    count = positive_int(subdivisions, "subdivisions")
    vertices = mesh.points[mesh.cells[index]]
    side = (
        int(np.argmax(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)))
        if diagonal_side is None
        else positive_int(diagonal_side, "diagonal_side", 0)
    )
    if side > 2:
        raise ValueError("diagonal_side must identify one of three macro sides")
    origin = vertices[(side + 2) % 3]
    transform = np.array([vertices[side] - origin, vertices[(side + 1) % 3] - origin])
    points: list[tuple[int, int]] = []
    lookup: dict[tuple[int, int], int] = {}
    cells: list[tuple[int, int, int]] = []

    def node(key: tuple[int, int]) -> int:
        """Create each integer half-grid vertex once, without geometric merging."""
        if key not in lookup:
            lookup[key] = len(points)
            points.append(key)
        return lookup[key]

    for i in range(count):
        for j in range(count - i):
            a, b = node((2 * i, 2 * j)), node((2 * i + 2, 2 * j))
            d, centre = node((2 * i, 2 * j + 2)), node((2 * i + 1, 2 * j + 1))
            cells.extend(((a, b, centre), (a, centre, d)))
            if i + j < count - 1:
                c = node((2 * i + 2, 2 * j + 2))
                cells.extend(((b, c, centre), (c, d, centre)))
    fine = TriangleMesh(
        origin + np.asarray(points, dtype=float) @ transform / (2 * count),
        np.asarray(cells, dtype=np.int64),
    )
    validate_submesh(mesh, index, fine)
    return fine
