"""Conforming triangular refinement by longest-edge propagation paths."""

from typing import Any

import numpy as np

from pymhm.mesh import TriangleMesh
from pymhm.refinement import TriangleRefinement


def refine_longest_edge(mesh: TriangleMesh, marked: Any) -> TriangleRefinement:
    """Bisect marked triangles along longest edges with conforming propagation.

    Follow longest edges through adjacent triangles until reaching a boundary
    edge or an edge longest in both incident triangles. Bisect that terminal
    edge and repeat until each originally marked triangle has been replaced.
    Only longest-edge bisections occur; no hanging nodes or green templates are
    introduced. Equal lengths use a consistent global endpoint ordering.

    This is two-dimensional Rivara propagation: in exact arithmetic successive
    longest-edge bisections preserve a lower angle bound of half the original
    minimum angle. Floating-point geometry remains subject to ``TriangleMesh``
    validation. The returned cell and face ancestors refer to the input mesh,
    including faces subdivided more than once during propagation.
    """
    selection = np.asarray(marked)
    if selection.dtype != bool or selection.shape != (len(mesh.cells),):
        raise ValueError("marked must be a boolean array with one entry per triangle")
    vertices = [point.copy() for point in mesh.points]
    triangles = {i: (int(cell[0]), int(cell[1]), int(cell[2])) for i, cell in enumerate(mesh.cells)}
    parents = {i: i for i in triangles}
    neighbors: dict[tuple[int, int], set[int]] = {}
    incident: list[set[int]] = [set() for _ in vertices]
    for face, (a, b) in enumerate(mesh.faces):
        incident[a].add(face)
        incident[b].add(face)
    next_cell = len(triangles)

    def edges(cell: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
        """Return canonical edge keys while preserving the triangle's edge order."""
        return tuple(
            (min(cell[i], cell[(i + 1) % 3]), max(cell[i], cell[(i + 1) % 3])) for i in range(3)
        )

    def longest(cell: int) -> tuple[int, int]:
        """Select a longest edge with a globally consistent exact-length tie break."""
        return max(
            edges(triangles[cell]),
            key=lambda edge: (
                float(np.sum((vertices[edge[1]] - vertices[edge[0]]) ** 2)),
                edge,
            ),
        )

    for index, cell in triangles.items():
        for edge in edges(cell):
            neighbors.setdefault(edge, set()).add(index)

    def bisect(edge: tuple[int, int]) -> None:
        """Split all incident triangles together at one shared terminal midpoint."""
        nonlocal next_cell
        midpoint = len(vertices)
        a, b = edge
        vertices.append(vertices[a] + (vertices[b] - vertices[a]) / 2)
        incident.append(incident[a] & incident[b])
        for index in sorted(neighbors[edge]):
            cell = triangles.pop(index)
            owner = parents.pop(index)
            local_edges = edges(cell)
            position = local_edges.index(edge)
            a, b, c = cell[position], cell[(position + 1) % 3], cell[(position + 2) % 3]
            for old_edge in local_edges:
                neighbors[old_edge].remove(index)
                if not neighbors[old_edge]:
                    del neighbors[old_edge]
            for child in ((a, midpoint, c), (midpoint, b, c)):
                triangles[next_cell] = child
                parents[next_cell] = owner
                for child_edge in edges(child):
                    neighbors.setdefault(child_edge, set()).add(next_cell)
                next_cell += 1

    for selected in np.flatnonzero(selection):
        while int(selected) in triangles:
            current = int(selected)
            while True:
                edge = longest(current)
                adjacent = neighbors[edge] - {current}
                if not adjacent:
                    break
                other = next(iter(adjacent))
                if longest(other) == edge:
                    break
                current = other
            bisect(edge)
    active = sorted(triangles)
    result = TriangleMesh(np.asarray(vertices), np.asarray([triangles[i] for i in active]))
    face_parents = np.full(len(result.faces), -1, dtype=np.int64)
    for face, (a, b) in enumerate(result.faces):
        common = incident[a] & incident[b]
        if common:
            face_parents[face] = next(iter(common))
    return TriangleRefinement(
        result, np.asarray([parents[i] for i in active], dtype=np.int64), face_parents
    )
