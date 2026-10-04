"""Conforming tetrahedral edge-star bisection with exact topological ancestry."""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from pymhm.core.validation import IntArray
from pymhm.meshes.tetrahedron import TetraMesh


@dataclass(frozen=True)
class TetraRefinement:
    """Refined tetrahedra and their original cell/face parents (-1 for new faces)."""

    mesh: TetraMesh
    cell_parents: IntArray
    face_parents: IntArray


def refine_tetrahedra(mesh: TetraMesh, marked: Any) -> TetraRefinement:
    """Bisect the longest edge of each marked tetrahedron and its entire edge star.

    All cells incident on a selected edge share one midpoint and are split
    together. Original face incidence supplies exact ancestry without coordinate
    rounding. The result is conforming and volume preserving. This local policy
    does not assert the finite-similarity-class theorem of a newest-vertex scheme;
    campaigns should monitor shape quality independently.
    """
    selection = np.asarray(marked)
    if selection.dtype != bool or selection.shape != (len(mesh.cells),):
        raise ValueError("marked must be boolean with one entry per tetrahedron")
    vertices = [point.copy() for point in mesh.points]
    incident: list[set[int]] = [set() for _ in vertices]
    for face, nodes in enumerate(mesh.faces):
        for node in nodes:
            incident[node].add(face)
    chosen: set[tuple[int, int]] = set()
    for cell in mesh.cells[selection]:
        edges = [(min(int(a), int(b)), max(int(a), int(b))) for a, b in combinations(cell, 2)]
        chosen.add(
            max(
                edges,
                key=lambda e: (float(np.sum((mesh.points[e[1]] - mesh.points[e[0]]) ** 2)), e),
            )
        )
    ordered = sorted(
        chosen, key=lambda e: (-float(np.sum((mesh.points[e[1]] - mesh.points[e[0]]) ** 2)), e)
    )
    cells = [tuple(int(v) for v in cell) for cell in mesh.cells]
    parents = list(range(len(cells)))
    for a, b in ordered:
        midpoint = len(vertices)
        vertices.append(vertices[a] + (vertices[b] - vertices[a]) / 2)
        incident.append(incident[a] & incident[b])
        refined: list[tuple[int, ...]] = []
        ancestry: list[int] = []
        for cell, parent in zip(cells, parents, strict=True):
            if a in cell and b in cell:
                refined.extend(
                    (
                        tuple(midpoint if v == a else v for v in cell),
                        tuple(midpoint if v == b else v for v in cell),
                    )
                )
                ancestry.extend((parent, parent))
            else:
                refined.append(cell)
                ancestry.append(parent)
        cells, parents = refined, ancestry
    result = TetraMesh(np.asarray(vertices), np.asarray(cells, dtype=np.int64))
    face_parents = np.full(len(result.faces), -1, dtype=np.int64)
    for face, nodes in enumerate(result.faces):
        common = set.intersection(*(incident[v] for v in nodes))
        if common:
            face_parents[face] = next(iter(common))
    return TetraRefinement(result, np.asarray(parents, dtype=np.int64), face_parents)
