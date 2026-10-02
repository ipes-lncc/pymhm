"""Conforming local closure for adaptively segmented flow skeletons."""

import numpy as np

from pymhm.longest_edge import refine_longest_edge
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.refinement import refine_triangles


def refine_flow_local_meshes(
    skeleton: SkeletonSpace,
    local_meshes: tuple[TriangleMesh, ...],
    marked_local: np.ndarray,
    maximum_cells: int,
    *,
    fine_marked: tuple[np.ndarray, ...] | None = None,
) -> tuple[TriangleMesh, ...] | None:
    """Resolve every new trace vertex while refining only the required local boundary cells.

    Local-error dominance red-refines all triangles of that local mesh once,
    unless ``fine_marked`` supplies selected cells for longest-edge refinement.
    Subsequent longest-edge propagation contains every skeletal breakpoint,
    preserving conformity and avoiding uniform closure of unrelated interiors.
    Return ``None`` before a local cell cap is exceeded. The input meshes are
    never mutated, so the caller can retain its last solved configuration.
    """
    result = []
    macro = skeleton.mesh
    for cell, original in enumerate(local_meshes):
        fine = original
        if marked_local[cell]:
            fine = (
                refine_triangles(original, np.ones(len(original.cells), dtype=bool)).mesh
                if fine_marked is None
                else refine_longest_edge(original, fine_marked[cell]).mesh
            )
        while len(fine.cells) <= maximum_cells:
            selected = np.zeros(len(fine.cells), dtype=bool)
            boundary = fine.faces[fine.boundary_faces]
            endpoints = fine.points[boundary]
            for face in macro.cell_faces[cell]:
                a, b = macro.points[macro.faces[face]]
                tangent = b - a
                square = float(tangent @ tangent)
                parameter = (endpoints - a) @ tangent / square
                distance = np.linalg.norm(endpoints - a - parameter[:, :, None] * tangent, axis=2)
                tolerance = (
                    256
                    * np.finfo(float).eps
                    * max(np.sqrt(square), float(np.max(abs(macro.points))))
                )
                on_face = np.all(distance <= tolerance, axis=1)
                edges = np.flatnonzero(on_face)
                t = parameter[edges]
                for point in skeleton.faces[face].breaks[1:-1]:
                    if np.any(abs(t - point) <= tolerance / np.sqrt(square)):
                        continue
                    contains = (t.min(axis=1) < point) & (point < t.max(axis=1))
                    owners = fine.face_cells[fine.boundary_faces[edges[contains]], 0]
                    selected[owners] = True
            if not np.any(selected):
                result.append(fine)
                break
            fine = refine_longest_edge(fine, selected).mesh
        else:
            return None
    return tuple(result)
