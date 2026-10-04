"""Conforming red-green refinement of triangular macro meshes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymhm.core.validation import IntArray
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.roundoff import area_coordinate_uncertainty
from pymhm.meshes.triangle import TriangleMesh

if TYPE_CHECKING:
    from pymhm.meshes.polygonal import PolygonMesh


def validate_submesh(coarse: "TriangleMesh | PolygonMesh", cell: int, fine: TriangleMesh) -> None:
    """Check a conforming triangular partition covers exactly one macro polygon.

    Vertices lie in the macro triangle, areas agree, and every fine boundary edge
    is contained in one macro edge. The last condition rejects holes and hanging
    nodes represented as unmatched interior boundaries. Membership accounts for
    absolute coordinate roundoff transported through the inverse macro map;
    domain-area conservation uses a separate determinant perturbation bound.
    """
    if not isinstance(fine, TriangleMesh):
        raise ValueError("local meshes must be TriangleMesh instances")
    vertices = coarse.points[coarse.cells[cell]]
    if len(vertices) != 3:
        _validate_polygon_submesh(coarse, cell, fine)
        return
    transform = (vertices[1:] - vertices[0]).T
    local = np.linalg.solve(transform, (fine.points - vertices[0]).T).T
    bary = np.column_stack((1 - local.sum(axis=1), local))
    condition = np.linalg.cond(transform)
    tolerance = 256 * np.finfo(float).eps * max(1.0, condition)
    # If T xi = x-v0, first-order coordinate perturbations satisfy
    # |delta xi| <= |T^-1| (|delta x|+|delta v0|+|delta T| |xi|).
    # Eight roundoff units cover coordinate reconstruction and subtraction;
    # the existing condition-scaled tolerance covers the small linear solve.
    epsilon = 8 * np.finfo(float).eps
    transform_error = epsilon * (abs(vertices[1:]) + abs(vertices[0])).T
    coordinate_error = epsilon * (abs(fine.points) + abs(vertices[0]))
    local_error = (coordinate_error + abs(local) @ transform_error.T) @ abs(
        np.linalg.inv(transform)
    ).T
    membership_tolerance = tolerance + np.column_stack((local_error.sum(axis=1), local_error))
    macro = TriangleMesh(vertices, np.array([[0, 1, 2]]))
    area_roundoff = area_coordinate_uncertainty(macro, np.zeros(2))
    area_roundoff += area_coordinate_uncertainty(fine, np.zeros(2))
    if np.any(bary < -membership_tolerance) or not np.isclose(
        fine.areas.sum(), coarse.areas[cell], rtol=tolerance, atol=area_roundoff
    ):
        raise ValueError("local mesh must cover its macro triangle exactly")
    boundary = bary[fine.faces[fine.boundary_faces]]
    boundary_tolerance = membership_tolerance[fine.faces[fine.boundary_faces]]
    if not np.all(np.any(np.all(abs(boundary) <= boundary_tolerance, axis=1), axis=1)):
        raise ValueError("local mesh has an unmatched interior boundary")


def _validate_polygon_submesh(
    coarse: "TriangleMesh | PolygonMesh", cell: int, fine: TriangleMesh
) -> None:
    """Check nonconvex membership, oriented boundary coverage and physical area.

    Polygon ears define the membership set, not a required fine triangulation.
    Every fine boundary interval must cover a portion of an original macroedge
    in the same direction, and those intervals must partition every macroedge
    once. This rejects holes, hanging internal boundaries and duplicate sheets.
    Coordinate envelopes and area perturbations remain independent checks.
    """
    macro = coarse.submesh(cell, 1)
    vertices = coarse.points[coarse.cells[cell]]
    epsilon = np.finfo(float).eps
    inside = np.zeros(len(fine.points), dtype=bool)
    for ear in macro.points[macro.cells]:
        transform = (ear[1:] - ear[0]).T
        inverse = np.linalg.inv(transform)
        local = (fine.points - ear[0]) @ inverse.T
        bary = np.column_stack((1 - local.sum(axis=1), local))
        coordinate_error = 8 * epsilon * (abs(fine.points) + abs(ear[0]))
        transform_error = 8 * epsilon * (abs(ear[1:]) + abs(ear[0])).T
        local_error = (coordinate_error + abs(local) @ transform_error.T) @ abs(inverse).T
        tolerance = 256 * epsilon * max(1.0, np.linalg.cond(transform))
        envelope = tolerance + np.column_stack((local_error.sum(axis=1), local_error))
        inside |= np.all(bary >= -envelope, axis=1)
    area_roundoff = area_coordinate_uncertainty(macro, np.zeros(2))
    area_roundoff += area_coordinate_uncertainty(fine, np.zeros(2))
    if not np.all(inside) or not np.isclose(
        fine.areas.sum(), coarse.areas[cell], rtol=256 * epsilon, atol=area_roundoff
    ):
        raise ValueError("local mesh must cover its macro polygon exactly")
    boundary = fine.points[fine.faces[fine.boundary_faces]]
    assigned = np.zeros(len(boundary), dtype=bool)
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        tangent = end - start
        squared_length = tangent @ tangent
        relative = boundary - start
        parameter = relative @ tangent / squared_length
        error = 16 * epsilon * (abs(boundary) + abs(start) + abs(end))
        parameter_error = error @ abs(tangent) / squared_length + 256 * epsilon
        cross = relative[..., 0] * tangent[1] - relative[..., 1] * tangent[0]
        cross_error = error @ abs(tangent[::-1]) + 256 * epsilon * squared_length
        matches = np.all(
            (abs(cross) <= cross_error)
            & (parameter >= -parameter_error)
            & (parameter <= 1 + parameter_error),
            axis=1,
        )
        if np.any(assigned & matches):
            raise ValueError("local boundary edge belongs to multiple macro edges")
        assigned |= matches
        intervals = parameter[matches]
        interval_error = parameter_error[matches]
        order = np.argsort(intervals[:, 0])
        intervals, interval_error = intervals[order], interval_error[order]
        if (
            not len(intervals)
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or abs(intervals[0, 0]) > interval_error[0, 0]
            or abs(intervals[-1, 1] - 1) > interval_error[-1, 1]
            or np.any(
                abs(intervals[1:, 0] - intervals[:-1, 1])
                > interval_error[1:, 0] + interval_error[:-1, 1]
            )
        ):
            raise ValueError("local boundary must partition each oriented macro edge exactly")
    if not np.all(assigned):
        raise ValueError("local mesh has an unmatched interior boundary")


@dataclass(frozen=True)
class TriangleRefinement:
    """Refined geometry and exact topological ancestry.

    ``parent_cells`` maps new cells to old cells. ``parent_faces`` maps new
    faces contained in old faces to their old index, and is -1 otherwise.
    """

    mesh: TriangleMesh
    parent_cells: IntArray
    parent_faces: IntArray


def refine_triangles(mesh: TriangleMesh, marked: Any) -> TriangleRefinement:
    """Red-refine selected cells and complete a conforming red-green partition.

    ``marked`` is a boolean array with one entry per cell. Red refinement
    bisects all three edges into four children. Closure promotes any cell with
    two bisected edges to red refinement; remaining single-edge cells are split
    into two green children. Shared midpoints are created once, without geometric
    searches or hanging nodes. Repeated green refinement is supported, but no
    mesh-independent minimum-angle bound is asserted for arbitrary marking.
    """
    selection = np.asarray(marked)
    if selection.dtype != bool or selection.shape != (len(mesh.cells),):
        raise ValueError("marked must be a boolean array with one entry per triangle")
    split = np.zeros(len(mesh.faces), dtype=bool)
    split[mesh.cell_faces[selection].ravel()] = True
    while True:
        count = split[mesh.cell_faces].sum(axis=1)
        promote = count == 2
        if not np.any(promote):
            break
        split[mesh.cell_faces[promote].ravel()] = True
    selected_faces = np.flatnonzero(split)
    midpoints = np.full(len(mesh.faces), -1, dtype=np.int64)
    midpoints[selected_faces] = len(mesh.points) + np.arange(len(selected_faces))
    points = np.vstack((mesh.points, mesh.points[mesh.faces[selected_faces]].mean(axis=1)))
    cells: list[tuple[int, int, int]] = []
    parents: list[int] = []
    for cell, vertices in enumerate(mesh.cells):
        edges = mesh.cell_faces[cell]
        active = np.flatnonzero(split[edges])
        a, b, c = vertices
        children: tuple[tuple[int, int, int], ...]
        if len(active) == 3:
            ab, bc, ca = midpoints[edges]
            children = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
        elif len(active) == 1:
            edge = int(active[0])
            a, b, c = np.roll(vertices, -edge)
            m = midpoints[edges[edge]]
            children = ((a, m, c), (m, b, c))
        else:
            children = ((a, b, c),)
        cells.extend(children)
        parents.extend([cell] * len(children))
    refined = TriangleMesh(points, np.asarray(cells, dtype=np.int64))
    incident: list[set[int]] = [set() for _ in points]
    for face, (a, b) in enumerate(mesh.faces):
        incident[a].add(face)
        incident[b].add(face)
        if split[face]:
            incident[midpoints[face]].add(face)
    face_parents = np.full(len(refined.faces), -1, dtype=np.int64)
    for face, (a, b) in enumerate(refined.faces):
        common = incident[a] & incident[b]
        if common:
            face_parents[face] = common.pop()
    return TriangleRefinement(refined, np.asarray(parents, dtype=np.int64), face_parents)


def transfer_skeleton(
    original: SkeletonSpace, refinement: TriangleRefinement, *, new_face: FaceSpace
) -> SkeletonSpace:
    """Restrict old face partitions to children and initialize new interior faces.

    Restriction respects both global face orientations. Existing breakpoints and
    degrees are retained on their physical support, including continuous spaces.
    The new-face specification defines approximation only on newly created edges.
    """
    faces = []
    old, new = original.mesh, refinement.mesh
    if len(refinement.parent_faces) != len(new.faces):
        raise ValueError("refinement face ancestry does not match the new mesh")
    for face, parent in enumerate(refinement.parent_faces):
        if parent < 0:
            faces.append(new_face)
            continue
        space = original.faces[parent]
        start, end = old.points[old.faces[parent]]
        tangent = end - start
        endpoints = (new.points[new.faces[face]] - start) @ tangent / (tangent @ tangent)
        left, right = float(endpoints.min()), float(endpoints.max())
        breaks = np.asarray(space.breaks)
        inner = breaks[
            (breaks > left + 64 * np.finfo(float).eps) & (breaks < right - 64 * np.finfo(float).eps)
        ]
        parameters = np.sort((inner - endpoints[0]) / (endpoints[1] - endpoints[0]))
        local_breaks = np.r_[0.0, parameters, 1.0]
        samples = (
            endpoints[0]
            + (endpoints[1] - endpoints[0]) * (local_breaks[:-1] + local_breaks[1:]) / 2
        )
        indices = np.searchsorted(breaks, samples, side="right") - 1
        faces.append(
            FaceSpace(
                tuple(local_breaks), tuple(space.degrees[i] for i in indices), space.continuous
            )
        )
    return SkeletonSpace(new, tuple(faces), original.components)
