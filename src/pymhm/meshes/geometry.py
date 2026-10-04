"""Oriented surface and geometric-kernel checks for star-shaped polyhedral cells."""

from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from pymhm.core.validation import FloatArray, IntArray


def oriented_cell_faces(
    points: FloatArray, faces: list[IntArray], cells: list[IntArray]
) -> tuple[IntArray, ...]:
    """Orient each connected closed cell boundary through opposing directed edge incidences.

    The global orientation is selected by its signed surface-volume integral;
    neither a convex hull nor an assumed interior centroid is used. The caller
    subsequently checks that the surface is embedded by its positive cone mesh.
    """
    result = []
    for ids in cells:
        edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for side, face in enumerate(ids):
            polygon = faces[face]
            for a, b in zip(polygon, np.roll(polygon, -1), strict=True):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                edges.setdefault(key, []).append((side, 1 if a < b else -1))
        adjacency: list[list[tuple[int, int]]] = [[] for _ in ids]
        for incident in edges.values():
            if len(incident) != 2:
                raise ValueError("polyhedral boundary must be a closed two-manifold")
            (a, da), (b, db) = incident
            adjacency[a].append((b, -da * db))
            adjacency[b].append((a, -da * db))
        signs = np.zeros(len(ids), dtype=np.int64)
        signs[0] = 1
        pending = [0]
        while pending:
            side = pending.pop()
            for neighbor, relation in adjacency[side]:
                wanted = signs[side] * relation
                if signs[neighbor] == 0:
                    signs[neighbor] = wanted
                    pending.append(neighbor)
                elif signs[neighbor] != wanted:
                    raise ValueError("polyhedral boundary must be orientable")
        if np.any(signs == 0):
            raise ValueError(
                "polyhedral boundary must be connected, without separate cavity shells"
            )
        vertices = points[np.unique(np.concatenate([faces[face] for face in ids]))]
        origin = vertices.mean(axis=0)
        scale = float(np.ptp(vertices, axis=0).max())
        if scale == 0:
            raise ValueError("polyhedral cell has zero or unresolved volume")
        volume, absolute_products = 0.0, 0.0
        for face, sign in zip(ids, signs, strict=True):
            polygon = (points[faces[face]] - origin) / scale
            for j in range(1, len(polygon) - 1):
                a, b, c = polygon[[0, j, j + 1]]
                volume += sign * float(np.dot(a, np.cross(b, c))) / 6
                absolute_products += (
                    float(
                        abs(a)
                        @ (abs(b[[1, 2, 0]] * c[[2, 0, 1]]) + abs(b[[2, 0, 1]] * c[[1, 2, 0]]))
                    )
                    / 6
                )
        if abs(volume) <= 256 * np.finfo(float).eps * absolute_products:
            raise ValueError("polyhedral cell has zero or unresolved signed volume")
        result.append(signs if volume > 0 else -signs)
    return tuple(result)


def kernel_center(
    vertices: FloatArray, origins: FloatArray, outward: FloatArray
) -> tuple[FloatArray, float, bool]:
    """Find a strictly interior ball in the intersection of all inward face half-spaces.

    Convex cells retain the arithmetic vertex center. Nonconvex cells maximize
    the ball radius by a four-variable linear program in cell-scaled coordinates.
    Canonically ordered inequalities make degenerate optimal-center selection
    independent of face-list order.
    The returned radius is verified against every original face, independently
    of the optimizer's feasibility tolerance. An empty or unresolved kernel is
    rejected rather than filled by a convex hull.
    """
    center = vertices.mean(axis=0)
    scale = float(np.ptp(vertices, axis=0).max())
    tolerance = 512 * np.finfo(float).eps * scale
    distances = np.einsum("fvi,fi->fv", vertices[None] - origins[:, None], outward)
    convex = bool(np.all(distances <= tolerance))
    if not convex:
        bound = np.einsum("fi,fi->f", origins - center, outward) / scale
        ordering = np.lexsort((bound, outward[:, 2], outward[:, 1], outward[:, 0]))
        problem = linprog(
            [0.0, 0.0, 0.0, -1.0],
            A_ub=np.column_stack((outward, np.ones(len(outward))))[ordering],
            b_ub=bound[ordering],
            bounds=[(None, None)] * 3 + [(0.0, None)],
            method="highs",
        )
        if not problem.success:
            raise ValueError("polyhedral cell requires a nonempty star-shaped kernel")
        center = center + scale * problem.x[:3]
    radius = float(np.min(np.einsum("fi,fi->f", origins - center, outward)))
    if not np.isfinite(radius) or radius <= tolerance:
        raise ValueError("polyhedral cell requires a resolvable strictly interior kernel ball")
    return center, radius, convex


def validate_disjoint_cones(points: FloatArray, cells: IntArray) -> None:
    """Reject interior overlap of tetrahedral cones using the convex separating-axis theorem.

    Face normals and pairwise edge cross products are a complete axis set for
    two tetrahedra. Coordinates are translated/scaled before projection; touching
    faces, edges and the common kernel center do not constitute overlap.
    """
    scale = float(np.ptp(points, axis=0).max())
    scaled = (points - points.mean(axis=0)) / scale
    tetrahedra = scaled[cells]
    tolerance = 1024 * np.finfo(float).eps
    edge_ids = np.asarray(list(combinations(range(4), 2)))
    face_ids = np.asarray(list(combinations(range(4), 3)))
    for first, second in combinations(tetrahedra, 2):
        if np.any(
            np.minimum(first.max(axis=0), second.max(axis=0))
            - np.maximum(first.min(axis=0), second.min(axis=0))
            <= tolerance
        ):
            continue
        edges_a = first[edge_ids[:, 1]] - first[edge_ids[:, 0]]
        edges_b = second[edge_ids[:, 1]] - second[edge_ids[:, 0]]
        axes = np.vstack(
            (
                np.cross(
                    first[face_ids[:, 1]] - first[face_ids[:, 0]],
                    first[face_ids[:, 2]] - first[face_ids[:, 0]],
                ),
                np.cross(
                    second[face_ids[:, 1]] - second[face_ids[:, 0]],
                    second[face_ids[:, 2]] - second[face_ids[:, 0]],
                ),
                np.cross(edges_a[:, None], edges_b[None]).reshape(-1, 3),
            )
        )
        lengths = np.linalg.norm(axes, axis=1)
        axes = axes[lengths > tolerance] / lengths[lengths > tolerance, None]
        a, b = first @ axes.T, second @ axes.T
        overlaps = np.minimum(a.max(axis=0), b.max(axis=0)) - np.maximum(
            a.min(axis=0), b.min(axis=0)
        )
        if np.all(overlaps > tolerance):
            raise ValueError("polyhedral boundary produces overlapping tetrahedral interiors")


def triangle_in_face(corners: FloatArray, triangles: FloatArray, normal: FloatArray) -> bool:
    """Test a fine boundary triangle against the original face's canonical triangulation.

    Conforming local refinements lie within a single canonical triangle even
    when the original polygon is nonconvex. This preserves individual coplanar
    faces and does not accept a triangle spanning a reentrant notch.
    """
    for triangle in triangles:
        edges = np.roll(triangle, -1, axis=0) - triangle
        scale = float(np.ptp(triangle, axis=0).max())
        tolerance = 512 * np.finfo(float).eps * scale
        if np.max(abs((corners - triangle[0]) @ normal)) > tolerance:
            continue
        half_planes = np.cross(edges[:, None], corners[None] - triangle[:, None]) @ normal
        if np.all(half_planes >= -tolerance * np.linalg.norm(edges, axis=1)[:, None]):
            return True
    return False
