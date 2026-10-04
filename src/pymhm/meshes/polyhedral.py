"""Star-shaped polyhedral macrocells with original polygonal faces and tetrahedral interiors."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.core.validation import real_array as _real
from pymhm.meshes.geometry import (
    kernel_center,
    oriented_cell_faces,
    validate_disjoint_cones,
)
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic, _reference_refinement


class PolyhedralMesh:
    """Conforming star-shaped cells bounded by planar simple polygonal faces.

    ``faces`` contains cyclic vertex indices and ``cells`` contains face indices.
    Each face has at most two adjacent cells. Canonical normals point outward
    from the first owner. Each closed cell must have a strictly interior kernel
    ball, whose verified radius is recorded in ``kernel_radii``. Nonconvex cells
    and faces are permitted; curved faces, separate cavity shells and empty
    kernels are rejected. Original polygonal faces remain single entities when
    their interiors are triangulated. Family-wise shape regularity additionally
    requires a uniform lower bound on kernel radius divided by cell diameter.
    """

    def __init__(self, points: Any, faces: Any, cells: Any) -> None:
        """Validate geometry, orientation, closed cell boundaries and positive measures."""
        vertices = _real(points, "points")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("points must have shape (n, 3), n >= 4")
        if len(np.unique(vertices, axis=0)) != len(vertices):
            raise ValueError("polyhedral points require unique coordinates and shared vertex IDs")
        face_ids = self._connectivity(faces, len(vertices), 3, "faces")
        cell_ids = self._connectivity(cells, len(face_ids), 4, "cells")
        if len({tuple(sorted(ids)) for ids in face_ids}) != len(face_ids):
            raise ValueError("duplicate polygonal face")
        if len({tuple(sorted(ids)) for ids in cell_ids}) != len(cell_ids):
            raise ValueError("duplicate polyhedral cell")
        owners: list[list[int]] = [[] for _ in face_ids]
        cell_vertices = []
        for cell, ids in enumerate(cell_ids):
            boundary_edges: dict[tuple[int, int], int] = {}
            nodes = np.unique(np.concatenate([face_ids[face] for face in ids]))
            cell_vertices.append(nodes)
            for face in ids:
                owners[face].append(cell)
                polygon = face_ids[face]
                for first, second in zip(polygon, np.roll(polygon, -1), strict=True):
                    edge = (min(int(first), int(second)), max(int(first), int(second)))
                    boundary_edges[edge] = boundary_edges.get(edge, 0) + 1
            if any(count != 2 for count in boundary_edges.values()):
                raise ValueError("polyhedral boundary must be a closed two-manifold")
        if any(not owner or len(owner) > 2 for owner in owners):
            raise ValueError("each polygonal face needs one or two adjacent cells")
        input_signs = oriented_cell_faces(vertices, face_ids, cell_ids)
        centers = np.array([vertices[ids].mean(axis=0) for ids in cell_vertices])
        triangles, normals, areas, origins, tangents = [], [], [], [], []
        canonical_faces, convex_faces = [], []
        for face, ids in enumerate(face_ids):
            polygon = vertices[ids]
            scale = float(np.ptp(polygon, axis=0).max())
            tolerance = 256 * np.finfo(float).eps * scale
            cross = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            length = float(np.linalg.norm(cross))
            if length <= tolerance * scale:
                candidates = np.cross(
                    polygon[1:] - polygon[0], np.roll(polygon[1:], -1, axis=0) - polygon[0]
                )
                lengths = np.linalg.norm(candidates, axis=1)
                cross = candidates[int(np.argmax(lengths))]
                length = float(lengths.max())
                if length <= tolerance * scale:
                    raise ValueError("polygonal faces must have nondegenerate corners")
            normal = cross / length
            if np.max(np.abs((polygon - polygon[0]) @ normal)) > tolerance:
                raise ValueError("polygonal faces must be planar")
            edge_vectors = np.roll(polygon, -1, axis=0) - polygon
            corners = np.cross(edge_vectors, np.roll(edge_vectors, -1, axis=0)) @ normal
            half_planes = (
                np.cross(edge_vectors[:, None, :], polygon[None] - polygon[:, None]) @ normal
            )
            convex_face = bool(
                np.all(half_planes >= -tolerance * scale) and np.all(corners > tolerance * scale)
            )
            if not convex_face:
                vector_area = np.cross(
                    polygon - polygon[0], np.roll(polygon, -1, axis=0) - polygon[0]
                ).sum(axis=0)
                if vector_area @ normal < 0:
                    normal = -normal
                tangent = polygon[1] - polygon[0]
                tangent /= np.linalg.norm(tangent)
                coordinates = (polygon - polygon[0]) @ np.array(
                    [tangent, np.cross(normal, tangent)]
                ).T
                PolygonMesh(coordinates, (np.arange(len(polygon)),))
            owner = owners[face][0]
            side = int(np.flatnonzero(cell_ids[owner] == face)[0])
            if input_signs[owner][side] < 0:
                ids = ids[::-1]
                normal = -normal
            if len(owners[face]) == 2:
                other = owners[face][1]
                other_side = int(np.flatnonzero(cell_ids[other] == face)[0])
                if input_signs[other][other_side] == input_signs[owner][side]:
                    raise ValueError("adjacent polyhedral cells require opposing face orientations")
            ids = np.roll(ids, -int(np.argmin(ids)))
            polygon = vertices[ids]
            if convex_face:
                fan = np.array([[ids[0], ids[j], ids[j + 1]] for j in range(1, len(ids) - 1)])
            else:
                tangent = polygon[1] - polygon[0]
                tangent /= np.linalg.norm(tangent)
                coordinates = (polygon - polygon[0]) @ np.array(
                    [tangent, np.cross(normal, tangent)]
                ).T
                face_mesh = PolygonMesh(coordinates, (np.arange(len(polygon)),))
                fan = ids[face_mesh._triangles[0]]
            tri = vertices[fan]
            area = (
                np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
            )
            origin = np.einsum("t,ti->i", area, tri.mean(axis=1)) / area.sum()
            tangent = polygon[1] - polygon[0]
            tangent /= np.linalg.norm(tangent)
            canonical_faces.append(ids)
            convex_faces.append(convex_face)
            triangles.append(fan)
            normals.append(normal)
            areas.append(float(area.sum()))
            origins.append(origin)
            tangents.append(np.array([tangent, np.cross(normal, tangent)]))
        normals_array = np.asarray(normals)
        origins_array = np.asarray(origins)
        signs, volumes, radii, convex_cells = [], [], [], []
        for cell, ids in enumerate(cell_ids):
            orientation = np.array([1 if owners[face][0] == cell else -1 for face in ids])
            center, radius, convex = kernel_center(
                vertices[cell_vertices[cell]],
                origins_array[ids],
                orientation[:, None] * normals_array[ids],
            )
            centers[cell] = center
            radii.append(radius)
            convex_cells.append(convex)
            volume = 0.0
            for face, sign in zip(ids, orientation, strict=True):
                normal = sign * normals_array[face]
                center_distance = float((centers[cell] - origins_array[face]) @ normal)
                volume -= areas[face] * center_distance / 3
            if not convex or not all(convex_faces[face] for face in ids):
                cone_points = np.vstack((vertices, center))
                cone_cells = np.array(
                    [[len(vertices), *tri] for face in ids for tri in triangles[face]]
                )
                validate_disjoint_cones(
                    cone_points[np.unique(cone_cells)],
                    np.searchsorted(np.unique(cone_cells), cone_cells),
                )
            signs.append(orientation)
            volumes.append(volume)
        self.points = vertices
        self.faces = tuple(canonical_faces)
        self.cells = tuple(cell_ids)
        self.cell_faces = self.cells
        self.face_cells = np.array([owner + [-1] * (2 - len(owner)) for owner in owners])
        self.normals = normals_array
        self.areas = np.asarray(areas)
        self.face_origins = origins_array
        self.face_tangents = np.asarray(tangents)
        self.face_triangles = tuple(triangles)
        self.centers = centers
        self.kernel_radii = np.asarray(radii)
        self.convex_cells = np.asarray(convex_cells)
        self.volumes = np.asarray(volumes)
        self.signs = tuple(signs)
        self.boundary_faces = np.flatnonzero(self.face_cells[:, 1] < 0)
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            else:
                for array in value:
                    array.setflags(write=False)

    @staticmethod
    def _connectivity(values: Any, size: int, minimum: int, name: str) -> list[IntArray]:
        """Validate variable-length integer index lists without coercing fractional values."""
        result = []
        for entry in values:
            raw = np.asarray(entry)
            if raw.ndim != 1 or len(raw) < minimum or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name} must contain integer index lists of length >= {minimum}")
            ids = np.array(raw, dtype=np.int64, copy=True)
            if np.any(ids < 0) or np.any(ids >= size) or len(np.unique(ids)) != len(ids):
                raise ValueError(f"{name} contains repeated or out-of-range indices")
            result.append(ids)
        if not result:
            raise ValueError(f"{name} cannot be empty")
        return result

    @classmethod
    def cubes(cls, subdivisions: int = 1) -> PolyhedralMesh:
        """Create n cubical cells per coordinate, retaining quadrilateral macrofaces."""
        n = positive_int(subdivisions, "subdivisions")
        lattice = np.indices((n + 1,) * 3).reshape(3, -1).T
        points = lattice / n
        faces: list[list[int]] = []
        cells = []
        lookup: dict[tuple[int, ...], int] = {}
        for i, j, k in np.ndindex((n, n, n)):
            nodes = np.array(
                [
                    np.ravel_multi_index((i + a, j + b, k + c), (n + 1,) * 3)
                    for a, b, c in np.ndindex((2, 2, 2))
                ]
            )
            local = (
                [0, 1, 3, 2],
                [4, 6, 7, 5],
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 2, 6, 4],
                [1, 5, 7, 3],
            )
            ids = []
            for order in local:
                face = nodes[list(order)].tolist()
                key = tuple(sorted(face))
                if key not in lookup:
                    lookup[key] = len(faces)
                    faces.append(face)
                ids.append(lookup[key])
            cells.append(ids)
        return cls(points, faces, cells)

    @classmethod
    def extrude(
        cls, mesh: PolygonMesh, layers: int = 1, *, interval: tuple[float, float] = (0.0, 1.0)
    ) -> PolyhedralMesh:
        """Extrude a conforming star-shaped polygon mesh into planar-faced prismatic cells."""
        if not isinstance(mesh, PolygonMesh):
            raise TypeError("extrusion requires PolygonMesh")
        layers = positive_int(layers, "layers")
        raw = _real(interval, "extrusion interval")
        if raw.shape != (2,) or raw[1] <= raw[0]:
            raise ValueError("extrusion interval must have two increasing coordinates")
        z = np.linspace(raw[0], raw[1], layers + 1)
        count = len(mesh.points)
        points = np.column_stack((np.tile(mesh.points, (layers + 1, 1)), np.repeat(z, count)))
        faces: list[list[int]] = []
        cells = []
        lookup: dict[tuple[int, ...], int] = {}
        for layer in range(layers):
            for polygon in mesh.cells:
                bottom = polygon + layer * count
                top = bottom + count
                local = [bottom.tolist(), top[::-1].tolist()]
                local.extend(
                    [
                        [int(a), int(b), int(b + count), int(a + count)]
                        for a, b in zip(bottom, np.roll(bottom, -1), strict=True)
                    ]
                )
                ids = []
                for face in local:
                    key = tuple(sorted(face))
                    if key not in lookup:
                        lookup[key] = len(faces)
                        faces.append(face)
                    ids.append(lookup[key])
                cells.append(ids)
        return cls(points, faces, cells)

    def submesh(self, cell: int, refinement: int = 1) -> TetraMesh:
        """Cone canonical face triangulations to the interior center and refine conformingly.

        Neighboring cells share the same face triangulation. Dyadic red refinement
        merges nodes by exact integer barycentric topology, not coordinate rounding.
        """
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.cells):
            raise ValueError("cell outside mesh")
        refinement = _dyadic(refinement, "refinement")
        nodes = np.unique(np.concatenate([self.faces[face] for face in self.cells[cell]]))
        local_ids = {int(node): index for index, node in enumerate(nodes)}
        points = np.vstack((self.points[nodes], self.centers[cell]))
        cells = [
            [len(nodes), *(local_ids[int(v)] for v in triangle)]
            for face in self.cells[cell]
            for triangle in self.face_triangles[face]
        ]
        coarse = TetraMesh(points, np.asarray(cells))
        if refinement == 1:
            return coarse
        bary, reference_cells = _reference_refinement(refinement)
        weights = np.rint(refinement * bary).astype(int)
        lookup: dict[tuple[tuple[int, int], ...], int] = {}
        refined_points: list[FloatArray] = []
        refined_cells: list[IntArray] = []
        for tetra in coarse.cells:
            ids = []
            for row in weights:
                key = tuple(sorted((int(v), int(w)) for v, w in zip(tetra, row, strict=True) if w))
                if key not in lookup:
                    lookup[key] = len(refined_points)
                    refined_points.append(row @ coarse.points[tetra] / refinement)
                ids.append(lookup[key])
            refined_cells.extend(np.asarray(ids)[reference_cells])
        return TetraMesh(np.asarray(refined_points), np.asarray(refined_cells))
