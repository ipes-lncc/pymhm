"""Affine tetrahedral geometry, oriented topology and conforming red subdivision."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.core.validation import real_array as _real


def _dyadic(value: int, name: str) -> int:
    """Require an edge subdivision count supported by conforming red refinement."""
    value = positive_int(value, name)
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two")
    return value


class TetraMesh:
    """Oriented straight-sided tetrahedra with signed triangular face incidence.

    Faces have a single normal directed outward from their first adjacent cell.
    ``cell_faces[:, i]`` is opposite vertex ``i``. Arrays are copied and frozen;
    duplicate, degenerate and locally nonmanifold cells are rejected.
    """

    def __init__(self, points: Any, cells: Any) -> None:
        """Validate finite three-dimensional coordinates and integer connectivity."""
        vertices = _real(points, "points")
        raw = np.asarray(cells)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("points must have shape (n, 3), n >= 4")
        if raw.ndim != 2 or raw.shape[1] != 4 or not len(raw):
            raise ValueError("cells must have shape (n, 4), n >= 1")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("cells must contain integers")
        indices = np.array(raw, dtype=np.int64, copy=True)
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError("cell vertex outside points")
        if len(np.unique(np.sort(indices, axis=1), axis=0)) != len(indices):
            raise ValueError("duplicate tetrahedron")
        edges = vertices[indices[:, 1:]] - vertices[indices[:, :1]]
        determinant = np.linalg.det(edges)
        scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
        if np.any(np.abs(determinant) <= 32 * np.finfo(float).eps * scale):
            raise ValueError("degenerate tetrahedron")
        negative = determinant < 0
        indices[negative, 1:3] = indices[negative, 2:0:-1]
        self.points, self.cells = vertices, indices
        self.volumes = np.abs(determinant) / 6
        faces: list[tuple[int, ...]] = []
        owners: list[list[int]] = []
        lookup: dict[tuple[int, ...], int] = {}
        self.cell_faces = np.empty((len(indices), 4), dtype=np.int64)
        for cell, ids in enumerate(indices):
            for side in range(4):
                key = tuple(sorted(np.delete(ids, side)))
                if key not in lookup:
                    lookup[key] = len(faces)
                    faces.append(key)
                    owners.append([])
                face = lookup[key]
                owners[face].append(cell)
                if len(owners[face]) > 2:
                    raise ValueError("nonmanifold triangular face")
                self.cell_faces[cell, side] = face
        self.faces = np.asarray(faces, dtype=np.int64)
        self.face_cells = np.array([owner + [-1] * (2 - len(owner)) for owner in owners])
        face_points = vertices[self.faces]
        cross = np.cross(
            face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0]
        )
        self.areas = np.linalg.norm(cross, axis=1) / 2
        self.normals = cross / (2 * self.areas[:, None])
        self.signs = np.empty((len(indices), 4), dtype=np.int64)
        centers = vertices[indices].mean(axis=1)
        midpoints = face_points.mean(axis=1)
        for face, adjacent in enumerate(owners):
            if np.dot(self.normals[face], midpoints[face] - centers[adjacent[0]]) < 0:
                self.normals[face] *= -1
            for index, cell in enumerate(adjacent):
                orientation = np.dot(self.normals[face], midpoints[face] - centers[cell])
                if (orientation > 0) != (index == 0):
                    raise ValueError("nonmanifold cells overlap across their shared face")
                self.signs[cell, np.flatnonzero(self.cell_faces[cell] == face)[0]] = 1 - 2 * index
        self.boundary_faces = np.flatnonzero(self.face_cells[:, 1] < 0)
        for value in vars(self).values():
            value.setflags(write=False)

    @classmethod
    def unit_cube(cls, subdivisions: int = 1) -> TetraMesh:
        """Triangulate each Cartesian cube into six conforming Freudenthal tetrahedra."""
        n = positive_int(subdivisions, "subdivisions")
        lattice = np.indices((n + 1,) * 3).reshape(3, -1).T
        points = lattice / n
        cells = []
        for corner in np.ndindex((n,) * 3):
            for permutation in permutations(range(3)):
                node = np.array(corner)
                vertices = [np.ravel_multi_index(tuple(node), (n + 1,) * 3)]
                for axis in permutation:
                    node[axis] += 1
                    vertices.append(np.ravel_multi_index(tuple(node), (n + 1,) * 3))
                cells.append(vertices)
        return cls(points, np.asarray(cells))

    def submesh(self, cell: int, refinement: int = 2) -> TetraMesh:
        """Return affine conforming red refinement of one macrocell, with r cubed cells."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.cells):
            raise ValueError("cell outside mesh")
        bary, cells = _reference_refinement(_dyadic(refinement, "refinement"))
        return TetraMesh(bary @ self.points[self.cells[cell]], cells)


@lru_cache(maxsize=8)
def _reference_refinement(refinement: int) -> tuple[FloatArray, IntArray]:
    """Build barycentric red refinement with one deterministic interior octahedron diagonal."""
    points = list(np.eye(4))
    cells = [[0, 1, 2, 3]]
    for _ in range(refinement.bit_length() - 1):
        cache: dict[tuple[int, ...], int] = {}
        refined = []
        for a, b, c, d in cells:
            mids = []
            for i, j in combinations((a, b, c, d), 2):
                key = tuple(sorted((i, j)))
                if key not in cache:
                    cache[key] = len(points)
                    points.append((points[i] + points[j]) / 2)
                mids.append(cache[key])
            ab, ac, ad, bc, bd, cd = mids
            refined.extend(
                [
                    [a, ab, ac, ad],
                    [ab, b, bc, bd],
                    [ac, bc, c, cd],
                    [ad, bd, cd, d],
                    [ab, ac, ad, cd],
                    [ab, ac, bc, cd],
                    [ab, ad, bd, cd],
                    [ab, bc, bd, cd],
                ]
            )
        cells = refined
    coordinates, connectivity = np.asarray(points), np.asarray(cells, dtype=np.int64)
    coordinates.setflags(write=False)
    connectivity.setflags(write=False)
    return coordinates, connectivity


def tetra_barycentric_gradients(mesh: TetraMesh) -> FloatArray:
    """Return affine gradients of lambda_0,...,lambda_3 on each tetrahedron.

    Axes are ``(cell, barycentric_coordinate, physical_coordinate)``. The
    inverse vertex Jacobian gives the last three rows; the first row is their
    negative sum. These geometric maps contain no finite-element tabulation.
    """
    inverse = np.linalg.inv(
        (mesh.points[mesh.cells[:, 1:]] - mesh.points[mesh.cells[:, :1]]).transpose(0, 2, 1)
    )
    return np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
