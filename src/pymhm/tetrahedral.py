"""Affine tetrahedral geometry and scalar Pk elements in three dimensions."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.mesh import FloatArray, IntArray, positive_int
from pymhm.tetra_lagrange import continuous_tetra_nodes, tetra_polynomials, tetra_values_gradients

_EDGES = tuple(combinations(range(4), 2))


def _real(value: Any, name: str) -> FloatArray:
    """Copy a finite real array without discarding an imaginary component."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite and real")
    return np.array(raw, dtype=float, copy=True)


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


def tetrahedron_quadrature(order: int = 5) -> tuple[FloatArray, FloatArray]:
    """Return positive Duffy-product quadrature with weights normalized to unit volume."""
    order = positive_int(order, "order")
    x, w = leggauss(order)
    x, w = (x + 1) / 2, w / 2
    a, b, c = np.meshgrid(x, x, x, indexing="ij")
    wa, wb, wc = np.meshgrid(w, w, w, indexing="ij")
    xyz = np.array([a, (1 - a) * b, (1 - a) * (1 - b) * c]).reshape(3, -1).T
    bary = np.column_stack((1 - xyz.sum(axis=1), xyz))
    weights = (6 * wa * wb * wc * (1 - a) ** 2 * (1 - b)).ravel()
    return bary, weights


def tetra_nodal_space(mesh: TetraMesh, degree: int) -> tuple[IntArray, FloatArray]:
    """Enumerate continuous Pk nodes using exact topological identities."""
    degree = positive_int(degree, "degree")
    if degree > 2:
        return continuous_tetra_nodes(mesh, degree)
    if degree == 1:
        return mesh.cells, mesh.points
    edges, inverse = np.unique(
        np.sort(mesh.cells[:, _EDGES].reshape(-1, 2), axis=1), axis=0, return_inverse=True
    )
    dofs = np.column_stack((mesh.cells, len(mesh.points) + inverse.reshape(-1, 6)))
    points = np.vstack((mesh.points, mesh.points[edges].mean(axis=1)))
    return dofs, points


def tetra_basis(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate Pk cardinal functions and barycentric first derivatives."""
    degree = positive_int(degree, "degree")
    if degree > 2:
        return tetra_values_gradients(degree, bary)
    bary = _real(bary, "barycentric points")
    if bary.ndim != 2 or bary.shape[1] != 4:
        raise ValueError("barycentric points must have shape (n, 4)")
    if degree == 1:
        return bary, np.broadcast_to(np.eye(4), (len(bary), 4, 4))
    values = np.column_stack(
        (bary * (2 * bary - 1), *(4 * bary[:, i] * bary[:, j] for i, j in _EDGES))
    )
    derivatives = np.zeros((len(bary), 10, 4))
    for i in range(4):
        derivatives[:, i, i] = 4 * bary[:, i] - 1
    for k, (i, j) in enumerate(_EDGES):
        derivatives[:, 4 + k, i] = 4 * bary[:, j]
        derivatives[:, 4 + k, j] = 4 * bary[:, i]
    return values, derivatives


def tetra_tabulate(
    mesh: TetraMesh, degree: int, bary: FloatArray
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
    """Return nodal DOFs/coordinates, basis values and physical gradients."""
    dofs, points = tetra_nodal_space(mesh, degree)
    values, derivatives = tetra_basis(degree, bary)
    jacobian = (mesh.points[mesh.cells[:, 1:]] - mesh.points[mesh.cells[:, :1]]).transpose(0, 2, 1)
    inverse = np.linalg.inv(jacobian)
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
    return dofs, points, values, np.einsum("qia,taj->tqij", derivatives, gradients)


def tetra_element_tabulate(
    mesh: TetraMesh, degree: int, bary: FloatArray
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Tabulate continuous Pk values, physical gradients and physical Hessians."""
    dofs, points, values, gradient = tetra_tabulate(mesh, degree, bary)
    _, _, second = tetra_polynomials(degree, bary)
    inverse = np.linalg.inv(
        (mesh.points[mesh.cells[:, 1:]] - mesh.points[mesh.cells[:, :1]]).transpose(0, 2, 1)
    )
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
    hessian = np.einsum("qiab,tac,tbd->tqicd", second, gradients, gradients)
    return dofs, points, values, gradient, hessian


def scalar_values_3d(coefficient: Any, points: FloatArray) -> FloatArray:
    """Evaluate finite real scalar data at points of shape (n, 3)."""
    values = _real(
        coefficient(points) if callable(coefficient) else coefficient, "scalar coefficient"
    )
    try:
        return np.broadcast_to(values, (len(points),))
    except ValueError as exc:
        raise ValueError("scalar coefficient must return one value per point") from exc


def tensor_values_3d(coefficient: Any, points: FloatArray) -> FloatArray:
    """Evaluate positive scalar or symmetric positive-definite 3 by 3 diffusion."""
    values = _real(coefficient(points) if callable(coefficient) else coefficient, "diffusion")
    if values.ndim == 0 or values.shape == (len(points),):
        values = np.broadcast_to(values, (len(points),))[:, None, None] * np.eye(3)
    try:
        values = np.broadcast_to(values, (len(points), 3, 3))
    except ValueError as exc:
        raise ValueError("diffusion must return scalar values or 3 by 3 tensors") from exc
    scale = np.max(np.abs(values), axis=(1, 2))
    if np.any(
        np.max(np.abs(values - values.transpose(0, 2, 1)), axis=(1, 2)) > 1e-12 * scale
    ) or np.any(np.linalg.eigvalsh(values) <= 0):
        raise ValueError("diffusion must be symmetric positive definite")
    return values


def tetra_operators(
    mesh: TetraMesh, degree: int = 2, *, diffusion: Any = 1.0, source: Any = 0.0, order: int = 5
) -> tuple[Any, Any, FloatArray]:
    """Assemble sparse scalar diffusion, consistent mass and source using positive quadrature."""
    bary, weights = tetrahedron_quadrature(max(order, degree + 2))
    dofs, points, values, gradients = tetra_tabulate(mesh, degree, bary)
    x = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    tensors = tensor_values_3d(diffusion, x.reshape(-1, 3)).reshape(*x.shape[:2], 3, 3)
    force = scalar_values_3d(source, x.reshape(-1, 3)).reshape(x.shape[:2])
    stiffness = np.einsum(
        "t,q,tqia,tqab,tqjb->tij", mesh.volumes, weights, gradients, tensors, gradients
    )
    mass = np.einsum("t,q,qi,qj->tij", mesh.volumes, weights, values, values)
    load = np.einsum("t,q,qi,tq->ti", mesh.volumes, weights, values, force)
    rows = np.broadcast_to(dofs[:, :, None], stiffness.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], stiffness.shape).ravel()
    size = len(points)
    matrix = sparse.coo_matrix((stiffness.ravel(), (rows, columns)), shape=(size, size)).tocsc()
    assembled_mass = sparse.coo_matrix((mass.ravel(), (rows, columns)), shape=(size, size)).tocsc()
    assembled_load = np.zeros(size)
    np.add.at(assembled_load, dofs, load)
    return matrix, assembled_mass, assembled_load
