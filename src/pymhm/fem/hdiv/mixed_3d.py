"""Affine mixed-cell H(div) volume and oriented trace forms in three dimensions.

Normal densities are dual to integral face moments; pressure uses the family's
cell polynomial convention. Trace degrees and partitions remain caller-chosen.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    face_size,
)
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class _FacePartition:
    """Affine subfaces with integral-moment-dual physical normal densities."""

    nodes: FloatArray
    cells: IntArray
    degree: int
    measure: float

    @property
    def width(self) -> int:
        """Return the number of polynomial moments per subface."""
        return face_size(self.cells.shape[1], self.degree)

    @property
    def size(self) -> int:
        """Return the subdivided face dimension."""
        return len(self.cells) * self.width

    def coordinates(self, piece: int, points: FloatArray) -> tuple[FloatArray, float]:
        """Map parent coordinates into one selected affine subface."""
        vertices = self.nodes[self.cells[piece]]
        jac = (vertices[1:3] - vertices[:1]).T
        return (points - vertices[0]) @ np.linalg.inv(jac).T, abs(float(np.linalg.det(jac)))

    def locate(self, points: FloatArray) -> int:
        """Locate a contained fine face, rejecting trace partitions that cut it."""
        for piece in range(len(self.cells)):
            uv, _ = self.coordinates(piece, points)
            upper = np.max(uv.sum(axis=1)) if self.cells.shape[1] == 3 else np.max(uv)
            if np.min(uv) >= -1e-10 and upper <= 1 + 1e-10:
                return piece
        raise ValueError("skeleton subfaces must align with local normal-flux faces")

    def evaluate(self, piece: int, points: FloatArray) -> FloatArray:
        """Evaluate physical normal densities dual to subface polynomial moments."""
        uv, det = self.coordinates(piece, points)
        q, w = face_quadrature(self.cells.shape[1], max(3, self.degree + 2))
        tests = face_polynomials(q, self.cells.shape[1], self.degree)
        mass = tests.T @ (w[:, None] * tests)
        return (
            face_polynomials(uv, self.cells.shape[1], self.degree)
            @ np.linalg.inv(mass)
            / (self.measure * det)
        )


def _partition(corners: int, subdivisions: int, degree: int, measure: float) -> _FacePartition:
    """Create uniform subfaces without replacing original macroface identities."""
    n = subdivisions
    if corners == 3:
        triangle = TriangleMesh(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
        ).submesh(0, n)
        return _FacePartition(triangle.points, triangle.cells, degree, measure)
    nodes = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1)])
    cells = np.array(
        [
            [i * (n + 1) + j, (i + 1) * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1]
            for i in range(n)
            for j in range(n)
        ]
    )
    return _FacePartition(nodes, cells, degree, measure)


class Mixed3DSkeleton:
    """Pk triangular and Qk rectangular physical flux traces on original macrofaces."""

    def __init__(self, mesh: AffineMixedMesh, degree: int = 1, subdivisions: int = 1) -> None:
        """Create canonical moment blocks for every face and aligned subdivision."""
        self.degree = positive_int(degree, "trace degree", 0)
        self.subdivisions = positive_int(subdivisions, "subdivisions")
        self.mesh = mesh
        self.partitions = tuple(
            _partition(len(face), subdivisions, degree, mesh.measures[i])
            for i, face in enumerate(mesh.faces)
        )
        self.offsets = np.r_[0, np.cumsum([part.size for part in self.partitions])]

    @property
    def size(self) -> int:
        """Return the global number of scalar normal-flux moments."""
        return int(self.offsets[-1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Return the concatenated face blocks of one macrocell."""
        return np.concatenate(
            [np.arange(self.offsets[f], self.offsets[f + 1]) for f in self.mesh.cell_faces[cell]]
        )


def hdiv3d_operators(
    mesh: AffineMixedMesh,
    family: HDiv3DFamily,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    quadrature_order: int = 5,
) -> tuple[Any, Any, FloatArray, FloatArray]:
    """Assemble conforming mass, divergence, source and physical pressure moments."""
    points, weights = cell_quadrature(mesh.kind, max(quadrature_order, family.pressure_degree + 2))
    dofs = hdiv3d_dofs(mesh, family)
    basis, div, pressure = hdiv3d_basis(mesh, family, points)
    physical = mesh.geometry(points)
    inverse = np.linalg.inv(tensor_values_3d(permeability, physical.reshape(-1, 3))).reshape(
        *physical.shape[:2], 3, 3
    )
    w = mesh.determinants[:, None] * weights
    mass = np.einsum("tq,tqia,tqab,tqjb->tij", w, basis, inverse, basis, optimize=True)
    divergence = np.einsum("tq,qi,tqj->tij", w, pressure, div, optimize=True)
    f = scalar_values_3d(source, physical.reshape(-1, 3)).reshape(w.shape)
    load = np.einsum("tq,qi,tq->ti", w, pressure, f)
    moment = np.einsum("tq,qi->ti", w, pressure)
    nq, npres = int(dofs.max()) + 1, len(mesh.cells) * family.pressure_size
    pids = np.arange(npres).reshape(len(mesh.cells), -1)
    return (
        assemble_element_blocks(mass, dofs, dofs, (nq, nq)),
        assemble_element_blocks(divergence, pids, dofs, (npres, nq)),
        load.ravel(),
        moment.ravel(),
    )


def hdiv3d_trace_mapping(
    skeleton: Mixed3DSkeleton, cell: int, fine: AffineMixedMesh, normal_degree: int = 1
) -> FloatArray:
    """Integrate canonical fine-face moments of each oriented macro normal density."""
    mesh = skeleton.mesh
    coarse_faces = mesh.cell_faces[cell]
    offsets = np.r_[0, np.cumsum([skeleton.partitions[f].size for f in coarse_faces])]
    mapping = np.zeros(
        (
            sum(face_size(len(fine.faces[f]), normal_degree) for f in fine.boundary_faces),
            offsets[-1],
        )
    )
    row = 0
    for face in fine.boundary_faces:
        nodes = fine.points[fine.faces[face]]
        count = face_size(len(nodes), normal_degree)
        uv, w = face_quadrature(len(nodes), max(4, normal_degree + 2))
        physical = face_shape(uv, len(nodes)) @ nodes
        found = False
        for side, parent in enumerate(coarse_faces):
            origin = mesh.points[mesh.faces[parent][0]]
            if np.max(abs((nodes - origin) @ mesh.normals[parent])) > 1e-10 * np.linalg.norm(
                mesh.jacobian[cell]
            ):
                continue
            canonical = mesh.face_coordinates(parent, physical)
            partition = skeleton.partitions[parent]
            piece = partition.locate(mesh.face_coordinates(parent, nodes))
            normal = partition.evaluate(piece, canonical)
            block = face_polynomials(uv, len(nodes), normal_degree).T @ (
                w[:, None] * normal * fine.measures[face] * mesh.signs[cell, side]
            )
            start = offsets[side] + piece * partition.width
            mapping[row : row + count, start : start + partition.width] = block
            found = True
            break
        if not found:
            raise ValueError("local boundary face has no containing macroface")
        row += count
    return mapping


def hdiv3d_boundary_data(
    skeleton: Mixed3DSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate weak pressure or outward physical normal-flux moments at the boundary."""
    mesh = skeleton.mesh
    if not set(neumann).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    load, fixed = np.zeros(skeleton.size), {}
    for face in mesh.boundary_faces:
        partition = skeleton.partitions[face]
        uv, w = face_quadrature(len(mesh.faces[face]), order)
        tests = face_polynomials(uv, len(mesh.faces[face]), skeleton.degree)
        for piece, cell in enumerate(partition.cells):
            nodes = partition.nodes[cell]
            canonical = face_shape(uv, len(cell)) @ nodes
            physical = face_shape(canonical, len(cell)) @ mesh.points[mesh.faces[face]]
            _, det = partition.coordinates(piece, canonical)
            indices = skeleton.offsets[face] + piece * partition.width + np.arange(partition.width)
            if face in neumann:
                value = tests.T @ (
                    w * det * partition.measure * scalar_values_3d(neumann[face], physical)
                )
                fixed.update({int(i): float(v) for i, v in zip(indices, value, strict=True)})
            else:
                load[indices] -= partition.evaluate(piece, canonical).T @ (
                    w * det * partition.measure * scalar_values_3d(dirichlet, physical)
                )
    return load, fixed
