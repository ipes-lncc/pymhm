"""Oriented triangular-face spaces, coupling and physical scalar boundary moments.

Trace coefficients represent densities in the macroface's canonical normal.
Each physical face partition retains its independent Bernstein coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_face_basis, tetra_nodal_space
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.meshes.triangle import TriangleMesh
from pymhm.meshes.validation import validate_face_partition


class TriangularSkeleton:
    """Independent Pk flux densities on subtriangles of each macroface.

    ``subdivisions`` counts uniform edge divisions, giving its square many
    subtriangles by default. Explicit ``face_partitions`` instead prescribe a
    conforming nonuniform triangulation of each original macroface. Each
    subtriangle has (k+1)(k+2)/2 Bernstein coefficients. Sequences select
    independent face degrees. Flux uses the canonical normal of ``TetraMesh``.
    """

    def __init__(
        self,
        mesh: TetraMesh,
        subdivisions: Any = 1,
        *,
        degree: Any = 0,
        face_partitions: tuple[FloatArray, ...] | None = None,
    ) -> None:
        """Create uniform or explicit conforming triangular partitions with Pk modes.

        Explicit vertices use barycentric coordinates in ``mesh.faces[face]``
        ordering. Every original face must be covered without holes, overlaps or
        hanging internal edges. ``subdivisions`` and ``partitions`` retain the
        uniform-grid configuration; ``face_partition`` returns the active geometry.
        """
        raw = np.asarray(subdivisions)
        if raw.ndim == 0:
            raw = np.full(len(mesh.faces), raw)
        if raw.shape != (len(mesh.faces),):
            raise ValueError("subdivisions must be one integer or one per macroface")
        self.mesh = mesh
        self.subdivisions = np.array([_dyadic(value, "subdivisions") for value in raw])
        degrees = np.asarray(degree)
        if degrees.ndim == 0:
            degrees = np.full(len(mesh.faces), degrees)
        if degrees.shape != (len(mesh.faces),):
            raise ValueError("degree must be one integer or one per macroface")
        self.degrees = np.array([positive_int(value, "degree", 0) for value in degrees])
        self.modes = (self.degrees + 1) * (self.degrees + 2) // 2
        reference = TriangleMesh(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
        )
        self.partitions: dict[int, FloatArray] = {}
        for count in np.unique(self.subdivisions):
            fine = reference.submesh(0, int(count))
            bary = np.column_stack((1 - fine.points.sum(axis=1), fine.points))
            self.partitions[int(count)] = bary[fine.cells]
        if face_partitions is None:
            self._face_partitions = tuple(self.partitions[int(n)] for n in self.subdivisions)
            self._face_weights = tuple(np.full(n * n, 1.0 / (n * n)) for n in self.subdivisions)
        else:
            if len(face_partitions) != len(mesh.faces):
                raise ValueError("face_partitions must contain one partition per macroface")
            validated = tuple(validate_face_partition(part) for part in face_partitions)
            self._face_partitions = tuple(item[0] for item in validated)
            self._face_weights = tuple(item[1] for item in validated)
        counts = np.array([len(part) for part in self._face_partitions])
        self.offsets = np.r_[0, np.cumsum(counts * self.modes)]
        self.size = int(self.offsets[-1])
        self.constant_coefficients = np.ones(self.size)

    def face_partition(self, face: int) -> FloatArray:
        """Return active subtriangles in original macroface barycentric coordinates."""
        self.dofs(face)
        return self._face_partitions[face]

    def face_weights(self, face: int) -> FloatArray:
        """Return positive subtriangle area fractions summing to one on a macroface."""
        self.dofs(face)
        return self._face_weights[face]

    def basis(self, face: int, barycentric: FloatArray) -> FloatArray:
        """Evaluate the Bernstein Pk density basis on a canonical subtriangle.

        Degrees zero and one retain their constant and barycentric coordinate
        conventions. All modes integrate to the same fraction of subface area,
        and coefficients equal to one represent a constant density exactly.
        """
        self.dofs(face)
        bary = np.asarray(barycentric)
        if (
            np.iscomplexobj(bary)
            or bary.ndim != 2
            or bary.shape[1] != 3
            or not np.isfinite(bary).all()
        ):
            raise ValueError("subface barycentric points must be finite real triples")
        degree = int(self.degrees[face])
        from pymhm.fem.reference import bernstein_tabulation

        exponents = tuple(
            (i, j, degree - i - j) for i in range(degree, -1, -1) for j in range(degree - i, -1, -1)
        )
        return bernstein_tabulation(bary[:, 1:], exponents)[0]

    def dofs(self, face: int) -> IntArray:
        """Return the global coefficients for one validated macroface index."""
        face = positive_int(face, "face", 0)
        if face >= len(self.mesh.faces):
            raise ValueError("face outside mesh")
        return np.arange(self.offsets[face], self.offsets[face + 1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate the four oriented macroface coefficient blocks of a cell."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cells):
            raise ValueError("cell outside mesh")
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])


def tetra_trace_coupling(
    coarse: TetraMesh, cell: int, fine: TetraMesh, skeleton: TriangularSkeleton, degree: int
) -> FloatArray:
    """Integrate signed P1–P4 traces against aligned Pk triangular face modes."""
    dofs, points = tetra_nodal_space(fine, degree)
    matrix = np.zeros((len(points), len(skeleton.cell_dofs(cell))))
    vertices = coarse.points[coarse.cells[cell]]
    jacobian = (vertices[1:] - vertices[:1]).T
    coordinates = (fine.points - vertices[0]) @ np.linalg.inv(jacobian).T
    macro_bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
    order = max(3, degree + 1, (degree + int(skeleton.degrees.max()) + 3) // 2)
    bary_face, weights = triangle_quadrature(order)
    offset = 0
    for side, face in enumerate(coarse.cell_faces[cell]):
        partitions = skeleton.face_partition(int(face))
        macro_vertices = [
            int(np.flatnonzero(coarse.cells[cell] == node)[0]) for node in coarse.faces[face]
        ]
        for fine_face in fine.boundary_faces:
            ids = fine.faces[fine_face]
            if np.max(np.abs(macro_bary[ids, side])) > 1e-10:
                continue
            local = int(fine.face_cells[fine_face, 0])
            face_bary = macro_bary[ids][:, macro_vertices]
            coordinates = np.einsum("i,sij->sj", face_bary.mean(axis=0), np.linalg.inv(partitions))
            candidates = np.flatnonzero(np.min(coordinates, axis=1) >= -1e-10)
            if len(candidates) != 1:
                raise ValueError("fine boundary face is not contained in one skeleton subtriangle")
            segment = int(candidates[0])
            if np.min(face_bary @ np.linalg.inv(partitions[segment])) < -1e-10:
                raise ValueError("skeleton partition must align with the tetrahedral boundary mesh")
            bary = np.zeros((len(weights), 4))
            for j, node in enumerate(ids):
                bary[:, np.flatnonzero(fine.cells[local] == node)[0]] = bary_face[:, j]
            opposite = int(np.flatnonzero(fine.cell_faces[local] == fine_face)[0])
            values = tetra_face_basis(degree, bary, opposite_vertex=opposite)
            if skeleton.degrees[face] == 0:
                matrix[dofs[local], offset + segment] += (
                    coarse.signs[cell, side] * fine.areas[fine_face] * (weights @ values)
                )
            else:
                trace_basis = skeleton.basis(
                    int(face), (bary_face @ face_bary) @ np.linalg.inv(partitions[segment])
                )
                block = (
                    coarse.signs[cell, side]
                    * fine.areas[fine_face]
                    * np.einsum("q,qi,qj->ij", weights, values, trace_basis)
                )
                modes = int(skeleton.modes[face])
                matrix[np.ix_(dofs[local], offset + segment * modes + np.arange(modes))] += block
        offset += len(partitions) * int(skeleton.modes[face])
    return matrix


def _boundary(
    skeleton: TriangularSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate weak pressure moments and project prescribed outward normal fluxes."""
    mesh = skeleton.mesh
    if any(face not in mesh.boundary_faces for face in neumann):
        raise ValueError("Neumann keys must identify external triangular faces")
    load = np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    bary, weights = triangle_quadrature(order)
    for face in mesh.boundary_faces:
        partitions = skeleton.face_partition(int(face))
        subvertices = partitions @ mesh.points[mesh.faces[face]]
        points = np.einsum("qi,sij->sqj", bary, subvertices)
        data = scalar_values_3d(neumann.get(int(face), dirichlet), points.reshape(-1, 3)).reshape(
            len(partitions), -1
        )
        if skeleton.degrees[face] == 0:
            moments = data @ weights
            if face in neumann:
                fixed.update(zip(skeleton.dofs(int(face)), moments, strict=True))
            else:
                load[skeleton.dofs(int(face))] = (
                    mesh.areas[face] * skeleton.face_weights(int(face)) * moments
                )
            continue
        basis = skeleton.basis(int(face), bary)
        moments = np.einsum("sq,q,qi->si", data, weights, basis)
        if face in neumann:
            mass = np.einsum("q,qi,qj->ij", weights, basis, basis)
            projection = np.linalg.solve(mass, moments.T).T
            fixed.update(zip(skeleton.dofs(int(face)), projection.ravel(), strict=True))
        else:
            load[skeleton.dofs(int(face))] = (
                mesh.areas[face] * skeleton.face_weights(int(face))[:, None] * moments
            ).ravel()
    return load, fixed
