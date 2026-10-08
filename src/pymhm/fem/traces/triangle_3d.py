"""Oriented triangular-face spaces, coupling and physical scalar boundary moments.

Trace coefficients represent densities in the macroface's canonical normal.
Bernstein coefficients are independent or shared within each physical macroface.
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


def _bernstein_exponents(degree: int) -> tuple[tuple[int, int, int], ...]:
    """Return the declared triangular Bernstein ordering, starting at vertex zero."""
    return tuple(
        (i, j, degree - i - j) for i in range(degree, -1, -1) for j in range(degree - i, -1, -1)
    )


def _partition_dofs(partition: FloatArray, degree: int, continuous: bool) -> IntArray:
    """Identify Bernstein coefficients by their vertex support and integer moments.

    Vertex identities use the exact reference coordinates used by partition
    validation. Sorting positive (vertex, exponent) pairs accounts for reversed
    subtriangle orientations without rounding coordinates or computing nullspaces.
    Continuous coefficients are numbered by sorted support keys; interior modes
    remain independent. Discontinuous numbering retains the subtriangle ordering.
    """
    exponents = _bernstein_exponents(degree)
    if not continuous:
        return np.arange(len(partition) * len(exponents)).reshape(len(partition), -1)
    _, vertices = np.unique(partition[..., 1:].reshape(-1, 2), axis=0, return_inverse=True)
    keys = [
        tuple(
            sorted((int(vertex), power) for vertex, power in zip(ids, powers, strict=True) if power)
        )
        for ids in vertices.reshape(-1, 3)
        for powers in exponents
    ]
    indices = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.array([indices[key] for key in keys]).reshape(len(partition), -1)


class TriangularSkeleton:
    """Piecewise Pk flux densities, discontinuous or C0 within each macroface.

    ``subdivisions`` counts uniform edge divisions, giving its square many
    subtriangles by default. Explicit ``face_partitions`` instead prescribe a
    conforming nonuniform triangulation of each original macroface. Each
    subtriangle has (k+1)(k+2)/2 local Bernstein modes. ``continuous=True`` shares
    vertex and edge coefficients inside a macroface and requires k >= 1.
    Sequences select independent face degrees and continuity flags. No degrees
    of freedom are shared between macrofaces, including their common edges.
    Flux uses the canonical normal of ``TetraMesh``.
    """

    def __init__(
        self,
        mesh: TetraMesh,
        subdivisions: Any = 1,
        *,
        degree: Any = 0,
        face_partitions: tuple[FloatArray, ...] | None = None,
        continuous: Any = False,
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
        flags = np.asarray(continuous)
        if flags.ndim == 0:
            flags = np.full(len(mesh.faces), flags)
        if flags.shape != (len(mesh.faces),) or flags.dtype.kind != "b":
            raise ValueError("continuous must be one boolean or one per macroface")
        self.continuous = flags.copy()
        if np.any(self.continuous & (self.degrees == 0)):
            raise ValueError("continuous triangular face spaces require degree >= 1")
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
        self._element_dofs = tuple(
            _partition_dofs(part, int(k), bool(flag))
            for part, k, flag in zip(
                self._face_partitions, self.degrees, self.continuous, strict=True
            )
        )
        for mapping in self._element_dofs:
            mapping.setflags(write=False)
        counts = np.array([int(mapping.max()) + 1 for mapping in self._element_dofs])
        self.offsets = np.r_[0, np.cumsum(counts)]
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

        return bernstein_tabulation(bary[:, 1:], _bernstein_exponents(degree))[0]

    def subtriangle_dofs(self, face: int, segment: int) -> IntArray:
        """Return global coefficients in the subtriangle's local Bernstein ordering.

        C0 vertex and edge coefficients are shared only within this macroface.
        The map transports local orientations to the declared canonical numbering.
        """
        self.dofs(face)
        segment = positive_int(segment, "segment", 0)
        if segment >= len(self._element_dofs[face]):
            raise ValueError("segment outside face partition")
        return self.offsets[face] + self._element_dofs[face][segment]

    def evaluate(self, face: int, barycentric: FloatArray) -> FloatArray:
        """Evaluate the complete macroface basis in original barycentric coordinates.

        Columns follow ``dofs(face)``. At partition interfaces the first incident
        subtriangle supplies the DG one-sided value; C0 values agree on both sides.
        """
        parts = self.face_partition(face)
        bary = np.asarray(barycentric)
        if (
            np.iscomplexobj(bary)
            or bary.ndim != 2
            or bary.shape[1] != 3
            or not np.isfinite(bary).all()
        ):
            raise ValueError("face barycentric points must be finite real triples")
        transformed = np.einsum("qi,sij->sqj", bary, np.linalg.inv(parts))
        valid = (np.min(transformed, axis=-1) >= -1e-10) & (
            abs(transformed.sum(axis=-1) - 1) <= 1e-10
        )
        if not np.all(np.any(valid, axis=0)):
            raise ValueError("conormal partition does not cover evaluation points")
        owners = np.argmax(valid, axis=0)
        output = np.zeros((len(bary), len(self.dofs(face))))
        for segment in np.unique(owners):
            rows = np.flatnonzero(owners == segment)
            columns = self.subtriangle_dofs(face, int(segment)) - self.offsets[face]
            output[np.ix_(rows, columns)] = self.basis(face, transformed[segment, rows])
        return output

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
    """Integrate signed nodal Pk traces against aligned triangular Bernstein modes."""
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
                columns = skeleton.subtriangle_dofs(int(face), segment) - skeleton.offsets[face]
                matrix[np.ix_(dofs[local], offset + columns)] += block
        offset += len(skeleton.dofs(int(face)))
    return matrix


def tetra_boundary_data(
    skeleton: TriangularSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate weak pressure moments and project prescribed outward normal fluxes.

    The quadrature floor integrates the Bernstein Gram matrix exactly even
    when the independently selected trace degree exceeds the local degree.
    Nonpolynomial boundary data can require a higher caller-supplied order.
    """
    mesh = skeleton.mesh
    if any(face not in mesh.boundary_faces for face in neumann):
        raise ValueError("Neumann keys must identify external triangular faces")
    load = np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    bary, weights = triangle_quadrature(max(order, int(skeleton.degrees.max()) + 1))
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
        if skeleton.continuous[face]:
            face_dofs = skeleton.dofs(int(face))
            rhs = np.zeros(len(face_dofs))
            mass = np.zeros((len(face_dofs), len(face_dofs)))
            local_mass = np.einsum("q,qi,qj->ij", weights, basis, basis)
            for segment, fraction in enumerate(skeleton.face_weights(int(face))):
                columns = skeleton.subtriangle_dofs(int(face), segment) - skeleton.offsets[face]
                rhs[columns] += fraction * moments[segment]
                mass[np.ix_(columns, columns)] += fraction * local_mass
            if face in neumann:
                fixed.update(zip(face_dofs, np.linalg.solve(mass, rhs), strict=True))
            else:
                load[face_dofs] = mesh.areas[face] * rhs
            continue
        if face in neumann:
            mass = np.einsum("q,qi,qj->ij", weights, basis, basis)
            projection = np.linalg.solve(mass, moments.T).T
            fixed.update(zip(skeleton.dofs(int(face)), projection.ravel(), strict=True))
        else:
            load[skeleton.dofs(int(face))] = (
                mesh.areas[face] * skeleton.face_weights(int(face))[:, None] * moments
            ).ravel()
    return load, fixed


_boundary = tetra_boundary_data
