"""Continuous pressure traces and physical boundary quadrature on tetrahedra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_face_basis, tetra_nodal_space
from pymhm.fem.scalar.triangle import multiindices, reference_basis
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic


@dataclass(frozen=True)
class PressureTraceSpace3D:
    """Continuous nodal Pk functions on a uniformly triangulated tetrahedral skeleton.

    Each original triangular face has subdivisions**2 subtriangles.
    Polynomial degrees and subdivision counts are independent of the broken
    conormal space. Nodes on shared macroedges and vertices are identified by
    exact integer barycentric weights, never by rounded physical coordinates.
    The uniform face degree and subdivisions ensure conforming edge traces.
    """

    mesh: TetraMesh
    degree: int = 1
    subdivisions: int = 1
    nodes: FloatArray = field(init=False, repr=False)
    face_dofs: tuple[IntArray, ...] = field(init=False, repr=False)
    element_dofs: tuple[IntArray, ...] = field(init=False, repr=False)
    partitions: tuple[FloatArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct topological node identities on each canonical subtriangle."""
        if not isinstance(self.mesh, TetraMesh):
            raise TypeError("pressure traces require a tetrahedral mesh")
        degree = positive_int(self.degree, "degree")
        count = _dyadic(self.subdivisions, "subdivisions")
        divided = TriangularSkeleton(self.mesh, count)
        indices = multiindices(degree)
        lookup: dict[tuple[tuple[int, int], ...], int] = {}
        nodes: list[FloatArray] = []
        face_dofs: list[IntArray] = []
        element_dofs: list[IntArray] = []
        partitions: list[FloatArray] = []
        for face, vertices in enumerate(self.mesh.faces):
            partition = divided.face_partition(face)
            weights = np.rint(np.einsum("ki,sij->skj", indices, partition) * count).astype(int)
            ids = np.empty(weights.shape[:2], dtype=np.int64)
            for segment in range(len(partition)):
                for node, bary in enumerate(weights[segment]):
                    key = tuple(
                        sorted((int(v), int(w)) for v, w in zip(vertices, bary, strict=True) if w)
                    )
                    if key not in lookup:
                        lookup[key] = len(nodes)
                        nodes.append(bary @ self.mesh.points[vertices] / (degree * count))
                    ids[segment, node] = lookup[key]
            face_dofs.append(np.unique(ids))
            element_dofs.append(ids)
            partitions.append(partition)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "subdivisions", count)
        object.__setattr__(self, "nodes", np.asarray(nodes))
        object.__setattr__(self, "face_dofs", tuple(face_dofs))
        object.__setattr__(self, "element_dofs", tuple(element_dofs))
        object.__setattr__(self, "partitions", tuple(partitions))

    @property
    def size(self) -> int:
        """Return the global continuous pressure-trace dimension before boundary elimination."""
        return len(self.nodes)

    def cell_dofs(self, cell: int) -> IntArray:
        """Return the sorted pressure-trace nodes incident to one macrocell."""
        return np.unique(np.concatenate([self.face_dofs[f] for f in self.mesh.cell_faces[cell]]))

    def evaluate(self, face: int, bary: FloatArray) -> FloatArray:
        """Evaluate all face-local nodal functions at original-face barycentric points."""
        bary = np.asarray(bary)
        if (
            np.iscomplexobj(bary)
            or bary.ndim != 2
            or bary.shape[1] != 3
            or not np.isfinite(bary).all()
            or np.any(bary < -1e-12)
            or not np.allclose(bary.sum(axis=1), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("face barycentric points must be finite triples in the triangle")
        parts = self.partitions[face]
        transformed = np.einsum("qi,sij->sqj", bary, np.linalg.inv(parts))
        valid = np.min(transformed, axis=-1) >= -1e-12
        if not np.all(np.any(valid, axis=0)):
            raise ValueError("pressure-trace partition does not cover evaluation points")
        owners = np.argmax(valid, axis=0)
        output = np.zeros((len(bary), len(self.face_dofs[face])))
        for segment in np.unique(owners):
            rows = np.flatnonzero(owners == segment)
            values = reference_basis(self.degree, transformed[segment, rows])[0]
            columns = np.searchsorted(self.face_dofs[face], self.element_dofs[face][segment])
            output[np.ix_(rows, columns)] = values
        return output


def boundary_rules(
    coarse: TetraMesh, cell: int, fine: TetraMesh, degree: int, order: int
) -> tuple[tuple[int, IntArray, FloatArray, FloatArray, FloatArray, FloatArray], ...]:
    """Tabulate physical fine-boundary facets and their original macroface coordinates.

    Each tuple contains macroface index, incident volume DOFs, nodal basis,
    physical XYZ points, physical weights, and macroface barycentric coordinates.
    Fine tetrahedra must conform to the macrocell boundary.
    """
    dofs, _ = tetra_nodal_space(fine, degree)
    vertices = coarse.points[coarse.cells[cell]]
    inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
    coordinates = (fine.points - vertices[0]) @ inverse.T
    macro_bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
    bary, weights = triangle_quadrature(order)
    rules = []
    for side, face in enumerate(coarse.cell_faces[cell]):
        macro_vertices = [
            int(np.flatnonzero(coarse.cells[cell] == node)[0]) for node in coarse.faces[face]
        ]
        for fine_face in fine.boundary_faces:
            ids = fine.faces[fine_face]
            if np.max(abs(macro_bary[ids, side])) > 1e-10:
                continue
            owner = int(fine.face_cells[fine_face, 0])
            local_bary = np.zeros((len(weights), 4))
            for j, node in enumerate(ids):
                local_bary[:, np.flatnonzero(fine.cells[owner] == node)[0]] = bary[:, j]
            opposite = int(np.flatnonzero(fine.cell_faces[owner] == fine_face)[0])
            rules.append(
                (
                    int(face),
                    dofs[owner],
                    tetra_face_basis(degree, local_bary, opposite_vertex=opposite),
                    bary @ fine.points[ids],
                    weights * fine.areas[fine_face],
                    bary @ macro_bary[ids][:, macro_vertices],
                )
            )
    return tuple(rules)


def broken_face_basis(skeleton: TriangularSkeleton, face: int, bary: Any) -> FloatArray:
    """Evaluate DG or macroface-C0 conormal modes in the skeleton's coefficient order."""
    return skeleton.evaluate(face, bary)
