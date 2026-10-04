"""Affine tetrahedral and prismatic meshes with canonical physical face moments."""

from functools import cache
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.hdiv.family_3d import (
    CellKind,
    HDiv3DFamily,
    face_polynomials,
    face_quadrature,
    face_shape,
    face_size,
    reference_faces,
    reference_vertices,
)
from pymhm.meshes.tetrahedron import TetraMesh, _reference_refinement
from pymhm.meshes.triangle import TriangleMesh


class AffineMixedMesh:
    """Conforming affine tetrahedra or prisms with one canonical normal per face.

    Prism corners are the triangle (0,1,2), followed by its translate (3,4,5).
    Nonaffine wedges are rejected: the pressure/divergence polynomial identity
    and this geometry contract both require a constant nonsingular Jacobian.
    Tetrahedral vertex orientation is normalized without changing geometry.
    """

    def __init__(self, points: Any, cells: Any, kind: CellKind = "tetrahedron") -> None:
        """Validate geometry, construct face incidence and retain exact affine maps."""
        reference = reference_vertices(kind)
        raw, ids = np.asarray(points), np.asarray(cells)
        if (
            np.iscomplexobj(raw)
            or raw.ndim != 2
            or raw.shape[1] != 3
            or not np.isfinite(raw).all()
            or ids.ndim != 2
            or ids.shape[1] != len(reference)
            or not len(ids)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= len(raw))
        ):
            raise ValueError("affine mixed mesh requires finite 3D points and valid integer cells")
        self.kind = kind
        self.points, self.cells = np.array(raw, dtype=float), np.array(ids, dtype=np.int64)
        if any(len(set(row)) != len(row) for row in self.cells):
            raise ValueError("cell vertices must be distinct")
        if len(np.unique(np.sort(self.cells, axis=1), axis=0)) != len(self.cells):
            raise ValueError("duplicate mixed cell")
        vertices = self.points[self.cells]
        last = 3
        jac = (vertices[:, [1, 2, last]] - vertices[:, :1]).transpose(0, 2, 1)
        determinant = np.linalg.det(jac)
        scale = np.prod(np.linalg.norm(jac, axis=1), axis=1)
        if np.any(abs(determinant) <= 32 * np.finfo(float).eps * scale):
            raise ValueError("degenerate mixed cell")
        negative = determinant < 0
        self.cells[negative, 1:3] = self.cells[negative, 2:0:-1]
        if kind == "prism":
            self.cells[negative, 4:6] = self.cells[negative, 5:3:-1]
        vertices = self.points[self.cells]
        self.jacobian = (vertices[:, [1, 2, 3]] - vertices[:, :1]).transpose(0, 2, 1)
        self.determinants = np.linalg.det(self.jacobian)
        self.inverse = np.linalg.inv(self.jacobian)
        reconstructed = vertices[:, :1] + np.einsum("tab,qb->tqa", self.jacobian, reference)
        if np.any(
            np.max(abs(reconstructed - vertices), axis=(1, 2))
            > 1e-12 * np.max(np.linalg.norm(self.jacobian, axis=1), axis=1)
        ):
            raise ValueError("prism vertices must define a single affine geometric map")
        facets: list[IntArray] = []
        owners: list[list[tuple[int, int]]] = []
        lookup: dict[tuple, int] = {}
        self._transforms: dict[HDiv3DFamily, FloatArray] = {}
        local_faces = reference_faces(kind)
        self.cell_faces = np.empty((len(ids), len(local_faces)), dtype=np.int64)
        for cell, nodes in enumerate(self.cells):
            for side, local in enumerate(local_faces):
                face_nodes = nodes[list(local)]
                key = tuple(sorted(face_nodes))
                if key not in lookup:
                    lookup[key] = len(facets)
                    facets.append(face_nodes.copy())
                    owners.append([])
                face = lookup[key]
                owners[face].append((cell, side))
                if len(owners[face]) > 2:
                    raise ValueError("nonmanifold mixed face")
                self.cell_faces[cell, side] = face
        self.faces = tuple(facets)
        self.incidence = tuple(tuple(row) for row in owners)
        self.boundary_faces = np.array([i for i, row in enumerate(owners) if len(row) == 1])
        self.normals = np.empty((len(facets), 3))
        self.measures = np.empty(len(facets))
        self.signs = np.ones(self.cell_faces.shape)
        centers = vertices.mean(axis=1)
        for face, nodes in enumerate(facets):
            x = self.points[nodes]
            normal = np.cross(x[1] - x[0], x[2] - x[0])
            self.measures[face] = np.linalg.norm(normal)
            normal /= self.measures[face]
            if normal @ (x.mean(axis=0) - centers[owners[face][0][0]]) < 0:
                normal *= -1
            self.normals[face] = normal
            for position, (cell, side) in enumerate(owners[face]):
                if (normal @ (x.mean(axis=0) - centers[cell]) > 0) != (position == 0):
                    raise ValueError("mixed cells overlap across a shared face")
                self.signs[cell, side] = 1 - 2 * position
        self.face_offsets = np.r_[0, np.cumsum([len(face) for face in self.faces])]
        self.volumes = self.determinants * (1 / 6 if kind == "tetrahedron" else 1 / 2)
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        for facet_nodes in self.faces:
            facet_nodes.setflags(write=False)

    def geometry(self, points: FloatArray) -> FloatArray:
        """Map common reference coordinates into every physical cell."""
        return self.points[self.cells[:, :1]] + np.einsum("tab,qb->tqa", self.jacobian, points)

    def face_coordinates(self, face: int, physical: FloatArray) -> FloatArray:
        """Return canonical affine coordinates on the specified triangular or rectangular face."""
        nodes = self.points[self.faces[face]]
        tangents = (nodes[1:3] - nodes[:1]).T
        return (physical - nodes[0]) @ np.linalg.pinv(tangents).T

    def submesh(self, cell: int, refinement: int) -> "AffineMixedMesh":
        """Uniformly refine one macrocell, preserving its affine boundary faces."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.cells):
            raise ValueError("cell index outside mesh")
        reference, cells = reference_submesh(self.kind, refinement)
        points = self.points[self.cells[cell, 0]] + reference @ self.jacobian[cell].T
        return AffineMixedMesh(points, cells, self.kind)

    def refined(self, refinement: int) -> "AffineMixedMesh":
        """Refine all cells, merging shared nodes by exact rational vertex weights."""
        reference, local_cells = reference_submesh(self.kind, refinement)
        if self.kind == "tetrahedron":
            shape = np.column_stack((1 - reference.sum(axis=1), reference))
        else:
            bary = np.column_stack((1 - reference[:, :2].sum(axis=1), reference[:, :2]))
            shape = np.column_stack(
                (bary * (1 - reference[:, 2, None]), bary * reference[:, 2, None])
            )
        integer = np.rint(shape * refinement**2).astype(np.int64)
        lookup: dict[tuple, int] = {}
        points: list[FloatArray] = []
        cells: list[IntArray] = []
        for vertices in self.cells:
            indices = []
            for weights in integer:
                key = tuple(
                    sorted(
                        (int(node), int(weight))
                        for node, weight in zip(vertices, weights, strict=True)
                        if weight
                    )
                )
                if key not in lookup:
                    lookup[key] = len(points)
                    points.append(weights @ self.points[vertices] / refinement**2)
                indices.append(lookup[key])
            cells.extend(np.asarray(indices)[local_cells])
        return AffineMixedMesh(np.asarray(points), np.asarray(cells), self.kind)

    @classmethod
    def unit_cube(cls, subdivisions: int = 1, kind: CellKind = "tetrahedron") -> "AffineMixedMesh":
        """Partition the unit cube into six tetrahedra or two prisms per Cartesian box."""
        n = positive_int(subdivisions, "subdivisions")
        if kind == "tetrahedron":
            mesh = TetraMesh.unit_cube(n)
            return cls(mesh.points, mesh.cells, kind)
        triangle = TriangleMesh.unit_square(n)
        count = len(triangle.points)
        points = np.vstack(
            [np.column_stack((triangle.points, np.full(count, z / n))) for z in range(n + 1)]
        )
        cells = np.vstack(
            [
                np.column_stack((triangle.cells + z * count, triangle.cells + (z + 1) * count))
                for z in range(n)
            ]
        )
        return cls(points, cells, kind)

    @classmethod
    def from_extruded_hexahedra(
        cls, points: Any, cells: Any, kind: CellKind = "prism"
    ) -> "AffineMixedMesh":
        """Split extruded hexahedra without changing their physical boundary.

        Eight corners follow lexicographic (x,y,z) order, as in HexMesh. Each
        planar base quadrilateral is divided along its 00--11 diagonal. Prisms
        are optionally split into three tetrahedra using globally ordered base
        vertices, so shared vertical-face diagonals remain conforming. Only
        genuine translational extrusions satisfy the affine geometry contract.
        """
        raw = np.asarray(cells)
        if raw.ndim != 2 or raw.shape[1] != 8 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("extruded hexahedra require integer eight-corner connectivity")
        prism = np.array(
            [
                row[list(indices)]
                for row in raw
                for indices in ((0, 4, 6, 1, 5, 7), (0, 6, 2, 1, 7, 3))
            ]
        )
        checked = cls(points, prism, "prism")
        if kind == "prism":
            return checked
        reference_vertices(kind)
        tetra: list[tuple[int, ...]] = []
        for row in checked.cells:
            order = np.argsort(row[:3])
            a, b, c = row[order]
            A, B, C = row[order + 3]
            tetra.extend(((a, b, c, C), (a, b, B, C), (a, A, B, C)))
        return cls(points, np.asarray(tetra), kind)


@cache
def reference_submesh(kind: CellKind, refinement: int) -> tuple[FloatArray, IntArray]:
    """Return a conforming r-cubed reference refinement and its cell connectivity."""
    r = positive_int(refinement, "refinement")
    if kind == "tetrahedron":
        if r & (r - 1):
            raise ValueError("tetrahedral refinement must be a power of two")
        bary, cells = _reference_refinement(r)
        mesh = TetraMesh(bary[:, 1:], cells)
        return mesh.points, mesh.cells
    reference_vertices(kind)
    triangle = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, r)
    count = len(triangle.points)
    points = np.vstack(
        [np.column_stack((triangle.points, np.full(count, z / r))) for z in range(r + 1)]
    )
    cells = np.vstack(
        [
            np.column_stack((triangle.cells + z * count, triangle.cells + (z + 1) * count))
            for z in range(r)
        ]
    )
    return points, cells


def hdiv3d_face_offsets(mesh: AffineMixedMesh, family: HDiv3DFamily) -> IntArray:
    """Return global face offsets for the family's declared normal order."""
    return np.r_[0, np.cumsum([face_size(len(face), family.normal_degree) for face in mesh.faces])]


def hdiv3d_dofs(mesh: AffineMixedMesh, family: HDiv3DFamily) -> IntArray:
    """Map canonical normal moments and private interior moments to global vector indices."""
    if mesh.kind != family.kind:
        raise ValueError("mesh and H(div) family must have the same cell kind")
    offsets = hdiv3d_face_offsets(mesh, family)
    face = np.column_stack(
        [
            offsets[mesh.cell_faces[:, side], None] + np.arange(count)
            for side, count in enumerate(family.face_sizes)
        ]
    )
    interior = (
        offsets[-1]
        + family.interior_size * np.arange(len(mesh.cells))[:, None]
        + np.arange(family.interior_size)
    )
    return np.column_stack((face, interior))


def hdiv3d_transform(
    mesh: AffineMixedMesh, family: HDiv3DFamily, *, coefficients: FloatArray | None = None
) -> FloatArray:
    """Construct polynomial moment transformations for face orientation and permutation.

    An explicitly supplied archived basis is used for replay and bypasses the
    mesh's cache of transformations for the current reference basis.
    """
    if coefficients is None and family in mesh._transforms:
        return mesh._transforms[family]
    transform = np.broadcast_to(
        np.eye(family.local_size), (len(mesh.cells), family.local_size, family.local_size)
    ).copy()
    reference = reference_vertices(mesh.kind)
    offset = 0
    for side, local in enumerate(reference_faces(mesh.kind)):
        count = family.face_sizes[side]
        uv, weights = face_quadrature(len(local), max(4, family.normal_degree + 2))
        nodes = reference[list(local)]
        points = face_shape(uv, len(local)) @ nodes
        values = family.tabulate(points, coefficients=coefficients)[0][:, offset : offset + count]
        normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        if normal @ (nodes.mean(axis=0) - reference.mean(axis=0)) < 0:
            normal *= -1
        normals = values @ normal
        for cell in range(len(mesh.cells)):
            physical = mesh.points[mesh.cells[cell, 0]] + points @ mesh.jacobian[cell].T
            face = mesh.cell_faces[cell, side]
            canonical = mesh.face_coordinates(face, physical)
            moment = (
                mesh.signs[cell, side]
                * face_polynomials(canonical, len(local), family.normal_degree).T
                @ (weights[:, None] * normals)
            )
            transform[cell, offset : offset + count, offset : offset + count] = np.linalg.inv(
                moment
            )
        offset += count
    transform.setflags(write=False)
    if coefficients is None:
        mesh._transforms[family] = transform
    return transform


def hdiv3d_basis(
    mesh: AffineMixedMesh,
    family: HDiv3DFamily,
    points: FloatArray,
    *,
    transform: FloatArray | None = None,
    coefficients: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Apply moment orientation and contravariant Piola to the reference family.

    ``coefficients`` replays an archived reference matrix in both basis
    evaluation and face transformations. If ``transform`` is also supplied,
    it must have been constructed with that same matrix.
    """
    hdiv3d_dofs(mesh, family)
    if transform is None:
        transform = hdiv3d_transform(mesh, family, coefficients=coefficients)
    values, div, pressure = family.tabulate(points, coefficients=coefficients)
    basis = np.einsum("tab,qib,tij->tqja", mesh.jacobian, values, transform, optimize=True)
    return (
        basis / mesh.determinants[:, None, None, None],
        np.einsum("qi,tij->tqj", div, transform) / mesh.determinants[:, None, None],
        pressure,
    )
