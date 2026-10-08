"""Physical scalar trace coordinates and boundary integrals on polygonal macrofaces.

Each original face owns one planar polynomial regardless of its triangulation.
Internal triangulation edges do not introduce global trace unknowns.
"""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.reference import monomial_tabulation
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_face_basis, tetra_nodal_space
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.meshes.geometry import triangle_in_face
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh


class PolygonalSkeleton3D:
    """Independent P0/P1 Robin-flux densities on original polygonal macrofaces.

    A planar P1 trace has exactly three coefficients regardless of the number
    of face vertices or triangles. Its basis is ``(1, s, t)`` in orthonormal
    tangential coordinates centered at the area centroid and scaled by sqrt(area).
    Internal triangulation edges introduce no skeletal degrees of freedom.
    """

    def __init__(self, mesh: PolyhedralMesh, degree: Any = 1) -> None:
        """Assign one degree or one independently selected P0/P1 degree per face."""
        if not isinstance(mesh, PolyhedralMesh):
            raise TypeError("polygonal skeleton requires PolyhedralMesh")
        raw = np.asarray(degree)
        if raw.ndim == 0:
            raw = np.full(len(mesh.faces), raw)
        if raw.shape != (len(mesh.faces),):
            raise ValueError("degree must be scalar or one value per polygonal face")
        self.degrees = np.array([positive_int(value, "degree", 0) for value in raw])
        if np.any(self.degrees > 1):
            raise ValueError("polygonal skeleton supports degree 0 or 1")
        self.mesh = mesh
        self.offsets = np.r_[0, np.cumsum(1 + 2 * self.degrees)]
        self.size = int(self.offsets[-1])
        self.constant_coefficients = np.zeros(self.size)
        self.constant_coefficients[self.offsets[:-1]] = 1

    def dofs(self, face: int) -> IntArray:
        """Return the coefficients belonging to one original polygonal face."""
        face = positive_int(face, "face", 0)
        if face >= len(self.mesh.faces):
            raise ValueError("face outside polygonal skeleton")
        return np.arange(self.offsets[face], self.offsets[face + 1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate the cell's complete original-face coefficient blocks."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cells):
            raise ValueError("cell outside mesh")
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])

    def basis(self, face: int, points: FloatArray) -> FloatArray:
        """Evaluate the single macroface polynomial in canonical physical coordinates."""
        self.dofs(face)
        points = np.asarray(points)
        if (
            np.iscomplexobj(points)
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("face evaluation points must be finite real XYZ points")
        if not self.degrees[face]:
            return np.ones((len(points), 1))
        local = (points - self.mesh.face_origins[face]) @ self.mesh.face_tangents[face].T
        return monomial_tabulation(
            local / np.sqrt(self.mesh.areas[face]), ((0, 0), (1, 0), (0, 1)), nderiv=0
        )[0]


def polygonal_trace_coupling(
    mesh: PolyhedralMesh, cell: int, fine: TetraMesh, skeleton: PolygonalSkeleton3D, degree: int
) -> FloatArray:
    """Integrate oriented Pk/P0–P1 coupling through the conforming fine boundary triangles."""
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied polyhedral mesh")
    dofs, nodes = tetra_nodal_space(fine, degree)
    matrix = np.zeros((len(nodes), len(skeleton.cell_dofs(cell))))
    bary_face, weights = triangle_quadrature(max(3, degree + 1))
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        for fine_face in fine.boundary_faces:
            ids = fine.faces[fine_face]
            corners = fine.points[ids]
            if not triangle_in_face(
                corners, mesh.points[mesh.face_triangles[face]], mesh.normals[face]
            ):
                continue
            element = int(fine.face_cells[fine_face, 0])
            bary = np.zeros((len(weights), 4))
            for j, node in enumerate(ids):
                bary[:, np.flatnonzero(fine.cells[element] == node)[0]] = bary_face[:, j]
            physical = corners[0] + bary_face[:, 1:] @ (corners[1:] - corners[0])
            opposite = int(np.flatnonzero(fine.cell_faces[element] == fine_face)[0])
            values = tetra_face_basis(degree, bary, opposite_vertex=opposite)
            trace = skeleton.basis(int(face), physical)
            block = (
                mesh.signs[cell][side]
                * fine.areas[fine_face]
                * np.einsum("q,qi,qj->ij", weights, values, trace)
            )
            matrix[np.ix_(dofs[element], offset + np.arange(trace.shape[1]))] += block
        offset += len(skeleton.dofs(int(face)))
    return matrix


def polygonal_face_quadrature(
    mesh: PolyhedralMesh, face: int, order: int
) -> tuple[FloatArray, FloatArray]:
    """Integrate the polygon by its canonical conforming triangulation with positive weights."""
    bary, weights = triangle_quadrature(order)
    corners = mesh.points[mesh.face_triangles[face]]
    area = (
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        / 2
    )
    return (
        (
            corners[:, :1] + np.einsum("qi,tia->tqa", bary[:, 1:], corners[:, 1:] - corners[:, :1])
        ).reshape(-1, 3),
        (area[:, None] * weights).ravel(),
    )


def polygonal_boundary_data(
    skeleton: PolygonalSkeleton3D, dirichlet: Any, natural: dict[int, Any], order: int
) -> tuple:
    """Form weak Dirichlet moments or L2-project outward Robin fluxes on original faces."""
    mesh = skeleton.mesh
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior polygonal faces")
    load = np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    for face in mesh.boundary_faces:
        points, weights = polygonal_face_quadrature(mesh, int(face), order)
        basis = skeleton.basis(int(face), points)
        data = scalar_values_3d(natural.get(int(face), dirichlet), points)
        moments = basis.T @ (weights * data)
        ids = skeleton.dofs(int(face))
        if face in natural:
            coefficients = np.linalg.solve(basis.T @ (weights[:, None] * basis), moments)
            fixed.update(zip(ids, coefficients, strict=True))
        else:
            load[ids] = moments
    return load, fixed
