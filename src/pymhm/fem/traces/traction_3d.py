"""Public physical moment and volume integration owners; no solver dispatch."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.hdiv.family_3d import (
    face_polynomials,
    face_quadrature,
    face_shape,
)
from pymhm.fem.hdiv.mixed_3d import Mixed3DSkeleton
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.mixed import AffineMixedMesh


class TractionSkeleton3D:
    """Three interleaved Cartesian moments of -sigma n on each original triangular macroface.

    Every subtriangle carries an independent complete polynomial. Coefficients
    are physical integral moments, not nodal tractions; canonical normals point
    outward from the first neighboring macrocell.
    """

    def __init__(self, mesh: AffineMixedMesh, degree: int = 1, subdivisions: int = 1) -> None:
        """Create the vector trace from the shared oriented scalar moment partition."""
        if not isinstance(mesh, AffineMixedMesh) or mesh.kind != "tetrahedron":
            raise ValueError("mixed elasticity requires affine tetrahedral macrocells")
        self.scalar = Mixed3DSkeleton(mesh, degree, subdivisions)
        self.mesh, self.degree, self.subdivisions = (
            mesh,
            self.scalar.degree,
            self.scalar.subdivisions,
        )
        self.size = 3 * self.scalar.size

    def cell_dofs(self, cell: int) -> IntArray:
        """Return interleaved traction coordinates on one macrocell's original faces."""
        return (3 * self.scalar.cell_dofs(cell)[:, None] + np.arange(3)).ravel()


def traction_boundary_data(
    skeleton: TractionSkeleton3D, dirichlet: Any, traction: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float], float, float]:
    """Integrate displacement loads and physical outward traction moments, including volume flux."""
    mesh, scalar = skeleton.mesh, skeleton.scalar
    if not set(traction).issubset(set(mesh.boundary_faces)):
        raise ValueError("traction keys must identify exterior macrofaces")
    load, volume_flux, volume_scale = np.zeros(skeleton.size), 0.0, 0.0
    fixed: dict[int, float] = {}
    uv, w = face_quadrature(3, order)
    tests = face_polynomials(uv, 3, skeleton.degree)
    for face in mesh.boundary_faces:
        partition = scalar.partitions[face]
        for piece, ids in enumerate(partition.cells):
            canonical = face_shape(uv, 3) @ partition.nodes[ids]
            physical = face_shape(canonical, 3) @ mesh.points[mesh.faces[face]]
            _, det = partition.coordinates(piece, canonical)
            weights = w * det * partition.measure
            indices = scalar.offsets[face] + piece * partition.width + np.arange(partition.width)
            vector_ids = (3 * indices[:, None] + np.arange(3)).ravel()
            if face in traction:
                values = -tests.T @ (weights[:, None] * vector_values_3d(traction[face], physical))
                fixed.update(zip(vector_ids.tolist(), values.ravel().tolist(), strict=True))
            else:
                values = vector_values_3d(dirichlet, physical)
                moments = partition.evaluate(piece, canonical).T @ (weights[:, None] * values)
                load[vector_ids] = -moments.ravel()
                normal_moments = (tests.T @ weights)[:, None] * mesh.normals[face]
                volume_flux += float(np.sum(moments * normal_moments))
                volume_scale += float(np.sum(abs(moments * normal_moments)))
    return load, fixed, volume_flux, volume_scale
