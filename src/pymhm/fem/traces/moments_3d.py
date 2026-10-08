"""Physical face moment rules on triangular and polygonal macrofaces."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.polygon_3d import polygonal_face_quadrature
from pymhm.meshes.polyhedral import PolyhedralMesh


def face_moment_rule_3d(
    mesh: Any, skeleton: Any, face: int, order: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return physical moment rules, integrating the degree-2ell face mass exactly."""
    order = max(positive_int(order, "face quadrature order"), int(skeleton.degrees[face]) + 1)
    if isinstance(mesh, PolyhedralMesh):
        points, weights = polygonal_face_quadrature(mesh, face, order)
        return points, weights, np.asarray(skeleton.basis(face, points), dtype=np.float64)
    bary, weights = triangle_quadrature(order)
    partition = skeleton.face_partition(int(face))
    points = np.einsum("qi,sij->sqj", bary, partition @ mesh.points[mesh.faces[face]])
    if skeleton.continuous[face]:
        original_bary = np.einsum("qi,sij->sqj", bary, partition)
        basis = skeleton.evaluate(face, original_bary.reshape(-1, 3))
    else:
        values = skeleton.basis(face, bary)
        basis = np.kron(np.eye(len(partition)), values)
    return (
        points.reshape(-1, 3),
        (skeleton.face_weights(int(face))[:, None] * weights).ravel() * mesh.areas[face],
        np.asarray(basis, dtype=np.float64),
    )
