"""Physical conformity checks for explicit tetrahedral submeshes and face partitions."""

from itertools import product
from typing import Any

import numpy as np

from pymhm.mesh import FloatArray, TriangleMesh, positive_int
from pymhm.refinement import validate_submesh
from pymhm.tetrahedral import TetraMesh, _real


def validate_face_partition(values: Any) -> tuple[FloatArray, FloatArray]:
    """Validate a conforming triangulation of one reference triangular macroface.

    Each vertex has three barycentric coordinates in the original macroface
    ordering. Subtriangle ordering is retained, including reversed vertex order.
    The returned positive weights are physical area fractions. Area equality
    alone is insufficient: every unmatched edge must lie on the original face
    boundary, which excludes holes and hanging interior edges.
    """
    triangles = _real(values, "face partitions")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("face partitions must have shape (nsub, 3, 3)")
    tolerance = 256 * np.finfo(float).eps
    if not np.allclose(triangles.sum(axis=2), 1, rtol=0, atol=tolerance):
        raise ValueError("face-partition barycentric coordinates must sum to one")
    points, cells = np.unique(triangles[..., 1:].reshape(-1, 2), axis=0, return_inverse=True)
    fine = TriangleMesh(points, cells.reshape(-1, 3))
    reference = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    validate_submesh(reference, 0, fine)
    weights = abs(np.linalg.det(triangles))
    triangles.setflags(write=False)
    weights.setflags(write=False)
    return triangles, weights


def _volume_roundoff(mesh: TetraMesh) -> float:
    """Bound determinant perturbations from coordinate subtraction by multilinearity."""
    vertices = mesh.points[mesh.cells]
    edges = vertices[:, 1:] - vertices[:, :1]
    norms = np.linalg.norm(edges, axis=2)
    errors = (
        8
        * np.finfo(float).eps
        * np.linalg.norm(abs(vertices[:, 1:]) + abs(vertices[:, :1]), axis=2)
    )
    perturbation = np.zeros(len(mesh.cells))
    for selector in product((False, True), repeat=3):
        if any(selector):
            perturbation += np.prod(np.where(selector, errors, norms), axis=1)
    return float(perturbation.sum() / 6)


def validate_tetra_submesh(coarse: TetraMesh, cell: int, fine: TetraMesh) -> None:
    """Require a conforming tetrahedral partition of exactly one macro tetrahedron.

    Check containment, physical volume and the complete exterior-face envelope.
    Combined with ``TetraMesh``'s oriented two-sided incidence, the exterior
    condition rejects holes and unmatched hanging internal faces. Coordinate
    roundoff is transported through the inverse macro map; volume uses a
    determinant perturbation bound rather than an absolute unit-scale floor.
    """
    cell = positive_int(cell, "cell", 0)
    if cell >= len(coarse.cells):
        raise ValueError("cell outside macro mesh")
    if not isinstance(fine, TetraMesh):
        raise ValueError("local meshes must be TetraMesh instances")
    vertices = coarse.points[coarse.cells[cell]]
    transform = (vertices[1:] - vertices[0]).T
    inverse = np.linalg.inv(transform)
    local = (fine.points - vertices[0]) @ inverse.T
    bary = np.column_stack((1 - local.sum(axis=1), local))
    tolerance = 256 * np.finfo(float).eps * max(1.0, np.linalg.cond(transform))
    epsilon = 8 * np.finfo(float).eps
    transform_error = epsilon * (abs(vertices[1:]) + abs(vertices[0])).T
    coordinate_error = epsilon * (abs(fine.points) + abs(vertices[0]))
    local_error = (coordinate_error + abs(local) @ transform_error.T) @ abs(inverse).T
    membership_tolerance = tolerance + np.column_stack((local_error.sum(axis=1), local_error))
    macro = TetraMesh(vertices, [[0, 1, 2, 3]])
    if np.any(bary < -membership_tolerance) or not np.isclose(
        fine.volumes.sum(),
        coarse.volumes[cell],
        rtol=tolerance,
        atol=_volume_roundoff(macro) + _volume_roundoff(fine),
    ):
        raise ValueError("local mesh must cover its macro tetrahedron exactly")
    boundary = fine.faces[fine.boundary_faces]
    if not np.all(
        np.any(np.all(abs(bary[boundary]) <= membership_tolerance[boundary], axis=1), axis=1)
    ):
        raise ValueError("local mesh has an unmatched interior triangular boundary")
