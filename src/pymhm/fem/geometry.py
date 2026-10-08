"""Physical point ownership and reference pullback for scalar and vector FEM fields."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, real_array
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.hexahedron import HexMesh, _geometry
from pymhm.meshes.polygonal import PolygonMesh


def volume_centroid(mesh: Any) -> FloatArray:
    """Integrate the geometric center of affine cells or triangulated polygons.

    Simplex and Cartesian cell coordinates integrate exactly by their vertex
    means and positive cell measures. Polygon cells use their actual interior
    triangle partition, so a concave boundary's vertex mean is never substituted
    for its area moment. Curved and trilinear cells require a quadrature-based
    geometric moment and are deliberately excluded from this affine operation.
    """
    if isinstance(mesh, HexMesh):
        raise TypeError("trilinear volume centroids require geometric quadrature")
    measures = mesh.volumes if hasattr(mesh, "volumes") else mesh.areas
    if isinstance(mesh, PolygonMesh):
        partitions = tuple(mesh.submesh(cell, 1) for cell in range(len(mesh.cells)))
        moment = sum(fine.areas @ fine.points[fine.cells].mean(axis=1) for fine in partitions)
    else:
        moment = measures @ mesh.points[mesh.cells].mean(axis=1)
    return np.asarray(moment / measures.sum(), dtype=float)


def pullback_points(
    mesh: Any, points: Any, *, cells: Any = None
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
    """Locate physical points and return owner, reference point, Jacobian and determinant.

    Affine simplex, rectangular, affine prismatic and trilinear hexahedral maps
    are supported. Trilinear coordinates use Newton iteration on the actual
    geometric map. Shared-interface evaluation chooses the first valid owner;
    explicit ``cells`` preserve independently chosen one-sided values.
    The Jacobian axes are (physical,reference), and det(J) is signed.
    """
    value = real_array(points, "physical points")
    dimension = mesh.points.shape[1]
    if value.ndim != 2 or value.shape[1] != dimension:
        raise ValueError("physical points must match the mesh dimension")
    if isinstance(mesh, CartesianMacroMesh):
        coordinates = (value - mesh.points[0]) / mesh.spacing
        counts = np.array([mesh.nx, mesh.ny])
        if np.any(coordinates < -1e-12) or np.any(coordinates > counts + 1e-12):
            raise ValueError("field points must belong to the declared mesh")
        if cells is None:
            # Choose the first valid x-fastest cell at shared interfaces.
            indices = np.clip(np.floor(coordinates - 1e-12).astype(int), 0, counts - 1)
            owners = indices[:, 0] + mesh.nx * indices[:, 1]
        else:
            owners = np.asarray(cells)
            if (
                owners.shape != (len(value),)
                or owners.dtype.kind not in "iu"
                or np.any(owners < 0)
                or np.any(owners >= len(mesh.cells))
            ):
                raise ValueError("cell owners must contain one valid integer per physical point")
            indices = np.column_stack((owners % mesh.nx, owners // mesh.nx))
        reference = coordinates - indices
        if np.any(reference < -1e-12) or np.any(reference > 1 + 1e-12):
            raise ValueError("points must belong to their declared one-sided cells")
        jacobian = np.broadcast_to(np.diag(mesh.spacing), (len(value), dimension, dimension))
        return owners.astype(np.int64), reference, jacobian, np.linalg.det(jacobian)
    vertices = mesh.points[mesh.cells]
    origins = vertices[:, 0]
    if isinstance(mesh, HexMesh):
        reference = np.empty((len(vertices), len(value), dimension))
        jacobians = np.empty((*reference.shape[:2], dimension, dimension))
        for cell, corners in enumerate(vertices):
            coordinates = np.full((len(value), dimension), 0.5)
            for _ in range(30):
                mapped, jacobian, _ = _geometry(corners[None], coordinates)
                step = np.linalg.solve(jacobian[0], (mapped[0] - value)[..., None])[..., 0]
                coordinates -= step
                if np.max(abs(step), initial=0) <= 16 * np.finfo(float).eps:
                    break
            mapped, jacobian, _ = _geometry(corners[None], coordinates)
            reference[cell], jacobians[cell] = coordinates, jacobian[0]
        valid = (reference.min(axis=-1) >= -1e-12) & (reference.max(axis=-1) <= 1 + 1e-12)
    else:
        if isinstance(mesh, CartesianMacroMesh):
            jacobian = np.broadcast_to(np.diag(mesh.spacing), (len(vertices), dimension, dimension))
        elif hasattr(mesh, "jacobian"):
            jacobian = mesh.jacobian
        else:
            jacobian = (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
        reference = np.einsum(
            "tij,tqj->tqi", np.linalg.inv(jacobian), value[None] - origins[:, None]
        )
        jacobians = np.broadcast_to(jacobian[:, None], (*reference.shape[:2], dimension, dimension))
        valid = reference.min(axis=-1) >= -1e-12
        if isinstance(mesh, CartesianMacroMesh):
            valid &= reference.max(axis=-1) <= 1 + 1e-12
        elif getattr(mesh, "kind", None) == "prism":
            valid &= (reference[..., :2].sum(axis=-1) <= 1 + 1e-12) & (
                reference[..., 2] <= 1 + 1e-12
            )
        else:
            valid &= reference.sum(axis=-1) <= 1 + 1e-12
    if cells is None:
        if not np.all(np.any(valid, axis=0)):
            raise ValueError("field points must belong to the declared mesh")
        owners = np.argmax(valid, axis=0)
    else:
        owners = np.asarray(cells)
        if (
            owners.shape != (len(value),)
            or owners.dtype.kind not in "iu"
            or np.any(owners < 0)
            or np.any(owners >= len(vertices))
        ):
            raise ValueError("cell owners must contain one valid integer per physical point")
        if not np.all(valid[owners, np.arange(len(value))]):
            raise ValueError("points must belong to their declared one-sided cells")
    jacobian = jacobians[owners, np.arange(len(value))]
    return (
        owners.astype(np.int64),
        reference[owners, np.arange(len(value))],
        jacobian,
        np.linalg.det(jacobian),
    )
