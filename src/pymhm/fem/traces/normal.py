"""Physical normal and representation checks for explicitly declared trace fields."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from pymhm.core.validation import real_array
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.forms import trace_quadrature
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D, polygonal_face_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.materials.evaluation import vector_values


def normal_component_zero(
    samples: Any,
    normal: Any,
    *,
    rtol: float = 64 * np.finfo(float).eps,
    pointwise: bool = False,
) -> bool:
    """Check sampled normal components against their declared cancellation scale.

    ``samples`` has shape (points, dimension), and ``normal`` has shape
    (dimension,). The comparison is ``abs(samples @ normal) <= rtol * scale``.
    The scale is the maximum uncancelled absolute dot product, or its value at
    each point when ``pointwise=True``. No absolute amplitude floor is added.
    This algebraic check does not certify a callback between sampling points.
    """
    values, direction = real_array(samples, "vector samples"), real_array(normal, "normal")
    if values.ndim != 2 or direction.shape != (values.shape[1],):
        raise ValueError("normal and vector samples must share their physical dimension")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("normal comparison rtol must be finite and nonnegative")
    terms = abs(values) @ abs(direction)
    scale = terms if pointwise else np.max(terms, initial=0.0)
    return bool(np.all(abs(values @ direction) <= rtol * scale))


def boundary_tangent_2d(
    space: SkeletonSpace, vector: Any, order: int = 8, faces: Iterable[int] | None = None
) -> bool:
    """Check sampled tangency on each declared edge partition in two dimensions.

    The owner's partition quadrature uses order at least eight. Each face uses
    the maximum uncancelled dot-product scale and relative tolerance 64*eps,
    without an absolute amplitude floor. This is a sampled compatibility check
    for a constant local kernel, not a continuum certificate for a callback.
    """
    if not isinstance(space, SkeletonSpace):
        raise TypeError("2D tangency requires a scalar edge trace mesh")
    selected = space.mesh.boundary_faces if faces is None else faces
    for face in selected:
        parameter, _ = space.faces[face].quadrature(max(order, 8))
        start, end = space.mesh.points[space.mesh.faces[face]]
        points = start + parameter[:, None] * (end - start)
        if not normal_component_zero(vector_values(vector, points), space.mesh.normals[face]):
            return False
    return True


def boundary_tangent_3d(
    space: TriangularSkeleton | PolygonalSkeleton3D,
    vector: Any,
    order: int = 8,
    *,
    faces: Iterable[int] | None = None,
) -> bool:
    """Check tangency on the canonical triangulation of original 3D faces.

    Triangular faces use one positive rule of order at least eight. Polygonal
    faces integrate through their geometry owner's triangles. Physical points
    use anchored affine evaluation to preserve coordinates constant on a face.
    Each face uses ``normal_component_zero`` with its maximum uncancelled scale
    and relative tolerance ``64 * eps``. This preserves the declared sampled
    compatibility convention; no unsampled continuum condition is certified.
    """
    if not isinstance(space, (TriangularSkeleton, PolygonalSkeleton3D)):
        raise TypeError("3D tangency requires a triangular or polygonal trace mesh")
    mesh = space.mesh
    selected = mesh.boundary_faces if faces is None else faces
    bary, _ = triangle_quadrature(max(8, order))
    for face in selected:
        if isinstance(space, PolygonalSkeleton3D):
            points, _ = polygonal_face_quadrature(space.mesh, int(face), max(8, order))
        else:
            vertices = mesh.points[mesh.faces[face]]
            points = vertices[0] + bary[:, 1:] @ (vertices[1:] - vertices[0])
        values = real_array(vector(points) if callable(vector) else vector, "vector field")
        values = np.broadcast_to(values, points.shape)
        if not normal_component_zero(values, mesh.normals[face]):
            return False
    return True


def boundary_normal_zero(
    space: Any, vector: Any, *, order: int = 8, faces: Iterable[int] | None = None
) -> bool:
    """Check sampled vector dot normal zero relative to its uncancelled terms.

    Exterior face normals are the mesh's canonical outward normals. Quadrature
    resolves declared trace partitions. This is a floating-point compatibility
    check, not a certified continuum tangency statement for arbitrary callbacks.
    Exact zero data remain zero without absolute amplitude thresholds.
    """
    selected = space.mesh.boundary_faces if faces is None else faces
    for face in selected:
        _, points, _, _ = trace_quadrature(space, int(face), order=order)
        value = real_array(vector(points) if callable(vector) else vector, "vector field")
        try:
            value = np.broadcast_to(value, points.shape)
        except ValueError as error:
            raise ValueError("vector field must match physical points and dimension") from error
        normal = space.mesh.normals[face]
        if not normal_component_zero(value, normal, rtol=256 * np.finfo(float).eps, pointwise=True):
            return False
    return True


def trace_represents(space: Any, face: int, value: Any, *, order: int = 8) -> bool:
    """Check a scalar/vector field belongs to the declared trace at integration nodes.

    A weighted least-squares solve uses the literal physical value basis and
    positive quadrature, avoiding a squared Gram condition number. Unresolved
    basis rank returns false. Residuals are checked relative to uncancelled input
    and projected magnitudes. The declaration does not certify unsampled values.
    This operation is independent of a PDE, material, gauge and solver family.
    """
    _, points, weights, basis = trace_quadrature(space, face, order=order)
    data = real_array(value(points) if callable(value) else value, "trace value")
    if basis.shape[-1] == 1 and data.shape == (len(points),):
        data = data[:, None]
    try:
        data = np.broadcast_to(data, (len(points), basis.shape[-1]))
    except ValueError as error:
        raise ValueError("trace value must match the physical basis components") from error
    factor = np.sqrt(weights)
    matrix = (factor[:, None, None] * basis).transpose(0, 2, 1).reshape(-1, basis.shape[1])
    load = (factor[:, None] * data).ravel()
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, load, rcond=None)
    if rank != matrix.shape[1]:
        return False
    projected = np.einsum("qia,i->qa", basis, coefficients)
    scale = np.maximum(np.max(abs(data), axis=1), np.max(abs(projected), axis=1))
    return bool(np.all(np.max(abs(data - projected), axis=1) <= 256 * np.finfo(float).eps * scale))
