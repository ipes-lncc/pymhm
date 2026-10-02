"""Positive integration rules split at explicitly described planar material interfaces."""

from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.elements import triangle_quadrature
from pymhm.mesh import FloatArray, TriangleMesh, positive_int
from pymhm.planar_fitting import fit_planar_material
from pymhm.planar_material import PlanarMaterial
from pymhm.tetrahedral import TetraMesh, tetrahedron_quadrature


def planar_simplex_quadrature(
    mesh: TriangleMesh | TetraMesh, material: PlanarMaterial, order: int = 5
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return cellwise barycentric rules and flattened material tensor samples.

    Every material subregion receives a positive Gaussian rule. Weights sum to
    one in each original cell; multiply them by that cell's area or volume.
    Shapes are ``(cells, points, dimension + 1)``, ``(cells, points)`` and
    ``(cells * points, dimension, dimension)``. Different rule lengths are
    padded with zero weights at the original-cell centroid. Integration is
    fitted to interfaces without modifying the original approximation space.
    """
    order = positive_int(order, "quadrature order")
    fitted = fit_planar_material(mesh, material)
    rule, weights = (
        triangle_quadrature(order)
        if isinstance(mesh, TriangleMesh)
        else tetrahedron_quadrature(order)
    )
    measure = fitted.mesh.areas if isinstance(fitted.mesh, TriangleMesh) else fitted.mesh.volumes
    original = mesh.areas if isinstance(mesh, TriangleMesh) else mesh.volumes
    counts = np.bincount(fitted.parents, minlength=len(mesh.cells)) * len(weights)
    size = int(counts.max())
    dimension = material.dimension
    barycentric = np.full((len(mesh.cells), size, dimension + 1), 1 / (dimension + 1))
    normalized = np.zeros((len(mesh.cells), size))
    tensors = np.broadcast_to(
        material.background, (len(mesh.cells), size, dimension, dimension)
    ).copy()
    offset = np.zeros(len(mesh.cells), dtype=int)
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv(np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2))
    for cell, parent in enumerate(fitted.parents):
        physical = rule @ fitted.mesh.points[fitted.mesh.cells[cell]]
        coordinates = (physical - vertices[parent, 0]) @ inverse[parent].T
        segment = slice(offset[parent], offset[parent] + len(weights))
        barycentric[parent, segment, 0] = 1 - coordinates.sum(axis=1)
        barycentric[parent, segment, 1:] = coordinates
        normalized[parent, segment] = weights * measure[cell] / original[parent]
        tensors[parent, segment] = material.tensors[fitted.regions[cell]]
        offset[parent] += len(weights)
    return barycentric, normalized, tensors.reshape(-1, dimension, dimension)


def planar_edge_quadrature(
    start: Any,
    end: Any,
    material: PlanarMaterial,
    order: int = 5,
    *,
    interior_point: Any,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate exact incident-side tensor traces along a physical edge.

    Return physical points, positive weights summing to one and one tensor
    per point. Multiply weights by physical edge length. Each plane intersection
    splits the rule. The supplied incident-cell interior selects the side of
    an edge lying on a material interface; no sample coordinate is displaced.
    """
    order = positive_int(order, "quadrature order")
    geometry = np.asarray([start, end, interior_point])
    dimension = material.dimension
    if (
        np.iscomplexobj(geometry)
        or geometry.shape != (3, dimension)
        or not np.isfinite(geometry).all()
    ):
        raise ValueError("edge endpoints and interior point must be finite real coordinates")
    a, b, interior = np.asarray(geometry, dtype=float)
    tangent = b - a
    if not np.any(tangent):
        raise ValueError("material edge must have positive length")
    normals, offsets = material.planes
    slopes = normals @ tangent
    active = slopes != 0
    intersections = (offsets[active] - normals[active] @ a) / slopes[active]
    breaks = np.unique(np.r_[0.0, intersections[(intersections > 0) & (intersections < 1)], 1.0])
    widths = np.diff(breaks)
    gauss, gauss_weights = leggauss(order)
    parameters = breaks[:-1, None] + widths[:, None] * (gauss + 1) / 2
    points = a + parameters[..., None] * tangent
    midpoint = a + ((breaks[:-1] + breaks[1:]) / 2)[:, None] * tangent
    tensors = material.trace_values(midpoint, interior)
    return (
        points.reshape(-1, dimension),
        (widths[:, None] * gauss_weights / 2).ravel(),
        np.repeat(tensors, order, axis=0),
    )
