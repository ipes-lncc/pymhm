"""Positive quadrature on intersections of triangular elements and material pixels."""

from typing import Any, Literal, overload

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.quadrature.planar import planar_simplex_quadrature
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.planar import PlanarMaterial
from pymhm.materials.sources import TriangleQuadratureField, triangle_field_quadrature
from pymhm.meshes.roundoff import area_coordinate_uncertainty, cartesian_coordinates
from pymhm.meshes.triangle import TriangleMesh


def _material_edge_breaks(
    start: FloatArray, end: FloatArray, field: CartesianCellField, existing: tuple[float, ...]
) -> FloatArray:
    """Merge pixel-edge intersections into an existing scalar partition."""
    tangent = end - start
    origin = np.asarray(field.origin)
    breaks = list(existing)
    unit = np.finfo(float).eps / 2
    for axis in range(2):
        if tangent[axis] != 0:
            indices = np.arange(1, field.values.shape[axis])
            offsets = field.spacing[axis] * indices
            levels = origin[axis] + offsets
            intersections = (levels - start[axis]) / tangent[axis]
            # Input quantization and two subtractions precede the division.
            # Keep accepted existing trace breakpoints as the representatives.
            level_error = (
                abs(np.spacing(origin[axis])) + indices * abs(np.spacing(field.spacing[axis]))
            ) / 2 + unit * (abs(offsets) + abs(levels)) / (1 - unit)
            numerator_error = level_error + abs(np.spacing(start[axis])) / 2
            numerator_error += unit * abs(levels - start[axis]) / (1 - unit)
            denominator_error = (
                abs(np.spacing(start[axis])) + abs(np.spacing(end[axis]))
            ) / 2 + unit * abs(tangent[axis]) / (1 - unit)
            internal = (intersections > 0) & (intersections < 1)
            if not np.any(internal):
                continue
            # A represented interior grid plane separates the two endpoints;
            # their subtraction uncertainty is then smaller than their distance.
            intersections = intersections[internal]
            numerator_error = numerator_error[internal]
            uncertainty = (numerator_error + abs(intersections) * denominator_error) / (
                abs(tangent[axis]) - denominator_error
            ) + unit * abs(intersections) / (1 - unit)
            tolerance = np.maximum(32 * np.finfo(float).eps, 2 * uncertainty)
            for parameter, envelope in zip(intersections, tolerance, strict=True):
                if min(abs(parameter - p) for p in breaks) > envelope:
                    breaks.append(float(parameter))
    return np.sort(breaks)


def cartesian_trace_values(
    field: CartesianCellField, points: Any, interior_points: Any
) -> FloatArray:
    """Evaluate exact pixel traces selected by incident-cell interior points.

    ``points`` has shape (n,2); ``interior_points`` has shape (2,) or (n,2).
    Coordinates on a pixel line use the side containing the corresponding
    interior point. Values elsewhere use the containing pixel. A direction
    tangent to an interface cannot define its one-sided value and is rejected.
    Coordinate comparisons propagate physical representation and transform roundoff;
    no sampling point is displaced.
    """
    if not isinstance(field, CartesianCellField) or len(field.spacing) != 2:
        raise ValueError("material traces require a planar CartesianCellField")
    raw, directions = np.asarray(points), np.asarray(interior_points)
    if (
        np.iscomplexobj(raw)
        or np.iscomplexobj(directions)
        or raw.ndim != 2
        or raw.shape[1] != 2
        or directions.shape not in ((2,), raw.shape)
        or not np.isfinite(raw).all()
        or not np.isfinite(directions).all()
    ):
        raise ValueError("trace points and interior points must be finite real coordinate arrays")
    locations = np.asarray(raw, dtype=float)
    interior = np.broadcast_to(np.asarray(directions, dtype=float), locations.shape)
    field(locations)
    field(interior)
    coordinates, error = cartesian_coordinates(
        locations, np.asarray(field.origin), np.asarray(field.spacing)
    )
    interior_coordinates, interior_error = cartesian_coordinates(
        interior, np.asarray(field.origin), np.asarray(field.spacing)
    )
    indices = np.floor(coordinates).astype(np.int64)
    shape = np.asarray(field.values.shape[:2])
    for axis in range(2):
        levels = np.rint(coordinates[:, axis])
        tolerance = np.maximum(32 * np.finfo(float).eps * max(1, shape[axis]), 2 * error[:, axis])
        on_line = abs(coordinates[:, axis] - levels) <= tolerance
        ambiguous_side = abs(interior_coordinates[:, axis] - levels) <= 2 * interior_error[:, axis]
        if np.any(on_line & ambiguous_side):
            raise ValueError("interior point must select a strict side of the material face")
        indices[on_line, axis] = levels[on_line].astype(int) - (
            interior_coordinates[on_line, axis] < levels[on_line]
        )
    if np.any(indices < 0) or np.any(indices >= shape):
        raise ValueError("selected material trace lies outside the field")
    return field.values[tuple(indices.T)]


def cartesian_edge_quadrature(
    start: Any,
    end: Any,
    field: CartesianCellField,
    order: int = 5,
    *,
    interior_point: Any,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate material traces from one specified side of a physical edge.

    Return physical points, nonnegative weights summing to one, and material
    values with a leading point axis. Multiply weights by the edge length.
    Integration splits at all Cartesian pixel intersections. On a face lying
    on a material line, the strictly interior point of the incident cell selects
    the side by its coordinate relative to that line. No coordinate perturbation
    or average of opposite material traces is used.
    """
    if not isinstance(field, CartesianCellField) or len(field.spacing) != 2:
        raise ValueError("edge quadrature requires a planar CartesianCellField")
    geometry = np.asarray([start, end, interior_point])
    if np.iscomplexobj(geometry) or geometry.shape != (3, 2) or not np.isfinite(geometry).all():
        raise ValueError("edge endpoints and interior point must be finite real pairs")
    start, end, interior = np.asarray(geometry, dtype=float)
    tangent = end - start
    if not np.any(tangent):
        raise ValueError("material edge must have positive length")
    field(geometry)
    breaks = _material_edge_breaks(start, end, field, (0.0, 1.0))
    gauss, gauss_weights = leggauss(positive_int(order, "quadrature order"))
    widths = np.diff(breaks)
    parameter = breaks[:-1, None] + widths[:, None] * (gauss + 1) / 2
    points = start + parameter[..., None] * tangent
    weights = widths[:, None] * gauss_weights / 2
    midpoint = start + ((breaks[:-1] + breaks[1:]) / 2)[:, None] * tangent
    values = cartesian_trace_values(field, midpoint, interior)
    return points.reshape(-1, 2), weights.ravel(), np.repeat(values, order, axis=0)


def material_triangle_quadrature(
    mesh: TriangleMesh, material: Any, order: int = 5
) -> tuple[FloatArray, FloatArray, Any]:
    """Return cellwise barycentric integration and unambiguously sampled material.

    Cartesian piecewise-constant materials are integrated on every positive-area
    intersection, and the last return value contains flattened material samples
    from pixel indices. A ``TriangleQuadratureField`` supplies its own positive
    rule and samples, including arbitrary tensor trailing axes. Such a provider
    may omit support where its field is exactly zero. For ordinary coefficients
    the common Gaussian rule is broadcast without copying and the coefficient
    itself is returned unchanged.
    Arrays have shapes ``(cells, points, 3)`` and ``(cells, points)``; weights
    must be multiplied by physical triangle areas. Whole-cell material rules
    sum to one per cell; compact-support field rules integrate only that support.
    """
    if isinstance(material, TriangleQuadratureField):
        return triangle_field_quadrature(mesh, material, order)
    if isinstance(material, PlanarMaterial):
        return planar_simplex_quadrature(mesh, material, order)
    if isinstance(material, CartesianCellField):
        if len(material.spacing) != 2:
            raise ValueError("triangle material quadrature requires a planar CartesianCellField")
        material(mesh.points)
        coordinates, error = cartesian_coordinates(
            mesh.points, np.asarray(material.origin), np.asarray(material.spacing)
        )
        vertices = coordinates[mesh.cells]
        shape = np.asarray(material.values.shape[:2])
        pixels = np.minimum(np.maximum(vertices.mean(axis=1), 0).astype(int), shape - 1)
        tolerance = np.maximum(
            32 * np.finfo(float).eps * np.maximum(shape, 1),
            2 * error[mesh.cells].max(axis=1),
        )
        if np.all(vertices.min(axis=1) >= pixels - tolerance) and np.all(
            vertices.max(axis=1) <= pixels + 1 + tolerance
        ):
            bary, weights = triangle_quadrature(order)
            values = material.values[tuple(pixels.T)]
            return (
                np.broadcast_to(bary, (len(mesh.cells), *bary.shape)),
                np.broadcast_to(weights, (len(mesh.cells), len(weights))),
                np.repeat(values, len(weights), axis=0),
            )
        bary, weights, pixels = cartesian_triangle_quadrature(
            mesh, material, order, return_cells=True
        )
        return bary, weights, material.values[tuple(pixels.reshape(-1, 2).T)]
    bary, weights = triangle_quadrature(order)
    return (
        np.broadcast_to(bary, (len(mesh.cells), *bary.shape)),
        np.broadcast_to(weights, (len(mesh.cells), len(weights))),
        material,
    )


def fit_material_faces(skeleton: SkeletonSpace, field: CartesianCellField) -> SkeletonSpace:
    """Split macroface segments at intersections with a Cartesian material grid.

    The macro and fine triangulations are unchanged. Every segment inherits its
    original degree; vector component count and continuity within each macroface
    are preserved. This fits the multiplier partition to material pixel edges,
    including boundaries between equal-valued pixels. It does not enrich local
    trial spaces or claim that an unfitted continuous local polynomial represents
    an exact solution with a gradient jump.
    """
    if not isinstance(field, CartesianCellField) or len(field.spacing) != 2:
        raise ValueError("face fitting requires a planar CartesianCellField")
    mesh = skeleton.mesh
    field(mesh.points)
    faces = []
    for vertices, space in zip(mesh.faces, skeleton.faces, strict=True):
        start, end = mesh.points[vertices]
        breaks = _material_edge_breaks(start, end, field, space.breaks)
        midpoints = (breaks[:-1] + breaks[1:]) / 2
        parents = np.searchsorted(space.breaks, midpoints, side="right") - 1
        faces.append(
            FaceSpace(
                tuple(breaks), tuple(space.degrees[int(i)] for i in parents), space.continuous
            )
        )
    return SkeletonSpace(mesh, tuple(faces), skeleton.components)


def fit_material_mesh(mesh: TriangleMesh, field: CartesianCellField) -> TriangleMesh:
    """Triangulate every original triangle/pixel intersection into a fitted mesh.

    Original edges and material interfaces remain mesh edges. Convex intersection
    polygons with more than three vertices use an interior centroid fan, retaining
    every boundary vertex and avoiding collinear fan triangles. Coincident vertices
    are identified within the propagated coordinate-roundoff envelope, including
    the grid-scaled arithmetic bound. Identification precedes triangulation. Sliver
    cells below the geometric distinguishability of ``TriangleMesh`` are rejected;
    no minimum-angle improvement or arbitrary-interface meshing is claimed.
    """
    if not isinstance(field, CartesianCellField) or len(field.spacing) != 2:
        raise ValueError("material fitting requires a planar CartesianCellField")
    field(mesh.points)
    shape = np.asarray(field.values.shape[:2])
    coordinates, error = cartesian_coordinates(
        mesh.points, np.asarray(field.origin), np.asarray(field.spacing)
    )
    tolerance = max(
        64 * np.finfo(float).eps * max(1.0, float(np.max(shape))),
        2 * float(np.max(error)),
    )
    levels = np.rint(coordinates)
    on_grid = abs(coordinates - levels) <= tolerance
    coordinates[on_grid] = levels[on_grid]
    parts = []
    for cell in mesh.cells:
        vertices = coordinates[cell]
        lower = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.floor(vertices.max(axis=0)).astype(int), shape - 1)
        for i in range(lower[0], upper[0] + 1):
            for j in range(lower[1], upper[1] + 1):
                polygon = vertices.copy()
                for axis, level, side in (
                    (0, i, True),
                    (0, i + 1, False),
                    (1, j, True),
                    (1, j + 1, False),
                ):
                    polygon = _clip_polygon(polygon, axis, level, side)
                    if len(polygon) < 3:
                        break
                if len(polygon) < 3:
                    continue
                keep = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) > tolerance
                polygon = polygon[keep]
                if len(polygon) < 3:
                    continue
                centroid = polygon.mean(axis=0)
                relative = polygon - centroid
                crosses = relative[:, 0] * np.roll(relative[:, 1], -1) - (
                    relative[:, 1] * np.roll(relative[:, 0], -1)
                )
                if crosses.sum() <= 0:
                    continue
                if len(polygon) == 3:
                    parts.append(polygon)
                else:
                    parts.extend(
                        np.array([centroid, polygon[k], polygon[(k + 1) % len(polygon)]])
                        for k in range(len(polygon))
                    )
    vertices = np.concatenate(parts)
    first, groups = _merge_grid_vertices(vertices, tolerance)
    physical = vertices[first] * np.asarray(field.spacing) + np.asarray(field.origin)
    result = TriangleMesh(physical, groups.reshape(-1, 3))
    # An area-only relative bound misses uncertainty of translated coordinates.
    displacement = tolerance * np.asarray(field.spacing)
    area_roundoff = area_coordinate_uncertainty(result, displacement)
    area_roundoff += area_coordinate_uncertainty(mesh, displacement)
    if abs(result.areas.sum() - mesh.areas.sum()) > area_roundoff:
        raise ValueError("material fitting failed to preserve domain area")
    return result


def _merge_grid_vertices(vertices: FloatArray, tolerance: float) -> tuple[IntArray, IntArray]:
    """Identify coincident grid vertices without extending the uncertainty by chains."""
    pairs = cKDTree(vertices).query_pairs(tolerance, output_type="ndarray")
    adjacency = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(vertices), len(vertices))
    )
    _, groups = connected_components(adjacency, directed=False)
    _, first = np.unique(groups, return_index=True)
    displacement = np.linalg.norm(vertices - vertices[first[groups]], axis=1)
    if np.any(displacement > tolerance * (1 + 8 * np.finfo(float).eps)):
        raise ValueError("vertex identification exceeds its coordinate-roundoff envelope")
    return first, groups


def _clip_polygon(polygon: FloatArray, axis: int, level: float, lower: bool) -> FloatArray:
    """Clip a counterclockwise convex polygon against one Cartesian half-plane."""
    vertices = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= level if lower else previous[axis] <= level
    for current in polygon:
        current_inside = current[axis] >= level if lower else current[axis] <= level
        if current_inside != previous_inside:
            fraction = (level - previous[axis]) / (current[axis] - previous[axis])
            intersection = previous + fraction * (current - previous)
            intersection[axis] = level
            vertices.append(intersection)
        if current_inside:
            vertices.append(current)
        previous, previous_inside = current, current_inside
    return np.asarray(vertices, dtype=float).reshape(-1, 2)


@overload
def cartesian_triangle_quadrature(
    mesh: TriangleMesh,
    field: CartesianCellField,
    order: int = 5,
    *,
    return_cells: Literal[False] = False,
) -> tuple[FloatArray, FloatArray]: ...


@overload
def cartesian_triangle_quadrature(
    mesh: TriangleMesh, field: CartesianCellField, order: int = 5, *, return_cells: Literal[True]
) -> tuple[FloatArray, FloatArray, IntArray]: ...


def cartesian_triangle_quadrature(
    mesh: TriangleMesh, field: CartesianCellField, order: int = 5, *, return_cells: bool = False
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, IntArray]:
    """Integrate each triangle separately over all positive-area material pixels.

    Return barycentric coordinates ``(cells, points, 3)`` and nonnegative,
    unit-sum weights ``(cells, points)``. Multiply weights by the triangle's
    physical area for integration. Each convex pixel intersection is triangulated
    and receives the ordinary Duffy-product Gauss rule of the requested order.
    Triangles with fewer points repeat their first point with zero padding weight.

    With ``return_cells=True``, also return pixel indices ``(cells, points, 2)``.
    Use these to retrieve discontinuous material values without rounding an
    interface-adjacent physical point into its neighboring pixel. Every material
    pixel with geometrically resolved positive intersection area is represented.
    Vertices within the propagated coordinate error of an integer grid line
    use that same line throughout material integration. This rule resolves
    coefficient jumps; it does not change the finite-element trial space.
    """
    if not isinstance(field, CartesianCellField) or len(field.spacing) != 2:
        raise ValueError("triangle cut quadrature requires a planar CartesianCellField")
    base, base_weights = triangle_quadrature(order)
    # Reuse the field's domain validation, including its roundoff-sized allowance.
    field(mesh.points)
    shape = np.asarray(field.values.shape[:2])
    coordinates, error = cartesian_coordinates(
        mesh.points, np.asarray(field.origin), np.asarray(field.spacing)
    )
    tolerance = np.maximum(32 * np.finfo(float).eps * np.maximum(shape, 1), 2 * error)
    levels = np.rint(coordinates)
    coordinates = np.where(abs(coordinates - levels) <= tolerance, levels, coordinates)
    coordinates = np.clip(coordinates, 0, shape)
    cell_bary, cell_weights, cell_pixels = [], [], []
    for cell in mesh.cells:
        vertices = coordinates[cell]
        lower = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.floor(vertices.max(axis=0)).astype(int), shape - 1)
        transform = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
        point_parts, weight_parts, pixel_parts = [], [], []
        for i in range(lower[0], upper[0] + 1):
            for j in range(lower[1], upper[1] + 1):
                polygon = vertices.copy()
                for axis, level, side in (
                    (0, i, True),
                    (0, i + 1, False),
                    (1, j, True),
                    (1, j + 1, False),
                ):
                    polygon = _clip_polygon(polygon, axis, level, side)
                    if len(polygon) < 3:
                        break
                for index in range(1, len(polygon) - 1):
                    subvertices = polygon[[0, index, index + 1]]
                    first, second = subvertices[1:] - subvertices[0]
                    double_area = first[0] * second[1] - first[1] * second[0]
                    if double_area <= 0:
                        continue
                    physical = base @ subvertices
                    local = np.linalg.solve(transform, (physical - vertices[0]).T).T
                    point_parts.append(np.column_stack((1 - local.sum(axis=1), local)))
                    weight_parts.append(base_weights * double_area)
                    pixel_parts.append(np.tile([i, j], (len(base), 1)))
        if not weight_parts:
            raise ValueError("triangle has no positive-area intersection with the material grid")
        weights = np.concatenate(weight_parts)
        cell_bary.append(np.concatenate(point_parts))
        cell_weights.append(weights / weights.sum())
        cell_pixels.append(np.concatenate(pixel_parts))
    count = max(map(len, cell_weights))
    bary = np.stack([np.tile(points[0], (count, 1)) for points in cell_bary])
    weights = np.zeros((len(mesh.cells), count))
    pixels = np.stack([np.tile(indices[0], (count, 1)) for indices in cell_pixels])
    for cell, (points, values, indices) in enumerate(
        zip(cell_bary, cell_weights, cell_pixels, strict=True)
    ):
        bary[cell, : len(points)] = points
        weights[cell, : len(points)] = values
        pixels[cell, : len(points)] = indices
    return (bary, weights, pixels) if return_cells else (bary, weights)
