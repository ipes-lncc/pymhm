"""Basix tensor Qk tables and scalar operators on Cartesian rectangles.

Rows retain the declared x-fastest nodal ordering. Product quadratures have
unit-sum weights; physical cell measures are applied by the operators.
"""

from functools import lru_cache
from typing import Any, cast

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre, leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.reference import simplex_lagrange_basis, simplex_lagrange_tabulation
from pymhm.fem.scalar.operators import _scalar_diffusion_blocks
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values, tensor_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.roundoff import cartesian_coordinates, coordinate_difference


def quadrilateral_quadrature(order: int = 4) -> tuple[FloatArray, FloatArray]:
    """Return product Gauss points on [0,1]^2 and unit-sum integration weights."""
    points, weights = leggauss(positive_int(order, "quadrature order"))
    points, weights = (points + 1) / 2, weights / 2
    return (
        np.array([(x, y) for y in points for x in points]),
        np.array([wx * wy for wy in weights for wx in weights]),
    )


@lru_cache(maxsize=16)
def _cardinals(degree: int) -> tuple[FloatArray, ...]:
    """Export the reordered Basix interval basis in ascending power coordinates.

    This representation conversion supports coefficient archives; evaluation
    and derivatives use Basix directly. The orthonormal native polynomial j
    equals sqrt(2j+1) times the shifted conventional Legendre polynomial.
    """
    degree = positive_int(degree, "degree")
    coordinates = np.linspace(0, 1, degree + 1)
    basis = simplex_lagrange_basis(
        "interval", degree, nodes=np.column_stack((1 - coordinates, coordinates))
    )
    transformation = np.zeros((degree + 1, degree + 1))
    for j in range(degree + 1):
        coefficients = Legendre.basis(j, domain=[0, 1]).convert(kind=Polynomial).coef
        transformation[j, : len(coefficients)] = np.sqrt(2 * j + 1) * coefficients
    return tuple(np.asarray(row) for row in basis.basis_matrix @ transformation)


def _cardinal_values(degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Tabulate Basix interval Pk in increasing equispaced nodal order."""
    coordinates = np.linspace(0, 1, degree + 1)
    values, derivatives, _ = simplex_lagrange_tabulation(
        "interval",
        degree,
        np.column_stack((1 - points, points)),
        nodes=np.column_stack((1 - coordinates, coordinates)),
        nderiv=1,
    )
    return values, derivatives[..., 0]


def qk_basis(degree: int, points: Any) -> tuple[FloatArray, FloatArray]:
    """Evaluate equidistant Qk basis and reference gradients at unit-square points.

    Nodes and basis functions are ordered first in x, then in y. Basix
    interval factors supply values and derivatives, including at nodal points;
    their tensor product retains the historical coefficient order.
    """
    degree = positive_int(degree, "degree")
    if np.iscomplexobj(points):
        raise ValueError("reference points must be real")
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or np.any(points < -1e-12)
        or np.any(points > 1 + 1e-12)
    ):
        raise ValueError("reference points must be finite with shape (n,2) in [0,1]^2")
    x, dx = _cardinal_values(degree, points[:, 0])
    y, dy = _cardinal_values(degree, points[:, 1])
    basis = np.einsum("qi,qj->qij", y, x).reshape(len(points), (degree + 1) ** 2)
    gradient = np.stack((np.einsum("qi,qj->qij", y, dx), np.einsum("qi,qj->qij", dy, x)), axis=-1)
    return basis, gradient.reshape(len(points), (degree + 1) ** 2, 2)


def qk_space(mesh: CartesianMacroMesh, degree: int) -> tuple[IntArray, FloatArray]:
    """Return the shared nodal DOFs and coordinates of a continuous Qk space."""
    degree = positive_int(degree, "degree")
    nx, ny = mesh.nx * degree, cast(int, mesh.ny) * degree
    x0, x1, y0, y1 = mesh.bounds
    nodes = np.array(
        [(x, y) for y in np.linspace(y0, y1, ny + 1) for x in np.linspace(x0, x1, nx + 1)]
    )
    local = np.array([j * (nx + 1) + i for j in range(degree + 1) for i in range(degree + 1)])
    origin = np.array(
        [
            j * degree * (nx + 1) + i * degree
            for j in range(cast(int, mesh.ny))
            for i in range(mesh.nx)
        ]
    )
    return (origin[:, None] + local).astype(np.int64), nodes


def _grid_resolves_material(mesh: CartesianMacroMesh, field: CartesianCellField) -> bool:
    """Determine whether every material line coincides with a fine-grid line."""
    if len(field.spacing) != 2:
        raise ValueError("quadrilateral material integration requires a two-dimensional field")
    spacing = mesh.spacing
    intervals = np.asarray(field.spacing) / spacing
    bounds = np.asarray(mesh.bounds)
    offsets, offset_error = cartesian_coordinates(bounds[[0, 2]], np.asarray(field.origin), spacing)
    _, width_error = coordinate_difference(bounds[[1, 3]], bounds[[0, 2]])
    spacing_error = width_error / np.array([mesh.nx, mesh.ny])
    interval_error = abs(intervals) * spacing_error / (spacing - spacing_error)
    offset_error += abs(offsets) * spacing_error / (spacing - spacing_error)
    values = np.r_[intervals, offsets]
    tolerance = np.maximum(
        64 * np.finfo(float).eps * max(1.0, np.max(abs(values))),
        2 * np.r_[interval_error, offset_error],
    )
    return bool(np.all(abs(values - np.rint(values)) <= tolerance))


def _cartesian_rectangle_quadrature(
    origins: FloatArray,
    spacing: FloatArray,
    field: CartesianCellField,
    order: int,
) -> tuple[FloatArray, FloatArray]:
    """Integrate each rectangle separately on its exact intersections with pixels.

    Reference points have shape (cells, points, 2); weights have shape
    (cells, points) and sum to one on each rectangle. Zero-weight padding
    combines variable intersection counts without extrapolating the material.
    """
    if len(field.spacing) != 2:
        raise ValueError("quadrilateral material integration requires a two-dimensional field")
    pixel_spacing = np.asarray(field.spacing)
    pixel_origin = np.asarray(field.origin)
    shape = np.asarray(field.values.shape[:2])
    lower, lower_error = cartesian_coordinates(origins, pixel_origin, pixel_spacing)
    upper, upper_error = cartesian_coordinates(origins + spacing, pixel_origin, pixel_spacing)
    tolerance = np.maximum(
        64 * np.finfo(float).eps * shape, 2 * np.maximum(lower_error, upper_error)
    )
    if np.any(lower < -tolerance) or np.any(upper > shape + tolerance):
        raise ValueError("a fine rectangle lies outside the Cartesian material field")
    lower = np.where(
        np.isclose(lower, np.rint(lower), rtol=0, atol=tolerance), np.rint(lower), lower
    )
    upper = np.where(
        np.isclose(upper, np.rint(upper), rtol=0, atol=tolerance), np.rint(upper), upper
    )
    first = np.floor(lower).astype(int)
    last = np.ceil(upper).astype(int) - 1
    parts = np.max(last - first + 1, axis=0)
    gauss, weights = leggauss(positive_int(order, "quadrature order"))
    gauss, weights = (gauss + 1) / 2, weights / 2
    axes, axis_weights = [], []
    for axis in range(2):
        indices = first[:, axis, None] + np.arange(parts[axis])
        physical_lo = pixel_origin[axis] + indices * pixel_spacing[axis]
        physical_hi = pixel_origin[axis] + (indices + 1) * pixel_spacing[axis]
        end = origins[:, axis, None] + spacing[axis]
        lo = np.minimum(np.maximum(physical_lo, origins[:, axis, None]), end)
        hi = np.maximum(np.minimum(physical_hi, end), lo)
        a = (lo - origins[:, axis, None]) / spacing[axis]
        width = (hi - lo) / spacing[axis]
        axes.append(a[..., None] + width[..., None] * gauss)
        axis_weights.append(width[..., None] * weights)
    count = len(origins)
    x = np.broadcast_to(axes[0][:, None, None, :, :], (count, parts[1], order, parts[0], order))
    y = np.broadcast_to(axes[1][:, :, :, None, None], x.shape)
    combined_weights = axis_weights[1][:, :, :, None, None] * axis_weights[0][:, None, None, :, :]
    reference = np.stack((x, y), axis=-1).reshape(count, -1, 2)
    return np.clip(reference, 0, 1), combined_weights.reshape(count, -1)


def quadrilateral_operators(
    mesh: CartesianMacroMesh,
    degree: int = 1,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    order: int = 4,
) -> tuple[Any, Any, FloatArray]:
    """Assemble sparse Qk diffusion, mass and load using bounded cell batches.

    CartesianCellField coefficients are integrated over exact intersections
    with material pixels, retaining a uniform-rule fast path on aligned grids.
    Arbitrary callback discontinuities are not inferred: those require an
    aligned grid or a declared quadrature-sensitivity study.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "order"), degree + 1)
    reference, weights = quadrilateral_quadrature(order)
    dofs, nodes = qk_space(mesh, degree)
    basis, gradient = qk_basis(degree, reference)
    gradient = gradient / mesh.spacing
    area = float(np.prod(mesh.spacing))
    width = basis.shape[1]
    n = len(nodes)
    cuts = isinstance(permeability, CartesianCellField) and not _grid_resolves_material(
        mesh, permeability
    )
    blocks, loads = [], []
    for begin in range(0, len(mesh.cells), 256):
        origins = mesh.points[mesh.cells[begin : begin + 256, 0]]
        if cuts:
            cell_reference, cell_weights = _cartesian_rectangle_quadrature(
                origins, mesh.spacing, permeability, order
            )
            cell_basis, cell_gradient = qk_basis(degree, cell_reference.reshape(-1, 2))
            cell_basis = cell_basis.reshape(len(origins), -1, width)
            cell_gradient = cell_gradient.reshape(len(origins), -1, width, 2) / mesh.spacing
            points = origins[:, None, :] + cell_reference * mesh.spacing
            tensors = tensor_values(permeability, points.reshape(-1, 2)).reshape(
                len(origins), -1, 2, 2
            )
            force = scalar_values(source, points.reshape(-1, 2)).reshape(len(origins), -1)
            blocks.append(_scalar_diffusion_blocks(cell_weights, cell_gradient, tensors, area))
            loads.append(area * np.einsum("tq,tqi,tq->ti", cell_weights, cell_basis, force))
        else:
            points = origins[:, None, :] + reference[None, :, :] * mesh.spacing
            tensors = tensor_values(permeability, points.reshape(-1, 2)).reshape(
                len(origins), len(weights), 2, 2
            )
            force = scalar_values(source, points.reshape(-1, 2)).reshape(len(origins), len(weights))
            blocks.append(_scalar_diffusion_blocks(weights, gradient, tensors, area))
            loads.append(area * np.einsum("q,qi,tq->ti", weights, basis, force))
    row = np.repeat(dofs, width, axis=1).ravel()
    column = np.tile(dofs, (1, width)).ravel()
    matrix = sparse.coo_matrix(
        (np.concatenate(blocks).ravel(), (row, column)), shape=(n, n)
    ).tocsc()
    mass_block = area * np.einsum("q,qi,qj->ij", weights, basis, basis)
    mass = sparse.coo_matrix(
        (np.tile(mass_block.ravel(), len(dofs)), (row, column)), shape=(n, n)
    ).tocsc()
    load = np.bincount(dofs.ravel(), weights=np.concatenate(loads).ravel(), minlength=n)
    return matrix, mass, load


def quadrilateral_trace_coupling(
    coarse: CartesianMacroMesh,
    cell: int,
    fine: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
) -> FloatArray:
    """Integrate signed Qk traces, splitting at fine and skeleton breakpoints."""
    _, nodes = qk_space(fine, degree)
    width = sum(skeleton.faces[f].size for f in coarse.cell_faces[cell])
    matrix = np.zeros((len(nodes), width))
    offset = 0
    nx, ny = fine.nx * degree, cast(int, fine.ny) * degree
    # Coordinate-ordered boundary nodes; the global face parameter may reverse them.
    edge_nodes = (
        np.arange(nx + 1),
        np.arange(ny + 1) * (nx + 1) + nx,
        ny * (nx + 1) + np.arange(nx + 1),
        np.arange(ny + 1) * (nx + 1),
    )
    for side, face in enumerate(coarse.cell_faces[cell]):
        space = skeleton.faces[face]
        ids = edge_nodes[side]
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        length = coarse.lengths[face]
        parameter_nodes = (nodes[ids] - start) @ tangent / length**2
        gauss, w = leggauss(max(space.degrees) + degree + 1)
        for begin in range(0, len(ids) - 1, degree):
            indices = ids[begin : begin + degree + 1]
            t0, t1 = parameter_nodes[begin], parameter_nodes[begin + degree]
            lo, hi = sorted((float(t0), float(t1)))
            cuts = sorted({lo, hi, *(b for b in space.breaks if lo < b < hi)})
            for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                parameter = left + (gauss + 1) * (right - left) / 2
                s = (parameter - t0) / (t1 - t0)
                local_basis, _ = _cardinal_values(degree, s)
                weights = w * (right - left) / 2 * length * coarse.signs[cell, side]
                matrix[indices, offset : offset + space.size] += local_basis.T @ (
                    weights[:, None] * space.evaluate(parameter)
                )
        offset += space.size
    return matrix
