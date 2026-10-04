"""Tensor-product Raviart--Thomas elements with independent interior enrichment.

On each rectangle, normal traces have degree k and interior vector functions
have RT order s=k+n. The pressure is Q_s. Face DOFs are integral Legendre
moments; cell DOFs are coefficients of zero-normal-trace polynomial bubbles.
The construction implements the face/interior separation used by the mixed
MHM family of Duran et al. (2019), without copying a reference implementation.
"""

from functools import cache
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.hdiv.reference import vector_tabulation
from pymhm.fem.reference import (
    ReferenceElementSpec,
    create_reference_element,
    interpolate_reference,
    legendre_values,
    reference_interpolation_points,
)
from pymhm.fem.scalar.quadrilateral import (
    _cartesian_rectangle_quadrature,
    _grid_resolves_material,
    quadrilateral_quadrature,
)
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values, tensor_values
from pymhm.meshes.cartesian import CartesianMacroMesh


@cache
def _reference_map(degree: int, enriched_degree: int) -> FloatArray:
    """Express declared face lifts and bubble coordinates in native rectangular RT."""
    k, s = degree, enriched_degree
    element = create_reference_element(
        ReferenceElementSpec("RT", "quadrilateral", s + 1, lagrange_variant="legendre")
    )
    points = reference_interpolation_points(element)
    x, y = points.T
    lx, ly = legendre_values(2 * x - 1, s), legendre_values(2 * y - 1, s)
    values = np.zeros((len(points), 4 * (k + 1) + 2 * s * (s + 1), 2))
    for edge in range(4):
        parameter = (x, y, 1 - x, 1 - y)[edge]
        moments = legendre_values(2 * parameter - 1, k) * (2 * np.arange(k + 1) + 1)
        indices = slice(edge * (k + 1), (edge + 1) * (k + 1))
        axis = 1 if edge in (0, 2) else 0
        values[:, indices, axis] = (y - 1, x, y, x - 1)[edge][:, None] * moments
    offset = 4 * (k + 1)
    for axis in range(2):
        for b in range(s + 1 if axis == 0 else s):
            for a in range(s if axis == 0 else s + 1):
                t = x if axis == 0 else y
                values[:, offset, axis] = t * (1 - t) * lx[:, a] * ly[:, b]
                offset += 1
    result = interpolate_reference(element, values)
    result.setflags(write=False)
    return result


def tensor_rt_basis(
    mesh: CartesianMacroMesh, degree: int, enrichment: int, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate globally oriented RT vectors, divergence and modal Q_s pressure.

    ``points`` are reference coordinates in [0,1]^2, either shared (q,2) or
    cell dependent (cells,q,2). Vector/divergence arrays
    include a leading cell axis. The pressure basis is cell independent and
    ordered with x degree varying fastest. The Piola transform preserves all
    normal moments even on anisotropic rectangles.
    """
    k = positive_int(degree, "RT degree", 0)
    s = k + positive_int(enrichment, "interior enrichment", 0)
    raw = np.asarray(points)
    if (
        np.iscomplexobj(raw)
        or raw.ndim not in (2, 3)
        or raw.shape[-1] != 2
        or (raw.ndim == 3 and raw.shape[0] != len(mesh.cells))
        or not np.isfinite(raw).all()
    ):
        raise ValueError("reference points must be finite real pairs")
    points = np.asarray(raw, dtype=float).reshape(-1, 2)
    x, y = points.T
    lx, ly = legendre_values(2 * x - 1, s), legendre_values(2 * y - 1, s)
    width = 4 * (k + 1) + 2 * s * (s + 1)
    native, native_divergence = vector_tabulation("RT", "quadrilateral", s + 1, points)
    transform = _reference_map(k, s)
    values = np.einsum("qia,ij->qja", native, transform)
    divergence = native_divergence @ transform
    orientation = np.ones((len(mesh.cells), width))
    orientation[:, : 4 * (k + 1)] = (mesh.signs[:, :, None] ** np.arange(1, k + 2)).reshape(
        len(mesh.cells), -1
    )
    pressure = (ly[:, :, None] * lx[:, None, :]).reshape(len(points), -1)
    if raw.ndim == 3:
        values = values.reshape(len(mesh.cells), -1, width, 2)
        divergence = divergence.reshape(len(mesh.cells), -1, width)
        pressure = pressure.reshape(*raw.shape[:2], -1)
    else:
        values, divergence = values[None], divergence[None]
    physical = values * mesh.spacing[None, None, None, :] / np.prod(mesh.spacing)
    physical = physical * orientation[:, None, :, None]
    div = divergence * orientation[:, None, :] / np.prod(mesh.spacing)
    return physical, div, pressure


def tensor_rt_dofs(mesh: CartesianMacroMesh, degree: int, enrichment: int) -> IntArray:
    """Map conforming face moments and independent cell bubbles to global DOFs."""
    k = positive_int(degree, "RT degree", 0)
    s = k + positive_int(enrichment, "interior enrichment", 0)
    face = ((k + 1) * mesh.cell_faces[:, :, None] + np.arange(k + 1)).reshape(len(mesh.cells), -1)
    interior = (
        (k + 1) * len(mesh.faces)
        + 2 * s * (s + 1) * np.arange(len(mesh.cells))[:, None]
        + np.arange(2 * s * (s + 1))
    )
    return np.column_stack((face, interior))


def _trace_map(
    mesh: CartesianMacroMesh,
    cell: int,
    fine: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
) -> FloatArray:
    """Integrate each oriented macro flux against fine-edge Legendre moments."""
    x, w = leggauss(degree + 2)
    t = (x + 1) / 2
    width = sum(skeleton.faces[face].size for face in mesh.cell_faces[cell])
    mapping = np.zeros(((degree + 1) * len(fine.boundary_faces), width))
    offset = 0
    for side, face_id in enumerate(mesh.cell_faces[cell]):
        start, end = mesh.points[mesh.faces[face_id]]
        tangent = end - start
        for row, edge in enumerate(fine.boundary_faces):
            coordinates = fine.points[fine.faces[edge]]
            interval = (coordinates - start) @ tangent / (tangent @ tangent)
            if (
                not np.allclose(
                    coordinates, start + interval[:, None] * tangent, rtol=0, atol=1e-12
                )
                or min(interval) < -1e-12
                or max(interval) > 1 + 1e-12
            ):
                continue
            parameter = np.clip(interval[0] + t * (interval[1] - interval[0]), 0, 1)
            space = skeleton.faces[face_id]
            mapping[(degree + 1) * row : (degree + 1) * (row + 1), offset : offset + space.size] = (
                fine.lengths[edge]
                * mesh.signs[cell, side]
                * legendre_values(x, degree).T
                @ (w[:, None] / 2 * space.evaluate(parameter))
            )
        offset += skeleton.faces[face_id].size
    return mapping


def _material_rule(
    mesh: CartesianMacroMesh, permeability: Any, order: int
) -> tuple[FloatArray, FloatArray]:
    """Return cellwise normalized quadrature, resolving declared material interfaces."""
    if isinstance(permeability, CartesianCellField) and not _grid_resolves_material(
        mesh, permeability
    ):
        return _cartesian_rectangle_quadrature(
            mesh.points[mesh.cells[:, 0]], mesh.spacing, permeability, order
        )
    points, weights = quadrilateral_quadrature(order)
    return (
        np.broadcast_to(points, (len(mesh.cells), *points.shape)),
        np.broadcast_to(weights, (len(mesh.cells), len(weights))),
    )


def _operators(
    mesh: CartesianMacroMesh,
    degree: int,
    enrichment: int,
    permeability: Any,
    source: Any,
    order: int,
) -> tuple[Any, Any, FloatArray]:
    """Assemble mixed inverse-permeability mass and divergence moments."""
    points, weights = _material_rule(mesh, permeability, order)
    basis, div, pressure = tensor_rt_basis(mesh, degree, enrichment, points)
    physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
    inverse = np.linalg.inv(tensor_values(permeability, physical.reshape(-1, 2))).reshape(
        len(mesh.cells), -1, 2, 2
    )
    blocks = np.einsum("t,tq,tqia,tqab,tqjb->tij", mesh.areas, weights, basis, inverse, basis)
    dofs = tensor_rt_dofs(mesh, degree, enrichment)
    nq = int(dofs.max()) + 1
    npres = len(mesh.cells) * pressure.shape[-1]
    width = dofs.shape[1]
    mass = sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
        ),
        shape=(nq, nq),
    ).tocsc()
    blocks = np.einsum("t,tq,tqi,tqj->tij", mesh.areas, weights, pressure, div)
    pids = np.arange(npres).reshape(len(mesh.cells), -1)
    divergence = sparse.coo_matrix(
        (
            blocks.ravel(),
            (
                np.repeat(pids, width, axis=1).ravel(),
                np.tile(dofs, (1, pressure.shape[-1])).ravel(),
            ),
        ),
        shape=(npres, nq),
    ).tocsc()
    force = scalar_values(source, physical.reshape(-1, 2)).reshape(physical.shape[:2])
    load = np.einsum("t,tq,tqi,tq->ti", mesh.areas, weights, pressure, force).ravel()
    return mass, divergence, load
