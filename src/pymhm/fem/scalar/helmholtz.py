"""Original Pk/Qk Helmholtz forms, material cuts and exact realification."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.quadrature.material import (
    _material_edge_breaks,
    cartesian_trace_values,
    material_triangle_quadrature,
)
from pymhm.fem.quadrature.planar import planar_simplex_quadrature
from pymhm.fem.scalar.quadrilateral import (
    _cartesian_rectangle_quadrature,
    qk_basis,
    qk_space,
    quadrilateral_quadrature,
)
from pymhm.fem.scalar.triangle import element_tabulate, nodal_space
from pymhm.fem.traces.scalar import edge_basis, edge_pieces
from pymhm.linalg.complex import complexify_vector, realify_operator, realify_vector
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.planar import PlanarMaterial
from pymhm.meshes.triangle import TriangleMesh


def complex_values(value: Any, points: FloatArray) -> Any:
    """Evaluate scalar complex data, rejecting unintended broadcasting and NaNs."""
    result = np.asarray(value(points) if callable(value) else value, dtype=complex)
    if result.shape == ():
        result = np.full(len(points), result, dtype=complex)
    if result.shape != (len(points),) or not np.isfinite(result).all():
        raise ValueError("Helmholtz scalar data must be finite scalars or arrays with shape (n,)")
    return result


def positive_values(value: Any, points: FloatArray) -> FloatArray:
    """Evaluate scalar acoustic material, accepting isotropic planar tensor data."""
    if isinstance(value, PlanarMaterial):
        value = value(points)
    array = np.asarray(value) if not callable(value) else None
    if array is not None and array.shape == (len(points), 2, 2):
        if (
            np.any(array[:, 0, 1] != 0)
            or np.any(array[:, 1, 0] != 0)
            or np.any(array[:, 0, 0] != array[:, 1, 1])
        ):
            raise ValueError("acoustic density and bulk_modulus require isotropic planar data")
        value = array[:, 0, 0]
    result = complex_values(value, points)
    if np.any(result.imag != 0) or np.any(result.real <= 0):
        raise ValueError("density and bulk_modulus must be strictly positive and real")
    return np.asarray(result.real, dtype=float)


def stretch_values(value: Any, points: FloatArray) -> Any:
    """Evaluate outgoing PML stretches with positive real and nonnegative imaginary parts."""
    data = np.asarray(value(points) if callable(value) else value, dtype=complex)
    if data.shape == (2,):
        data = np.broadcast_to(data, (len(points), 2))
    if (
        data.shape != (len(points), 2)
        or not np.isfinite(data).all()
        or np.any(data.real <= 0)
        or np.any(data.imag < 0)
    ):
        raise ValueError(
            "PML stretches require shape (2,) or (n,2), "
            "positive real and nonnegative imaginary parts"
        )
    return data


def real_vector(value: Any) -> FloatArray:
    """Interleave the real and imaginary coordinates of a complex vector."""
    return realify_vector(value)


def complex_vector(value: Any) -> Any:
    """Decode the component-interleaved real coordinates without a sign change."""
    return complexify_vector(value)


def real_matrix(matrix: Any) -> Any:
    """Represent complex multiplication by [[Re,-Im],[Im,Re]] per scalar entry."""
    return realify_operator(matrix)


def acoustic_space(mesh: Any, degree: int) -> tuple[IntArray, FloatArray]:
    """Return continuous local nodal coordinates and element maps for Pk or Qk."""
    return nodal_space(mesh, degree) if isinstance(mesh, TriangleMesh) else qk_space(mesh, degree)


def acoustic_quadrature(
    mesh: Any, degree: int, material: Any, order: int
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray, Any]:
    """Tabulate physical local fields over all intersections with material pixels.

    Return element DOFs, physical points, physical weights, basis values,
    gradients and material samples. Separate terms can use separate material
    partitions, so density and modulus need not share a pixel grid.
    """
    if isinstance(mesh, TriangleMesh):
        bary, weights, data = material_triangle_quadrature(mesh, material, order)
        dofs, _, basis, gradient, _ = element_tabulate(mesh, degree, bary)
        points = np.einsum("tqi,tia->tqa", bary, mesh.points[mesh.cells])
    else:
        dofs, _ = qk_space(mesh, degree)
        origins = mesh.points[mesh.cells[:, 0]]
        if isinstance(material, CartesianCellField):
            reference, weights = _cartesian_rectangle_quadrature(
                origins, mesh.spacing, material, order
            )
        elif isinstance(material, PlanarMaterial):
            triangles = TriangleMesh(
                mesh.points, mesh.cells[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
            )
            bary, triangle_weights, data = planar_simplex_quadrature(triangles, material, order)
            points = np.einsum("tqi,tia->tqa", bary, triangles.points[triangles.cells])
            points = points.reshape(len(mesh.cells), -1, 2)
            reference = (points - origins[:, None, :]) / mesh.spacing
            weights = triangle_weights.reshape(len(mesh.cells), -1) / 2
        else:
            reference, weights = quadrilateral_quadrature(order)
            reference = np.broadcast_to(reference, (len(mesh.cells), *reference.shape))
            weights = np.broadcast_to(weights, reference.shape[:2])
        basis, gradient = qk_basis(degree, reference.reshape(-1, 2))
        basis = basis.reshape(*reference.shape[:2], -1)
        gradient = gradient.reshape(*basis.shape, 2) / mesh.spacing
        points = origins[:, None, :] + reference * mesh.spacing
        if not isinstance(material, PlanarMaterial):
            data = material
    return dofs, points, weights * mesh.areas[:, None], basis, gradient, data


def volume_forms(
    mesh: Any,
    degree: int,
    density: Any,
    modulus: Any,
    source: Any,
    order: int,
    pml_stretch: Any = None,
) -> tuple[Any, Any, Any]:
    """Assemble diffusion with rho^-1, mass with kappa^-1 and complex forcing."""
    dofs, nodes = acoustic_space(mesh, degree)
    width, count = dofs.shape[1], len(nodes)
    row = np.repeat(dofs, width, axis=1).ravel()
    column = np.tile(dofs, (1, width)).ravel()
    matrices = []
    for derivative, coefficient in ((True, density), (False, modulus)):
        _, points, weights, basis, gradient, data = acoustic_quadrature(
            mesh, degree, coefficient, order
        )
        samples = positive_values(data, points.reshape(-1, 2)).reshape(weights.shape)
        if pml_stretch is not None:
            stretch = stretch_values(pml_stretch, points.reshape(-1, 2)).reshape(*weights.shape, 2)
        if derivative:
            if pml_stretch is None:
                blocks = np.einsum("tq,tqia,tqja->tij", weights / samples, gradient, gradient)
            else:
                diagonal = stretch[..., ::-1] / stretch
                blocks = np.einsum(
                    "tq,tqa,tqia,tqja->tij", weights / samples, diagonal, gradient, gradient
                )
        else:
            factor = 1.0 if pml_stretch is None else np.prod(stretch, axis=-1)
            blocks = np.einsum("tq,tqi,tqj->tij", weights / samples * factor, basis, basis)
        matrices.append(
            sparse.coo_matrix((blocks.ravel(), (row, column)), shape=(count, count)).tocsc()
        )
    _, points, weights, basis, _, data = acoustic_quadrature(mesh, degree, source, order)
    samples = complex_values(data, points.reshape(-1, 2)).reshape(weights.shape)
    element_load = np.einsum("tq,tqi,tq->ti", weights, basis, samples)
    load = np.zeros(count, dtype=complex)
    np.add.at(load, dofs.ravel(), element_load.ravel())
    return matrices[0], matrices[1], load


def acoustic_edge_pieces(
    macro: Any, fine: Any, cell: int, face: int, degree: int
) -> Iterator[tuple[FloatArray, IntArray]]:
    """Yield parameter endpoints and local edge DOFs without crossing a fine edge."""
    if isinstance(fine, TriangleMesh):
        yield from edge_pieces(macro, fine, face, degree)
        return
    _, nodes = qk_space(fine, degree)
    nx, ny = fine.nx * degree, fine.ny * degree
    side = int(np.flatnonzero(macro.cell_faces[cell] == face)[0])
    edge_nodes = (
        np.arange(nx + 1),
        np.arange(ny + 1) * (nx + 1) + nx,
        ny * (nx + 1) + np.arange(nx + 1),
        np.arange(ny + 1) * (nx + 1),
    )[side]
    start, end = macro.points[macro.faces[face]]
    tangent = end - start
    for begin in range(0, len(edge_nodes) - 1, degree):
        ordered = edge_nodes[begin : begin + degree + 1]
        ids = ordered[np.r_[0, degree, np.arange(1, degree)]]
        parameter = (nodes[ids[:2]] - start) @ tangent / (tangent @ tangent)
        yield parameter, ids


def edge_rule(
    macro: Any,
    fine: Any,
    cell: int,
    face: int,
    degree: int,
    space: Any,
    order: int,
    materials: tuple[Any, ...],
) -> Iterator[tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]]:
    """Integrate on the common fine, skeletal and explicit material partition."""
    start, end = macro.points[macro.faces[face]]
    length = np.linalg.norm((end - start)[None, :], axis=1)[0]
    breaks = np.asarray(space.breaks)
    for material in materials:
        if isinstance(material, CartesianCellField):
            breaks = _material_edge_breaks(start, end, material, tuple(breaks))
        elif isinstance(material, PlanarMaterial):
            normals, offsets = material.planes
            slopes = normals @ (end - start)
            active = slopes != 0
            intersections = (offsets[active] - normals[active] @ start) / slopes[active]
            breaks = np.unique(
                np.r_[breaks, intersections[(intersections > 0) & (intersections < 1)]]
            )
    gauss, weights = leggauss(order)
    for positions, ids in acoustic_edge_pieces(macro, fine, cell, face, degree):
        low, high = sorted(positions)
        cuts = np.r_[low, breaks[(breaks > low) & (breaks < high)], high]
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            parameter = left + (gauss + 1) * (right - left) / 2
            local = (parameter - positions[0]) / (positions[1] - positions[0])
            yield (
                ids,
                parameter,
                start + parameter[:, None] * (end - start),
                weights * (right - left) / 2 * length,
                edge_basis(degree, local),
            )


def material_trace(material: Any, points: FloatArray, interior: FloatArray) -> FloatArray:
    """Evaluate acoustic material on the incident side of a boundary face."""
    data = (
        cartesian_trace_values(material, points, interior)
        if isinstance(material, CartesianCellField)
        else material
    )
    if isinstance(material, PlanarMaterial):
        data = material.trace_values(points, interior)
    return positive_values(data, points)
