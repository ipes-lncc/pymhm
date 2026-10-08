"""Basix RT spaces with declared tensor moments and contravariant Piola maps.

Normal DOFs are integral tensor-Legendre moments. Scalar Qk pressure uses an
ordinary pullback; divergence carries the inverse geometric determinant.
"""

from functools import lru_cache
from itertools import product

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.hdiv.reference import vector_tabulation
from pymhm.fem.reference import (
    ReferenceElementSpec,
    create_reference_element,
    interpolate_reference,
    legendre_values,
    reference_interpolation_points,
)
from pymhm.meshes.hexahedron import HexMesh, _points


def tensor_legendre_values(degree: int, points: FloatArray) -> FloatArray:
    """Evaluate tensor Legendre polynomials, with the last coordinate varying fastest."""
    tables = [legendre_values(2 * points[:, a] - 1, degree) for a in range(points.shape[1])]
    return np.column_stack(
        [
            np.prod([tables[a][:, index[a]] for a in range(len(tables))], axis=0)
            for index in product(range(degree + 1), repeat=len(tables))
        ]
    )


def mapped_rt_dofs(mesh: HexMesh, degree: int) -> IntArray:
    """Map shared integral normal moments and independent cell-interior bubbles."""
    k = positive_int(degree, "RT degree", 0)
    count, interior = (k + 1) ** 2, 3 * k * (k + 1) ** 2
    face = (count * mesh.cell_faces[:, :, None] + np.arange(count)).reshape(len(mesh.cells), -1)
    inner = (
        count * len(mesh.faces)
        + interior * np.arange(len(mesh.cells))[:, None]
        + np.arange(interior)
    )
    return np.column_stack((face, inner))


@lru_cache(maxsize=8)
def mapped_reference_coefficients(degree: int) -> FloatArray:
    """Express declared hexahedral face lifts and bubble coefficients in native RT."""
    k = degree
    element = create_reference_element(
        ReferenceElementSpec("RT", "hexahedron", k + 1, lagrange_variant="legendre")
    )
    points = reference_interpolation_points(element)
    count = (k + 1) ** 2
    values = np.zeros((len(points), 3 * (k + 2) * count, 3))
    normalizers = np.array([(2 * i + 1) * (2 * j + 1) for i, j in product(range(k + 1), repeat=2)])
    for side in range(6):
        axis, end = divmod(side, 2)
        face = tensor_legendre_values(k, points[:, np.arange(3) != axis]) * normalizers
        indices = slice(side * count, (side + 1) * count)
        values[:, indices, axis] = (points[:, axis] - (1 - end))[:, None] * face
    modal = tensor_legendre_values(k, points)
    offset = 6 * count
    for axis in range(3):
        for index in product(*(range(k) if a == axis else range(k + 1) for a in range(3))):
            column = index[0] * (k + 1) ** 2 + index[1] * (k + 1) + index[2]
            t = points[:, axis]
            values[:, offset, axis] = t * (1 - t) * modal[:, column]
            offset += 1
    result = interpolate_reference(element, values)
    result.setflags(write=False)
    return result


def mapped_rt_basis(
    mesh: HexMesh, degree: int, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tabulate oriented physical RT_k vectors, their exact Piola divergence and scalar Q_k."""
    k = positive_int(degree, "RT degree", 0)
    points = _points(points)
    _, jacobian, determinant = mesh.geometry(points)
    count = (k + 1) ** 2
    width = 3 * (k + 2) * count
    native, native_divergence = vector_tabulation("RT", "hexahedron", k + 1, points)
    transform = mapped_reference_coefficients(k)
    local = np.einsum("qia,ij->qja", native, transform)
    local_divergence = native_divergence @ transform
    reference = np.broadcast_to(local, (len(mesh.cells), len(points), width, 3)).copy()
    divergence = np.broadcast_to(local_divergence, reference.shape[:-1]).copy()
    exponents = np.array(list(product(range(k + 1), repeat=2)))
    for side in range(6):
        linear = mesh.face_transforms[:, side, 1:]
        local_exponents = np.einsum("tab,ib->tia", np.abs(linear), exponents).astype(int)
        permutation = local_exponents[..., 0] * (k + 1) + local_exponents[..., 1]
        phase = np.prod(linear.sum(axis=1)[:, None, :] ** exponents[None], axis=-1)
        phase *= mesh.signs[:, side, None]
        indices = slice(side * count, (side + 1) * count)
        reference[:, :, indices] = (
            np.take_along_axis(reference[:, :, indices], permutation[:, None, :, None], axis=2)
            * phase[:, None, :, None]
        )
        divergence[:, :, indices] = (
            np.take_along_axis(divergence[:, :, indices], permutation[:, None, :], axis=2)
            * phase[:, None, :]
        )
    values = np.einsum("tqab,tqib->tqia", jacobian, reference) / determinant[:, :, None, None]
    return values, divergence / determinant[:, :, None], tensor_legendre_values(k, points)


_modal = tensor_legendre_values
_reference_rt_map = mapped_reference_coefficients
