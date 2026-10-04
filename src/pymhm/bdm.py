"""Quadratic Brezzi–Douglas–Marini vectors on affine triangles.

The nine face degrees of freedom are integrals of the oriented normal
component against Legendre polynomials of degrees zero through two. Three
cell moments complete the element. Contravariant Piola transforms preserve
normal moments and divergence; odd edge moments include parameter reversal.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.element_backends import legendre_values
from pymhm.elements import triangle_quadrature
from pymhm.hdiv_reference import vector_tabulation
from pymhm.mesh import FloatArray, IntArray, SkeletonSpace, TriangleMesh


def _polynomials(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Tabulate native Basix BDM2 before applying the declared moment coordinates."""
    return vector_tabulation("BDM", "triangle", 2, points)


@lru_cache(maxsize=1)
def _dual_coefficients() -> FloatArray:
    """Construct the BDM2 dual basis using polynomial-exact moment integrals."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x, w = leggauss(4)
    t, w = (x + 1) / 2, w / 2
    dual = np.empty((12, 12))
    for edge in range(3):
        start, end = vertices[edge], vertices[(edge + 1) % 3]
        tangent = end - start
        normal_measure = np.array([tangent[1], -tangent[0]])
        values, _ = _polynomials(start + t[:, None] * tangent)
        dual[3 * edge : 3 * edge + 3] = legendre_values(x, 2).T @ (
            w[:, None] * (values @ normal_measure)
        )
    bary, weights = triangle_quadrature(4)
    points = bary[:, 1:]
    values, _ = _polynomials(points)
    tests = np.zeros((len(points), 3, 2))
    tests[:, 0, 0], tests[:, 1, 1] = 1, 1
    tests[:, 2, 0], tests[:, 2, 1] = -points[:, 1], points[:, 0]
    dual[9:] = np.einsum("q,qia,qja->ij", weights / 2, tests, values)
    return np.linalg.solve(dual, np.eye(12))


def bdm2_dofs(mesh: TriangleMesh) -> IntArray:
    """Return twelve conforming vector DOFs per triangle, with cell moments last."""
    edges = (3 * mesh.cell_faces[:, :, None] + np.arange(3)).reshape(-1, 9)
    interior = 3 * len(mesh.faces) + 3 * np.arange(len(mesh.cells))[:, None] + np.arange(3)
    return np.column_stack((edges, interior))


def bdm2_basis(mesh: TriangleMesh, barycentric: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate global-oriented BDM2 bases and divergences in every triangle.

    The outputs have shapes ``(cells, points, 12, 2)`` and
    ``(cells, points, 12)``. Barycentric coordinates have shape ``(points, 3)``
    or ``(cells, points, 3)`` for separate material-intersection rules.
    """
    if np.iscomplexobj(barycentric):
        raise ValueError("barycentric coordinates must be real")
    bary = np.asarray(barycentric, dtype=float)
    if (
        bary.ndim not in (2, 3)
        or bary.shape[-1] != 3
        or (bary.ndim == 3 and bary.shape[0] != len(mesh.cells))
        or not np.isfinite(bary).all()
        or not np.allclose(bary.sum(axis=-1), 1, atol=1e-13, rtol=0)
    ):
        raise ValueError("barycentric coordinates must be finite with shape (n,3) and sum one")
    polynomial, derivative = _polynomials(bary.reshape(-1, 3)[:, 1:])
    coefficients = _dual_coefficients()
    reference = np.einsum("qja,ji->qia", polynomial, coefficients)
    reference_div = derivative @ coefficients
    if bary.ndim == 2:
        reference = np.broadcast_to(reference, (len(mesh.cells), *reference.shape))
        reference_div = np.broadcast_to(reference_div, (len(mesh.cells), *reference_div.shape))
    else:
        reference = reference.reshape(*bary.shape[:2], 12, 2)
        reference_div = reference_div.reshape(*bary.shape[:2], 12)
    vertices = mesh.points[mesh.cells]
    jacobian = (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
    determinant = 2 * mesh.areas
    orientation = np.ones((len(mesh.cells), 12))
    orientation[:, :9] = (mesh.signs[:, :, None] ** np.arange(1, 4)).reshape(-1, 9)
    values = np.einsum("tab,tqib,ti,t->tqia", jacobian, reference, orientation, 1 / determinant)
    divergence = np.einsum("tqi,ti,t->tqi", reference_div, orientation, 1 / determinant)
    return values, divergence


def bdm2_evaluate(
    mesh: TriangleMesh, coefficients: FloatArray, barycentric: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate a BDM2 vector or row-wise tensor and its physical divergence.

    Coefficients have shape ``(3*faces+3*cells,)`` for a vector or
    ``(3*faces+3*cells, rows)`` for a row-wise tensor. Tensor output axes are
    ``(cell, point, row, column)``; divergence retains the row axis.
    """
    coefficients = np.asarray(coefficients)
    if (
        coefficients.ndim not in (1, 2)
        or coefficients.shape[0] != 3 * (len(mesh.faces) + len(mesh.cells))
        or np.iscomplexobj(coefficients)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError(
            "BDM2 coefficients must have a finite real vector or row-wise tensor shape"
        )
    values, divergence = bdm2_basis(mesh, barycentric)
    local = coefficients[bdm2_dofs(mesh)]
    return (
        np.einsum("tqia,ti...->tq...a", values, local),
        np.einsum("tqi,ti...->tq...", divergence, local),
    )


def validate_bdm2_trace(skeleton: SkeletonSpace, refinement: int) -> None:
    """Require a representable piecewise quadratic trace on aligned fine edges."""
    for face in skeleton.faces:
        if max(face.degrees) > 2 or not np.allclose(
            np.array(face.breaks) * refinement,
            np.round(np.array(face.breaks) * refinement),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("BDM2 requires degrees <=2 and trace segments aligned with fine edges")


def bdm2_trace_map(
    mesh: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
) -> FloatArray:
    """Map scalar signed macro normal traces to fine boundary BDM2 moments.

    The result has one row per boundary face and Legendre moment (degrees 0–2),
    and one column per scalar macroface basis. Vector traces reuse this matrix
    independently for each component. Breaks must align with fine edges, as
    checked by ``validate_bdm2_trace`` before local assembly.
    """
    count = sum(skeleton.faces[f].size for f in mesh.cell_faces[cell])
    result = np.zeros((3 * len(fine.boundary_faces), count))
    x, w = leggauss(4)
    parameter, weights = (x + 1) / 2, w / 2
    fine_basis = legendre_values(x, 2)
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        space = skeleton.faces[face]
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start
        for row, edge in enumerate(fine.boundary_faces):
            coordinates = fine.points[fine.faces[edge]]
            t = (coordinates - start) @ tangent / (tangent @ tangent)
            if (
                not np.allclose(coordinates, start + t[:, None] * tangent, atol=1e-12, rtol=0)
                or min(t) < -1e-12
                or max(t) > 1 + 1e-12
            ):
                continue
            macro_parameter = np.clip(t[0] + parameter * (t[1] - t[0]), 0, 1)
            result[3 * row : 3 * row + 3, offset : offset + space.size] = (
                fine.lengths[edge]
                * mesh.signs[cell, side]
                * fine_basis.T
                @ (weights[:, None] * space.evaluate(macro_parameter))
            )
        offset += space.size
    return result
