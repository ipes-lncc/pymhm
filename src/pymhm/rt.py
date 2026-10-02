"""Raviart–Thomas vectors and their canonical moments on affine triangles."""

from dataclasses import dataclass
from functools import cache, lru_cache
from math import factorial, prod
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from pymhm.elements import triangle_quadrature, vector_values
from pymhm.mesh import FloatArray, IntArray, TriangleMesh, positive_int


def rt_degree(degree: int) -> int:
    """Validate a nonnegative mathematical Raviart--Thomas order."""
    return positive_int(degree, "RT degree", 0)


def _powers(degree: int) -> tuple[tuple[int, int], ...]:
    """Enumerate scalar monomials by total degree, starting with the constant."""
    return tuple((i, total - i) for total in range(degree + 1) for i in range(total, -1, -1))


def rt_interior_tests(degree: int, points: FloatArray) -> FloatArray:
    """Evaluate canonical P_(m-1) interior tests: monomials through RT2, orthonormal above."""
    m = rt_degree(degree)
    if m >= 3:
        return _bernstein(m - 1, points)[0] @ _interior_transform(m)
    return (
        np.column_stack([points[:, 0] ** i * points[:, 1] ** j for i, j in _powers(m - 1)])
        if m
        else np.empty((len(points), 0))
    )


def _bernstein(degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray, tuple]:
    """Tabulate total-degree Bernstein functions and physical reference derivatives."""
    bary = np.column_stack((1 - points.sum(axis=1), points))
    exponents = tuple(
        (degree - i - j, i, j) for j in range(degree + 1) for i in range(degree + 1 - j)
    )

    def monomial(exponent: tuple[int, ...]) -> FloatArray:
        """Evaluate one normalized barycentric monomial, including derivative zeros."""
        if min(exponent) < 0:
            return np.zeros(len(points))
        multiplier = factorial(sum(exponent)) / prod(factorial(e) for e in exponent)
        return np.asarray(multiplier * np.prod(bary**exponent, axis=1), dtype=float)

    values = np.column_stack([monomial(e) for e in exponents])
    derivatives = np.stack(
        [
            np.column_stack(
                [
                    degree
                    * (
                        monomial(tuple(e[i] - (i == axis) for i in range(3)))
                        - monomial(tuple(e[i] - (i == 0) for i in range(3)))
                    )
                    for e in exponents
                ]
            )
            for axis in (1, 2)
        ],
        axis=-1,
    )
    return values, derivatives, exponents


@cache
def _interior_transform(degree: int) -> FloatArray:
    """Fix an orthonormal high-order cell-test basis by positive Cholesky diagonals."""
    bary, weights = triangle_quadrature(degree + 3)
    values = _bernstein(degree - 1, bary[:, 1:])[0]
    gram = values.T @ ((weights / 2)[:, None] * values)
    lower = np.linalg.cholesky(gram)
    return np.linalg.solve(lower, np.eye(len(lower))).T


def _bernstein_vectors(degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Span RTk with Bernstein Pk vectors and the homogeneous radial enrichment."""
    basis, gradients, exponents = _bernstein(degree, points)
    count = len(exponents)
    values = np.zeros((len(points), (degree + 1) * (degree + 3), 2))
    divergence = np.zeros(values.shape[:2])
    values[:, : 2 * count : 2, 0], values[:, 1 : 2 * count : 2, 1] = basis, basis
    divergence[:, : 2 * count : 2], divergence[:, 1 : 2 * count : 2] = (
        gradients[..., 0],
        gradients[..., 1],
    )
    homogeneous = basis[:, [i for i, exponent in enumerate(exponents) if exponent[0] == 0]]
    values[:, 2 * count :] = points[:, None, :] * homogeneous[..., None]
    divergence[:, 2 * count :] = (degree + 2) * homogeneous
    return values, divergence


def _polynomials(degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate [P_m]^2 plus x times homogeneous P_m and their divergences."""
    if degree >= 3:
        return _bernstein_vectors(degree, points)
    powers = _powers(degree)
    size = (degree + 1) * (degree + 3)
    values = np.zeros((len(points), size, 2))
    divergence = np.zeros((len(points), size))
    x, y = points.T
    for n, (i, j) in enumerate(powers):
        monomial = x**i * y**j
        values[:, 2 * n, 0], values[:, 2 * n + 1, 1] = monomial, monomial
        if i:
            divergence[:, 2 * n] = i * x ** (i - 1) * y**j
        if j:
            divergence[:, 2 * n + 1] = j * x**i * y ** (j - 1)
    for n, i in enumerate(range(degree, -1, -1), start=2 * len(powers)):
        monomial = x**i * y ** (degree - i)
        values[:, n] = points * monomial[:, None]
        divergence[:, n] = (degree + 2) * monomial
    return values, divergence


@lru_cache(maxsize=8)
def _dual_coefficients(degree: int) -> FloatArray:
    """Construct the canonical RT dual basis from exact reference moments."""
    size = (degree + 1) * (degree + 3)
    matrix = np.empty((size, size))
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x, w = leggauss(degree + 3)
    parameter = (x + 1) / 2
    for edge in range(3):
        start, end = vertices[edge], vertices[(edge + 1) % 3]
        tangent = end - start
        normal_measure = np.array([tangent[1], -tangent[0]])
        basis, _ = _polynomials(degree, start + parameter[:, None] * tangent)
        matrix[(degree + 1) * edge : (degree + 1) * (edge + 1)] = legvander(x, degree).T @ (
            w[:, None] / 2 * (basis @ normal_measure)
        )
    if degree:
        bary, weights = triangle_quadrature(degree + 3)
        points = bary[:, 1:]
        scalar = rt_interior_tests(degree, points)
        basis, _ = _polynomials(degree, points)
        matrix[3 * (degree + 1) :] = np.einsum("q,qi,qja->iaj", weights / 2, scalar, basis).reshape(
            -1, size
        )
    return np.linalg.solve(matrix, np.eye(size))


def rt_dofs(mesh: TriangleMesh, degree: int) -> IntArray:
    """Return oriented face moments followed by cell-interior moments per triangle."""
    return _cell_dofs(mesh, rt_degree(degree), np.arange(len(mesh.cells)))


def _cell_dofs(mesh: TriangleMesh, m: int, cells: IntArray) -> IntArray:
    """Return canonical indices only for requested cells, with repetition allowed."""
    face = ((m + 1) * mesh.cell_faces[cells, :, None] + np.arange(m + 1)).reshape(len(cells), -1)
    interior = (m + 1) * len(mesh.faces) + m * (m + 1) * cells[:, None] + np.arange(m * (m + 1))
    return np.column_stack((face, interior))


def rt_basis(
    mesh: TriangleMesh, degree: int, barycentric: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate globally oriented RT_m basis vectors and physical divergence.

    Mathematical order m corresponds to Basix/DOLFINx RT degree m+1. The
    outputs have shapes (cells, points, (m+1)(m+3), 2) and (cells, points, DOFs).
    Barycentric points may have shape (points,3) or (cells,points,3).
    """
    m = rt_degree(degree)
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
    polynomial, derivative = _polynomials(m, bary.reshape(-1, 3)[:, 1:])
    coefficients = _dual_coefficients(m)
    reference = np.einsum("qja,ji->qia", polynomial, coefficients)
    reference_divergence = derivative @ coefficients
    if bary.ndim == 2:
        reference = np.broadcast_to(reference, (len(mesh.cells), *reference.shape))
        reference_divergence = np.broadcast_to(
            reference_divergence, (len(mesh.cells), *reference_divergence.shape)
        )
    else:
        reference = reference.reshape(*bary.shape[:2], (m + 1) * (m + 3), 2)
        reference_divergence = reference_divergence.reshape(*bary.shape[:2], (m + 1) * (m + 3))
    vertices = mesh.points[mesh.cells]
    jacobian = (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
    orientation = np.ones((len(mesh.cells), (m + 1) * (m + 3)))
    orientation[:, : 3 * (m + 1)] = (mesh.signs[:, :, None] ** np.arange(1, m + 2)).reshape(
        len(mesh.cells), -1
    )
    values = np.einsum(
        "tab,tqib,ti,t->tqia", jacobian, reference, orientation, 1 / (2 * mesh.areas)
    )
    divergence = np.einsum("tqi,ti,t->tqi", reference_divergence, orientation, 1 / (2 * mesh.areas))
    return values, divergence


def rt_evaluate(
    mesh: TriangleMesh, coefficients: FloatArray, degree: int, barycentric: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate an RT vector field and its divergence in every fine triangle."""
    m = rt_degree(degree)
    size = (m + 1) * len(mesh.faces) + m * (m + 1) * len(mesh.cells)
    coefficients = np.asarray(coefficients)
    if (
        coefficients.shape != (size,)
        or np.iscomplexobj(coefficients)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("RT coefficients must be a finite real vector of the expected size")
    values, divergence = rt_basis(mesh, m, barycentric)
    local = coefficients[rt_dofs(mesh, m)]
    return np.einsum("tqia,ti->tqa", values, local), np.einsum("tqi,ti->tq", divergence, local)


def _validated_coefficients(mesh: TriangleMesh, coefficients: Any, degree: int) -> FloatArray:
    """Validate the complete canonical coefficient vector once for one field."""
    size = (degree + 1) * len(mesh.faces) + degree * (degree + 1) * len(mesh.cells)
    values = np.asarray(coefficients)
    if values.shape != (size,) or np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("RT coefficients must be a finite real vector of the expected size")
    return values


@dataclass(frozen=True, init=False)
class RTField:
    """Immutable canonical Raviart--Thomas coefficients for repeated point evaluation.

    Construction validates and copies the complete real coefficient vector.
    Subsequent calls validate only their points and incident cells, and use the
    same Piola arithmetic as :func:`rt_evaluate_points`. This avoids scanning an
    entire global field for every small evaluation batch. The original mesh is
    retained; its geometry arrays follow :class:`TriangleMesh`'s read-only contract.
    """

    mesh: TriangleMesh
    coefficients: FloatArray
    degree: int

    def __init__(self, mesh: TriangleMesh, coefficients: Any, degree: int) -> None:
        """Copy coefficients into bytes-backed read-only storage, preserving precision."""
        order = rt_degree(degree)
        values = _validated_coefficients(mesh, coefficients, order)
        immutable = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "coefficients", immutable)
        object.__setattr__(self, "degree", order)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Reestablish immutable storage after serialization, including spawn workers."""
        return type(self), (self.mesh, self.coefficients, self.degree)

    def evaluate(self, points: FloatArray, cell_indices: IntArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one incident cell per point, preserving one-sided physical values."""
        return _evaluate_points(self.mesh, self.coefficients, self.degree, points, cell_indices)


def rt_evaluate_points(
    mesh: TriangleMesh,
    coefficients: FloatArray,
    degree: int,
    points: FloatArray,
    cell_indices: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate one specified incident RT cell at each physical point.

    Points have shape (n,2), with an integer cell index for each point. No
    geometric nearest-neighbor guess or interface averaging is performed.
    Points must belong to their declared triangle, up to barycentric roundoff.
    Values and divergence use the same canonical moments and Piola orientation
    as :func:`rt_evaluate`; storage scales with the number of requested points.
    """
    m = rt_degree(degree)
    values = _validated_coefficients(mesh, coefficients, m)
    return _evaluate_points(mesh, values, m, points, cell_indices)


def _evaluate_points(
    mesh: TriangleMesh,
    coefficients: FloatArray,
    m: int,
    points: FloatArray,
    cell_indices: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply physical Piola evaluation to a previously validated coefficient vector."""
    points = np.asarray(points)
    owners = np.asarray(cell_indices)
    if (
        np.iscomplexobj(points)
        or points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or owners.shape != (len(points),)
        or not np.issubdtype(owners.dtype, np.integer)
        or np.any(owners < 0)
        or np.any(owners >= len(mesh.cells))
    ):
        raise ValueError("provide finite physical points and valid integer incident-cell indices")
    vertices = mesh.points[mesh.cells[owners]]
    jacobian = (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
    coordinates = np.linalg.solve(jacobian, (points - vertices[:, 0])[..., None])[..., 0]
    bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
    if np.any(bary < -1e-12):
        raise ValueError("evaluation point lies outside its declared incident triangle")
    polynomial, derivative = _polynomials(m, coordinates)
    dual = _dual_coefficients(m)
    reference = np.einsum("qja,ji->qia", polynomial, dual)
    orientation = np.ones((len(points), (m + 1) * (m + 3)))
    orientation[:, : 3 * (m + 1)] = (mesh.signs[owners, :, None] ** np.arange(1, m + 2)).reshape(
        len(points), -1
    )
    local = coefficients[_cell_dofs(mesh, m, owners)] * orientation
    reference_values = np.einsum("qia,qi->qa", reference, local)
    determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    values = np.einsum("qab,qb->qa", jacobian, reference_values) / determinant[:, None]
    divergence = np.einsum("qi,qi->q", derivative @ dual, local) / determinant
    return values, divergence


def rt_interpolate(mesh: TriangleMesh, field: Any, degree: int, order: int = 6) -> FloatArray:
    """Interpolate a single-valued physical vector field using canonical RT moments.

    This routine requires a well-defined normal trace. The MHM moment
    reconstruction supplies its own averaged one-sided interior-face data.
    """
    m = rt_degree(degree)
    order = positive_int(order, "quadrature order", m + 2)
    coefficients = np.empty((m + 1) * len(mesh.faces) + m * (m + 1) * len(mesh.cells))
    x, w = leggauss(order)
    parameter = (x + 1) / 2
    vertices = mesh.points[mesh.faces]
    points = vertices[:, :1] + parameter[None, :, None] * (vertices[:, 1:] - vertices[:, :1])
    values = vector_values(field, points.reshape(-1, 2)).reshape(points.shape)
    normal = np.einsum("fqa,fa->fq", values, mesh.normals)
    coefficients[: (m + 1) * len(mesh.faces)] = np.einsum(
        "q,qi,fq,f->fi", w / 2, legvander(x, m), normal, mesh.lengths
    ).ravel()
    if m:
        bary, weights = triangle_quadrature(order)
        vertices = mesh.points[mesh.cells]
        points = np.einsum("qi,tij->tqj", bary, vertices)
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        values = vector_values(field, points.reshape(-1, 2)).reshape(points.shape)
        pullback = np.einsum("tab,tqb->tqa", inverse, values)
        coefficients[(m + 1) * len(mesh.faces) :] = np.einsum(
            "q,qi,tqa,t->tia", weights, rt_interior_tests(m, bary[:, 1:]), pullback, mesh.areas
        ).ravel()
    return coefficients
