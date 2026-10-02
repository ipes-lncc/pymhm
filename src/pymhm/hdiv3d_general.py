"""Moment-defined BDM and tensor-prismatic spaces of arbitrary polynomial order.

Complete BDM spaces are restricted through their normal *polynomials*, rather
than by discarding high face moments. All zero-normal bubbles are retained.
    The construction uses explicit Nedelec interior tests and positive triangular
factors; no singular-vector orientation defines persisted coordinates.
"""

from functools import cache
from itertools import product
from math import factorial, prod

import numpy as np
from scipy.linalg import block_diag

from pymhm.hdiv3d_family import (
    CellKind,
    _powers,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    reference_faces,
    reference_vertices,
)
from pymhm.mesh import FloatArray


def _bernstein(points: FloatArray, degree: int) -> tuple[FloatArray, FloatArray]:
    """Tabulate a simplex Bernstein basis and analytic Cartesian derivatives."""
    dimension = points.shape[1]
    bary = np.column_stack((1 - points.sum(axis=1), points))
    powers = tuple(e for e in product(range(degree + 1), repeat=dimension + 1) if sum(e) == degree)
    values = np.column_stack(
        [
            factorial(degree) / prod(factorial(a) for a in e) * np.prod(bary**e, axis=1)
            for e in powers
        ]
    )
    gradient = np.zeros((*values.shape, dimension))
    for i, exponent in enumerate(powers):
        factor = factorial(degree) / prod(factorial(a) for a in exponent)
        for axis in range(dimension):
            for coordinate, sign in ((0, -1), (axis + 1, 1)):
                if exponent[coordinate]:
                    lower = np.array(exponent)
                    lower[coordinate] -= 1
                    gradient[:, i, axis] += (
                        sign * exponent[coordinate] * factor * np.prod(bary**lower, axis=1)
                    )
    return values, gradient


def candidates(kind: CellKind, degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Tabulate stable Bernstein vector candidates and their divergence.

    Rows in a persisted matrix are grouped by component. Within a component,
    simplex barycentric exponents are lexicographic; prism factors enumerate
    the triangle first and then the interval. Horizontal prism degrees are
    (degree+1, degree), and vertical degrees are (degree, degree+1).
    """
    fields, derivatives = [], []
    for axis in range(3):
        if kind == "tetrahedron":
            scalar, gradient = _bernstein(points, degree + 1)
            derivative = gradient[..., axis]
        else:
            triangle_degree = degree + (axis < 2)
            interval_degree = degree + (axis == 2)
            triangle, dtriangle = _bernstein(points[:, :2], triangle_degree)
            interval, dinterval = _bernstein(points[:, 2:], interval_degree)
            scalar = np.einsum("qi,qj->qij", triangle, interval).reshape(len(points), -1)
            derivative = np.einsum(
                "qi,qj->qij",
                dtriangle[..., axis] if axis < 2 else triangle,
                interval if axis < 2 else dinterval[..., 0],
            ).reshape(len(points), -1)
        fields.append(scalar[..., None] * np.eye(3)[axis])
        derivatives.append(derivative)
    return np.concatenate(fields, axis=1), np.concatenate(derivatives, axis=1)


def interior_tests(kind: CellKind, degree: int, points: FloatArray) -> FloatArray:
    """Evaluate ordered interior moment tests for pressure degree ``degree``.

    Tetrahedral BDM_(degree+1) uses the first-kind Nedelec space of order
    degree-1. Prisms use triangular Nedelec tests tensor P_degree(z) in
    horizontal components and P_degree(x,y) tensor P_(degree-1)(z) vertically.
    Negative polynomial orders denote the empty space.
    """
    x = np.asarray(points)
    tests = []
    if kind == "tetrahedron":
        for exponent in _powers(kind, degree - 1):
            for axis in range(3):
                tests.append(np.prod(x**exponent, axis=1)[:, None] * np.eye(3)[axis])
        homogeneous = [e for e in _powers(kind, degree - 1) if sum(e) == degree - 1]
        for axis in range(3):
            for exponent in homogeneous:
                # x cross (x * scalar) is zero. Removing z-divisible tests
                # in the final component removes exactly that dependency.
                if axis == 2 and exponent[2]:
                    continue
                tests.append(np.cross(x, np.eye(3)[axis]) * np.prod(x**exponent, axis=1)[:, None])
    else:
        for z in range(degree + 1):
            for a, b in product(range(degree), repeat=2):
                if a + b <= degree - 1:
                    scalar = x[:, 0] ** a * x[:, 1] ** b * x[:, 2] ** z
                    tests.extend(scalar[:, None] * np.eye(3)[axis] for axis in range(2))
            for a in range(degree):
                scalar = x[:, 0] ** a * x[:, 1] ** (degree - 1 - a) * x[:, 2] ** z
                tests.append(
                    scalar[:, None] * np.column_stack((-x[:, 1], x[:, 0], np.zeros(len(x))))
                )
        for a, b, z in product(range(degree + 1), range(degree + 1), range(degree)):
            if a + b <= degree:
                scalar = x[:, 0] ** a * x[:, 1] ** b * x[:, 2] ** z
                tests.append(scalar[:, None] * np.eye(3)[2])
    return np.stack(tests, axis=1) if tests else np.empty((len(x), 0, 3))


@cache
def coefficients(kind: CellKind, pressure_degree: int, normal_degree: int) -> FloatArray:
    """Construct uniquely oriented face lifts and all normalized interior bubbles."""
    degree = pressure_degree
    xyz, weights = cell_quadrature(kind, degree + 4)
    values = candidates(kind, degree, xyz)[0]
    vertices = reference_vertices(kind)
    face_moments, embeddings = [], []
    for indices in reference_faces(kind):
        nodes = vertices[list(indices)]
        uv, w = face_quadrature(len(indices), degree + 4)
        normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        normal *= np.sign(normal @ (nodes.mean(axis=0) - vertices.mean(axis=0)))
        normal_values = candidates(kind, degree, face_shape(uv, len(indices)) @ nodes)[0] @ normal
        if kind == "tetrahedron":
            full = face_polynomials(uv, 3, degree + 1)
        elif len(indices) == 3:
            full = face_polynomials(uv, 3, degree)
        else:
            full = np.column_stack(
                [
                    uv[:, 0] ** a * uv[:, 1] ** b
                    for a in range(degree + 2)
                    for b in range(degree + 1)
                ]
            )
        low = face_polynomials(uv, len(indices), normal_degree)
        face_moments.append(full.T @ (w[:, None] * normal_values))
        embeddings.append((full.T @ (w[:, None] * low)) @ np.linalg.inv(low.T @ (w[:, None] * low)))
    interior = interior_tests(kind, degree, xyz)
    moments = np.vstack((*face_moments, np.einsum("q,qia,qja->ij", weights, interior, values)))
    if moments.shape[0] != moments.shape[1]:
        raise ArithmeticError("BDM moment count does not match the polynomial dimension")
    dual = np.linalg.solve(moments, np.eye(len(moments)))
    # Resolve cancellation in the higher-order moment dual before normalizing
    # bubbles. Residuals use the original moment equations, accumulated wider
    # than binary64 where available; the basis remains stored in binary64.
    extended_moments = moments.astype(np.longdouble)
    identity = np.eye(len(moments), dtype=np.longdouble)
    for _ in range(3):
        residual = identity - extended_moments @ dual.astype(np.longdouble)
        dual += np.linalg.solve(moments, np.asarray(residual, dtype=float))
    boundary = sum(len(block) for block in face_moments)
    bubbles = dual[:, boundary:]
    if bubbles.shape[1]:
        sampled = np.einsum("qia,ij->qja", values, bubbles)
        weighted = (sampled * np.sqrt(weights)[:, None, None]).transpose(0, 2, 1)
        _, triangular = np.linalg.qr(weighted.reshape(-1, bubbles.shape[1]))
        triangular *= np.sign(np.diag(triangular))[:, None]
        bubbles = np.linalg.solve(triangular.T, bubbles.T).T
    lift = dual[:, :boundary] @ block_diag(*embeddings)
    bvalues = np.einsum("qia,ij->qja", values, bubbles)
    lvalues = np.einsum("qia,ij->qja", values, lift)
    lift -= bubbles @ np.einsum("q,qia,qja->ij", weights, bvalues, lvalues)
    result = np.column_stack((lift, bubbles))
    result.setflags(write=False)
    return result
