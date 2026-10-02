"""Topological equispaced tetrahedral Pk nodes and polynomial derivatives."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial

from pymhm.mesh import FloatArray, IntArray, positive_int


def _compositions(total: int, count: int) -> Iterator[tuple[int, ...]]:
    """Yield positive compositions in lexicographic order without a Cartesian-product search."""
    if count == 1:
        if total > 0:
            yield (total,)
        return
    for first in range(1, total - count + 2):
        for tail in _compositions(total - first, count - 1):
            yield (first, *tail)


@lru_cache(maxsize=16)
def tetra_indices(degree: int) -> IntArray:
    """Return equispaced barycentric nodal multi-indices for any positive degree.

    Vertices precede edge interiors, face interiors and cell interiors. Node
    identity depends on vertex indices and integer barycentric weights, never
    on rounded physical coordinates.
    """
    degree = positive_int(degree, "degree")
    indices: list[list[int]] = []
    for count in range(1, 5):
        for axes in combinations(range(4), count):
            for weights in _compositions(degree, count):
                entry = [0] * 4
                for axis, weight in zip(axes, weights, strict=True):
                    entry[axis] = weight
                indices.append(entry)
    result = np.asarray(indices, dtype=np.int64)
    result.flags.writeable = False
    return result


def continuous_tetra_nodes(mesh: Any, degree: int) -> tuple[IntArray, FloatArray]:
    """Enumerate continuous Pk DOFs through exact shared vertex/edge/face identities."""
    indices = tetra_indices(degree)
    lookup: dict[tuple[tuple[int, int], ...], int] = {}
    coordinates: list[FloatArray] = []
    dofs = np.empty((len(mesh.cells), len(indices)), dtype=np.int64)
    for cell, vertices in enumerate(mesh.cells):
        for node, weights in enumerate(indices):
            key = tuple(
                sorted((int(v), int(w)) for v, w in zip(vertices, weights, strict=True) if w)
            )
            if key not in lookup:
                lookup[key] = len(coordinates)
                coordinates.append(weights @ mesh.points[vertices] / degree)
            dofs[cell, node] = lookup[key]
    return dofs, np.asarray(coordinates)


@lru_cache(maxsize=4)
def _factors(degree: int) -> tuple[tuple[Polynomial, ...], ...]:
    """Cache one-dimensional cardinal factors and their first two derivatives."""
    factors = []
    for weight in range(degree + 1):
        polynomial = Polynomial([1.0])
        for j in range(weight):
            polynomial *= Polynomial([-j, degree]) / (weight - j)
        factors.append((polynomial, polynomial.deriv(), polynomial.deriv(2)))
    return tuple(factors)


def _points(bary: FloatArray) -> FloatArray:
    """Validate real barycentric coordinates without imposing a particular derivative direction."""
    raw = np.asarray(bary)
    if np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("barycentric points must be finite and real")
    points = np.asarray(raw, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("barycentric points must have shape (n, 4)")
    return points


def _legacy_polynomials(
    degree: int, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Retain the executed P1--P4 arithmetic for coefficient-archive compatibility."""
    indices = tetra_indices(degree)
    factors = _factors(degree)
    values = np.ones((len(points), len(indices)))
    gradient = np.empty((*values.shape, 4))
    hessian = np.empty((*values.shape, 4, 4))
    for node, weights in enumerate(indices):
        univariate = np.asarray(
            [
                [factors[w][order](points[:, axis]) for order in range(3)]
                for axis, w in enumerate(weights)
            ]
        )
        values[:, node] = univariate[:, 0].prod(axis=0)
        for a in range(4):
            derivative = [univariate[i, int(i == a)] for i in range(4)]
            gradient[:, node, a] = np.prod(derivative, axis=0)
            for b in range(4):
                second = [univariate[i, int(i == a) + int(i == b)] for i in range(4)]
                hessian[:, node, a, b] = np.prod(second, axis=0)
    return values, gradient, hessian


def _recursive_tabulation(
    degree: int, points: FloatArray, *, second: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tabulate factored cardinal polynomials without monomial expansion or division by values.

    F_j(x) = F_(j-1)(x) (k*x-j+1)/j. Differentiating this recurrence
    evaluates derivatives even at roots. Every axis/weight is tabulated once;
    storage scales with requested polynomial values and derivatives. Equispaced
    high-degree interpolation retains its inherent conditioning limitations.
    """
    indices = tetra_indices(degree)
    derivatives = 3 if second else 2
    factors = np.zeros((4, degree + 1, derivatives, len(points)))
    factors[:, 0, 0] = 1
    for weight in range(1, degree + 1):
        multiplier = (degree * points.T - weight + 1) / weight
        previous = factors[:, weight - 1]
        factors[:, weight, 0] = previous[:, 0] * multiplier
        factors[:, weight, 1] = previous[:, 1] * multiplier + (degree / weight) * previous[:, 0]
        if second:
            factors[:, weight, 2] = (
                previous[:, 2] * multiplier + (2 * degree / weight) * previous[:, 1]
            )
    selected = [factors[axis, indices[:, axis]].transpose(1, 2, 0) for axis in range(4)]
    values = np.prod([axis[0] for axis in selected], axis=0)
    gradient = np.empty((*values.shape, 4))
    hessian = np.empty((*values.shape, 4, 4)) if second else np.empty((0, 0, 4, 4))
    for a in range(4):
        gradient[..., a] = np.prod([selected[i][int(i == a)] for i in range(4)], axis=0)
        if second:
            for b in range(a, 4):
                entry = np.prod([selected[i][int(i == a) + int(i == b)] for i in range(4)], axis=0)
                hessian[..., a, b] = entry
                hessian[..., b, a] = entry
    return values, gradient, hessian


def tetra_polynomials(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate Pk cardinal values, first derivatives and Hessians in barycentric variables.

    Positive degrees are unrestricted. P1--P4 retain their established arithmetic;
    higher degrees use factored recurrences. Derivatives are taken with respect
    to four independent coordinates, before imposing their unit-sum constraint.
    """
    degree = positive_int(degree, "degree")
    points = _points(bary)
    if degree <= 4:
        return _legacy_polynomials(degree, points)
    return _recursive_tabulation(degree, points, second=True)


def tetra_values_gradients(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate Pk values/gradients without allocating discarded high-order Hessians."""
    degree = positive_int(degree, "degree")
    points = _points(bary)
    if degree <= 4:
        values, gradient, _ = _legacy_polynomials(degree, points)
    else:
        values, gradient, _ = _recursive_tabulation(degree, points, second=False)
    return values, gradient
