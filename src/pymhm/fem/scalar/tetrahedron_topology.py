"""Topological tetrahedral Pk nodes and inherited Basix derivatives."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int


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


def tetra_polynomials(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return native Pk values and canonical barycentric gradients/Hessians.

    Basix tabulates equispaced Pk in :func:`tetra_indices` order. Derivative
    axes have length four, with lambda_0 entries zero and Cartesian reference
    derivatives in lambda_1 through lambda_3. This declares the extension
    ``p(lambda_1,lambda_2,lambda_3)``; contraction with physical barycentric
    gradients gives the affine physical derivatives. Finite real coordinates
    must sum to one, including extrapolation points. Positive degree support
    is inherited from Basix; equispaced interpolation retains its conditioning
    limitations at high degree.
    """
    from pymhm.fem.reference import barycentric_simplex_tabulation

    return barycentric_simplex_tabulation(
        "tetrahedron", degree, bary, nodes=tetra_indices(degree) / degree
    )


def tetra_values_gradients(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return native Pk values/canonical gradients without allocating Hessians.

    The node order, unit-sum point contract and padded four-coordinate
    derivative convention follow :func:`tetra_polynomials`.
    """
    from pymhm.fem.reference import barycentric_simplex_tabulation

    values, gradient, _ = barycentric_simplex_tabulation(
        "tetrahedron", degree, bary, nodes=tetra_indices(degree) / degree, nderiv=1
    )
    return values, gradient
