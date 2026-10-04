"""Exact tensor-product comparison of nested Cartesian finite element fields."""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm._legacy.models.darcy.conforming import ConformingQuadrilateralSolution
from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature


@lru_cache(maxsize=32)
def cross_moments(coarse_degree: int, fine_degree: int, ratio: int) -> tuple:
    """Integrate coarse Lagrange functions against a continuous piecewise fine basis."""
    gauss, weights = leggauss(max(coarse_degree, fine_degree) + 1)
    gauss, weights = (gauss + 1) / 2, weights / 2

    def basis(degree: int, points: np.ndarray) -> tuple:
        """Evaluate cardinal products and derivatives without expanded monomials."""
        nodes = np.linspace(0, 1, degree + 1, dtype=np.longdouble)
        points = np.asarray(points, dtype=np.longdouble)
        values = np.empty((len(points), degree + 1), dtype=np.longdouble)
        derivatives = np.zeros_like(values)
        for index in range(degree + 1):
            others = np.delete(np.arange(degree + 1), index)
            factors = (points[:, None] - nodes[others]) / (nodes[index] - nodes[others])
            values[:, index] = factors.prod(axis=1)
            for omitted, node in enumerate(others):
                derivatives[:, index] += np.delete(factors, omitted, axis=1).prod(axis=1) / (
                    nodes[index] - nodes[node]
                )
        return values, derivatives

    gauss, weights = (
        np.asarray(gauss, dtype=np.longdouble),
        np.asarray(weights, dtype=np.longdouble),
    )
    fine, fine_derivative = basis(fine_degree, gauss)
    mass = np.zeros((coarse_degree + 1, ratio * fine_degree + 1), dtype=np.longdouble)
    stiffness = np.zeros_like(mass)
    for child in range(ratio):
        coarse, coarse_derivative = basis(coarse_degree, (child + gauss) / ratio)
        indices = slice(child * fine_degree, (child + 1) * fine_degree + 1)
        mass[:, indices] += coarse.T @ (weights[:, None] * fine) / ratio
        stiffness[:, indices] += coarse_derivative.T @ (weights[:, None] * fine_derivative)
    return mass, stiffness


def _tensor_product(
    first: np.ndarray, second: np.ndarray, y: np.ndarray, x: np.ndarray
) -> np.longdouble:
    """Sum an exact separable bilinear form without constructing its Kronecker matrix."""
    transformed = np.matmul(y, np.matmul(second, x.T))
    return np.sum(first * transformed, dtype=np.longdouble)


def _patches(
    field: ConformingQuadrilateralSolution,
    origin: np.ndarray,
    spacing: np.ndarray,
    counts: np.ndarray,
    ratio: np.ndarray,
    begin: int,
    end: int,
) -> np.ndarray:
    """Read one continuous nodal patch for every selected coarse cell, in physical order."""
    ids = np.arange(begin, end)
    physical = origin + np.column_stack((ids % counts[0], ids // counts[0])) * spacing
    start = (
        np.rint((physical - field.mesh.points[0]) / field.mesh.spacing).astype(int) * field.degree
    )
    width = field.mesh.nx * field.degree + 1
    x = np.arange(ratio[0] * field.degree + 1)
    y = np.arange(ratio[1] * field.degree + 1)
    indices = (start[:, 1, None, None] + y[None, :, None]) * width
    indices = indices + start[:, 0, None, None] + x[None, None, :]
    return field.pressure[indices]


def field_norms(field: ConformingQuadrilateralSolution, bounds: tuple | None = None) -> np.ndarray:
    """Integrate pressure squared and gradient squared on a cell-aligned subrectangle."""
    lower = np.array(field.mesh.bounds)[[0, 2]] if bounds is None else np.array(bounds)[[0, 2]]
    upper = np.array(field.mesh.bounds)[[1, 3]] if bounds is None else np.array(bounds)[[1, 3]]
    spacing = field.mesh.spacing
    offsets = (np.r_[lower, upper].reshape(2, 2) - field.mesh.points[0]) / spacing
    if (
        not np.allclose(offsets, np.rint(offsets), rtol=0, atol=1e-10)
        or np.any(offsets[0] < 0)
        or np.any(offsets[1] > np.array([field.mesh.nx, field.mesh.ny]))
        or np.any(upper <= lower)
    ):
        raise ValueError("integration subrectangle must be cell-aligned and inside the field")
    counts = np.rint((upper - lower) / spacing).astype(int)
    mass, stiffness = cross_moments(field.degree, field.degree, 1)
    total = np.zeros(2, dtype=np.longdouble)
    for begin in range(0, int(np.prod(counts)), 4096):
        values = _patches(
            field,
            lower,
            spacing,
            counts,
            np.ones(2, dtype=int),
            begin,
            min(begin + 4096, int(np.prod(counts))),
        )
        total[0] += np.prod(spacing) * _tensor_product(values, values, mass, mass)
        horizontal = values - values[:, :, :1]
        vertical = values - values[:, :1, :]
        total[1] += (
            spacing[1] / spacing[0] * _tensor_product(horizontal, horizontal, mass, stiffness)
        )
        total[1] += spacing[0] / spacing[1] * _tensor_product(vertical, vertical, stiffness, mass)
    return total


def cross_norms(
    reference: ConformingQuadrilateralSolution, field: ConformingQuadrilateralSolution
) -> np.ndarray:
    """Integrate pressure/gradient products of nested partitions on the field domain."""
    bounds = np.array(field.mesh.bounds)
    if np.all(reference.mesh.spacing >= field.mesh.spacing):
        coarse, fine = reference, field
    elif np.all(field.mesh.spacing >= reference.mesh.spacing):
        coarse, fine = field, reference
    else:
        raise ValueError("comparison requires nested Cartesian partitions in both directions")
    spacing = coarse.mesh.spacing
    ratio = spacing / fine.mesh.spacing
    origin = bounds[[0, 2]]
    counts = (bounds[[1, 3]] - origin) / spacing
    if not np.allclose(ratio, np.rint(ratio), rtol=0, atol=1e-10) or not np.allclose(
        counts, np.rint(counts), rtol=0, atol=1e-10
    ):
        raise ValueError("comparison domain and fine cells must align with the coarser grid")
    ratio, counts = np.rint(ratio).astype(int), np.rint(counts).astype(int)
    mx, sx = cross_moments(coarse.degree, fine.degree, int(ratio[0]))
    my, sy = cross_moments(coarse.degree, fine.degree, int(ratio[1]))
    total = np.zeros(2, dtype=np.longdouble)
    for begin in range(0, int(np.prod(counts)), 2048):
        end = min(begin + 2048, int(np.prod(counts)))
        first = _patches(coarse, origin, spacing, counts, np.ones(2, dtype=int), begin, end)
        second = _patches(fine, origin, spacing, counts, ratio, begin, end)
        total[0] += np.prod(spacing) * _tensor_product(first, second, my, mx)
        total[1] += (
            spacing[1]
            / spacing[0]
            * _tensor_product(first - first[:, :, :1], second - second[:, :, :1], my, sx)
        )
        total[1] += (
            spacing[0]
            / spacing[1]
            * _tensor_product(first - first[:, :1, :], second - second[:, :1, :], sy, mx)
        )
    return total


def quadrature_difference(
    reference: ConformingQuadrilateralSolution, fields: tuple[ConformingQuadrilateralSolution, ...]
) -> dict:
    """Evaluate positive exact quadrature directly, retaining near-zero differences."""
    order = max(reference.degree, *(field.degree for field in fields)) + 1
    points, weights = quadrilateral_quadrature(order)
    total, norm = np.zeros(2, dtype=np.longdouble), np.zeros(2, dtype=np.longdouble)
    spacing = np.minimum(
        reference.mesh.spacing, np.min([field.mesh.spacing for field in fields], axis=0)
    )
    for field in fields:
        origin = field.mesh.points[0]
        counts = np.rint((np.array(field.mesh.bounds)[[1, 3]] - origin) / spacing).astype(int)
        ratio = field.mesh.spacing / spacing
        if not np.allclose(ratio, np.rint(ratio), rtol=0, atol=1e-10):
            raise ValueError("integration grid must resolve both finite-element partitions")
        for begin in range(0, int(np.prod(counts)), 1024):
            ids = np.arange(begin, min(begin + 1024, int(np.prod(counts))))
            cell_origins = origin + np.column_stack((ids % counts[0], ids // counts[0])) * spacing
            physical = cell_origins[:, None] + points[None] * spacing
            flat = physical.reshape(-1, 2)
            p, g = reference.evaluate(flat)
            v, d = field.evaluate(flat)
            errors = np.column_stack(((p - v) ** 2, np.sum((g - d) ** 2, axis=1)))
            values = np.column_stack((p * p, np.sum(g * g, axis=1)))
            total += np.prod(spacing) * np.einsum(
                "q,tqi->i", weights, errors.reshape(-1, len(weights), 2)
            )
            norm += np.prod(spacing) * np.einsum(
                "q,tqi->i", weights, values.reshape(-1, len(weights), 2)
            )
    return _report(total, norm)


def _report(total: np.ndarray, norm: np.ndarray) -> dict:
    """State both the gradient seminorm and the full H1 relative norm explicitly."""
    if not np.isfinite([total, norm]).all() or norm[0] <= 0:
        raise ValueError("relative comparison requires finite fields and a nonzero reference")
    return dict(
        l2=float(np.sqrt(total[0])),
        h1_seminorm=float(np.sqrt(total[1])),
        relative_l2=float(np.sqrt(total[0] / norm[0])),
        relative_h1=float(np.sqrt(total.sum() / norm.sum())),
        reference_l2=float(np.sqrt(norm[0])),
        reference_h1=float(np.sqrt(norm.sum())),
    )


def difference(
    reference: ConformingQuadrilateralSolution, fields: tuple[ConformingQuadrilateralSolution, ...]
) -> dict:
    """Integrate polynomial differences exactly, using direct quadrature near cancellation.

    Separable mass/stiffness products avoid re-evaluating all polynomials at every
    fine-grid quadrature point. If subtraction of integrated products approaches
    roundoff, positive pointwise quadrature evaluates the differences directly.
    The denominator is always the reference restricted to the compared domain.
    """
    total, norm = np.zeros(2, dtype=np.longdouble), np.zeros(2, dtype=np.longdouble)
    for field in fields:
        norm_reference = field_norms(reference, field.mesh.bounds)
        if reference.degree == field.degree and np.array_equal(
            reference.mesh.spacing, field.mesh.spacing
        ):
            start = np.rint(
                (field.mesh.points[0] - reference.mesh.points[0]) / reference.mesh.spacing
            ).astype(int)
            width = reference.mesh.nx * reference.degree + 1
            x = start[0] * field.degree + np.arange(field.mesh.nx * field.degree + 1)
            y = start[1] * field.degree + np.arange(field.mesh.ny * field.degree + 1)
            indices = y[:, None] * width + x
            delta = ConformingQuadrilateralSolution(
                field.mesh,
                field.degree,
                reference.pressure[indices.ravel()] - field.pressure,
                1.0,
                0.0,
            )
            total += field_norms(delta)
        else:
            norm_field = field_norms(field)
            total += norm_reference + norm_field - 2 * cross_norms(reference, field)
        norm += norm_reference
    common_basis = all(
        reference.degree == field.degree
        and np.array_equal(reference.mesh.spacing, field.mesh.spacing)
        for field in fields
    )
    if not common_basis and np.any(total < 1e-10 * norm):
        return quadrature_difference(reference, fields)
    return _report(total, norm)
