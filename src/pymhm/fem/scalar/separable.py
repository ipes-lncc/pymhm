"""Cartesian scalar operators from explicit one-dimensional weighted factors."""

from typing import cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, positive_int, real_array
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.materials.separable import SeparableField, factor_values
from pymhm.meshes.cartesian import CartesianMacroMesh

_UNIT = SeparableField(((1.0, 1.0),))
_ZERO = SeparableField(())


def interval_nodal_quadrature(
    count: int, interval: tuple[float, float], degree: int, order: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, np.ndarray, float]:
    """Return points, unit-interval weights, values, derivatives, DOFs and spacing.

    Physical points have shape (cells, quadrature); value/derivative tables have
    shape (quadrature, degree+1). Derivatives are in reference coordinates and
    DOFs share equispaced interval nodes. ``interval_weighted_operators`` applies
    the physical spacing to mass, stiffness and load. No boundary is prescribed.
    """
    count = positive_int(count, "cell count")
    degree = positive_int(degree, "degree")
    order = positive_int(order, "quadrature order")
    bounds = np.asarray(interval)
    if (
        bounds.shape != (2,)
        or np.iscomplexobj(bounds)
        or not np.isfinite(bounds).all()
        or bounds[1] <= bounds[0]
    ):
        raise ValueError("interval must contain two increasing finite real coordinates")
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    basis, gradient = qk_basis(degree, np.column_stack((nodes, np.zeros_like(nodes))))
    basis, derivative = basis[:, : degree + 1], gradient[:, : degree + 1, 0]
    spacing = (interval[1] - interval[0]) / count
    physical = interval[0] + spacing * (np.arange(count)[:, None] + nodes)
    dofs = degree * np.arange(count)[:, None] + np.arange(degree + 1)
    return physical, weights, basis, derivative, dofs, spacing


def interval_weighted_operators(data: tuple, values: FloatArray) -> tuple:
    """Assemble physical weighted mass, stiffness and load in shared nodal order.

    ``data`` is the explicit interval_nodal_quadrature tuple, and values has its
    (cells, quadrature) shape. Signed weights are admissible; PDE coefficient
    positivity is a separate condition. Return two CSR matrices and one vector.
    """
    physical, weights, basis, derivative, dofs, spacing = data
    values = real_array(values, "interval coefficient values")
    if values.shape != physical.shape:
        raise ValueError("interval coefficient values must match the physical quadrature points")
    weighted = values * weights
    mass = spacing * np.einsum("tq,qi,qj->tij", weighted, basis, basis)
    stiffness = np.einsum("tq,qi,qj->tij", weighted, derivative, derivative) / spacing
    load = spacing * np.einsum("tq,qi->ti", weighted, basis)
    rows = np.broadcast_to(dofs[:, :, None], mass.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], mass.shape).ravel()
    size = int(dofs.max()) + 1
    matrices = tuple(
        sparse.coo_matrix((item.ravel(), (rows, columns)), shape=(size, size)).tocsr()
        for item in (mass, stiffness)
    )
    forcing = np.bincount(dofs.ravel(), weights=load.ravel(), minlength=size)
    return *matrices, forcing


def require_positive_separated(values: list[tuple[FloatArray, ...]]) -> None:
    """Require sum(a(x)*b(y)) positive at its declared tensor sampling points.

    Each pair contains finite sampled x/y factor arrays. Signed terms are
    admissible. The check uses a bound when sufficient and otherwise bounded
    temporary blocks. It does not certify the coefficient between these points.
    """
    if not values:
        raise ValueError("positivity requires a nonempty separated sum")
    lower = sum(
        min(
            float(a.min() * b.min()),
            float(a.min() * b.max()),
            float(a.max() * b.min()),
            float(a.max() * b.max()),
        )
        for a, b in values
    )
    if not np.isfinite(lower):
        raise ValueError("separable permeability products must remain finite")
    if lower > 0:
        return
    for begin in range(0, values[0][0].size, 128):
        coefficient = sum(a.ravel()[begin : begin + 128, None] * b.ravel() for a, b in values)
        if not np.isfinite(coefficient).all() or np.any(coefficient <= 0):
            raise ValueError("separable permeability must be positive at every tensor Gauss point")


def separable_diffusion_operators(
    mesh: CartesianMacroMesh,
    degree: int,
    *,
    permeability: SeparableField = _UNIT,
    source: SeparableField = _ZERO,
    order: int = 6,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, FloatArray]:
    """Assemble scalar diffusion, mass and load by exact Kronecker identities.

    The quadrature rule uses at least ``degree+1`` Gauss points per coordinate.
    Coefficients and source are explicit separated sums; their spatial variation
    is integrated with this same rule, without homogenization or low-rank fitting.
    Unknowns retain the native ordering ``ix + (nx*degree+1)*iy``.
    """
    if not isinstance(mesh, CartesianMacroMesh):
        raise TypeError("separable diffusion requires CartesianMacroMesh")
    if not isinstance(permeability, SeparableField) or not permeability.terms:
        raise ValueError("permeability requires a nonempty SeparableField")
    if not isinstance(source, SeparableField):
        raise TypeError("source must be a SeparableField")
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "quadrature order"), degree + 1)
    x0, x1, y0, y1 = mesh.bounds
    axes = (
        interval_nodal_quadrature(mesh.nx, (x0, x1), degree, order),
        interval_nodal_quadrature(cast(int, mesh.ny), (y0, y1), degree, order),
    )
    values = [
        tuple(
            factor_values(factor, axis[0].ravel()).reshape(axis[0].shape)
            for factor, axis in zip(term, axes, strict=True)
        )
        for term in permeability.terms
    ]
    require_positive_separated(values)
    base = tuple(interval_weighted_operators(axis, np.ones_like(axis[0])) for axis in axes)
    mass = sparse.kron(base[1][0], base[0][0], format="csr")
    matrix = sparse.csr_matrix(mass.shape)
    for pair in values:
        x, y = (
            interval_weighted_operators(axis, value) for axis, value in zip(axes, pair, strict=True)
        )
        matrix = (
            matrix + sparse.kron(y[0], x[1], format="csr") + sparse.kron(y[1], x[0], format="csr")
        )
    load = np.zeros(mass.shape[0])
    for term in source.terms:
        vectors = [
            interval_weighted_operators(
                axis, factor_values(factor, axis[0].ravel()).reshape(axis[0].shape)
            )[2]
            for factor, axis in zip(term, axes, strict=True)
        ]
        load += np.kron(vectors[1], vectors[0])
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or not np.isfinite(load).all():
        raise ValueError("assembled separable diffusion and load must remain finite")
    return matrix, mass, load
