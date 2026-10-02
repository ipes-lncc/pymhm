"""Conforming Cartesian diffusion with finite sums of separable scalar fields.

Kronecker assembly is algebraically the same tensor Gauss Galerkin operator as
elementwise Qk assembly. It avoids evaluating two-dimensional coefficient arrays
and dense element matrices when a coefficient is explicitly given in separated
form. This module does not approximate a nonseparable field by a low-rank fit.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.conforming import ConformingQuadrilateralSolution
from pymhm.elements import scalar_values
from pymhm.mesh import FloatArray, positive_int
from pymhm.quadrilateral import CartesianMacroMesh, qk_basis
from pymhm.solvers import solve_linear


def _factor(value: Any, points: FloatArray) -> FloatArray:
    """Evaluate a real one-coordinate factor with a finite scalar/vector contract."""
    raw = np.asarray(value(points) if callable(value) else value)
    if np.iscomplexobj(raw) or raw.shape not in ((), points.shape) or not np.isfinite(raw).all():
        raise ValueError(
            "separable factors must return finite real scalars or coordinate-shaped arrays"
        )
    return np.broadcast_to(np.asarray(raw, dtype=float), points.shape)


@dataclass(frozen=True)
class SeparableField:
    """Represent ``sum(a(x)*b(y) for a,b in terms)`` without approximation.

    Each factor is a finite real scalar or a callback accepting a one-dimensional
    coordinate array. An empty sum represents zero. Signed terms are allowed;
    a diffusion assembler separately verifies positivity at its Gauss points.
    """

    terms: tuple[tuple[Any, Any], ...]

    def __post_init__(self) -> None:
        """Normalize term pairs and reject nonscalar, nonfinite constant factors."""
        try:
            terms = tuple(tuple(term) for term in self.terms)
        except TypeError as exc:
            raise ValueError("separable terms must be pairs of one-coordinate factors") from exc
        if any(len(term) != 2 for term in terms):
            raise ValueError("separable terms must be pairs of one-coordinate factors")
        for term in terms:
            for factor in term:
                if not callable(factor):
                    if np.asarray(factor).ndim != 0:
                        raise ValueError("constant separable factors must be scalar")
                    _factor(factor, np.array([0.0]))
        object.__setattr__(self, "terms", terms)

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate the declared scalar field at physical XY point pairs."""
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
            raise ValueError("separable field points must be finite real XY pairs")
        result = np.zeros(len(raw))
        for x, y in self.terms:
            result += _factor(x, raw[:, 0]) * _factor(y, raw[:, 1])
        if not np.isfinite(result).all():
            raise ValueError("separable field evaluation must remain finite")
        return result


_UNIT = SeparableField(((1.0, 1.0),))
_ZERO = SeparableField(())


def _line_data(
    count: int, interval: tuple[float, float], degree: int, order: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, np.ndarray, float]:
    """Prepare one-dimensional shared nodal assembly using the native Qk cardinals."""
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    basis, gradient = qk_basis(degree, np.column_stack((nodes, np.zeros_like(nodes))))
    basis, derivative = basis[:, : degree + 1], gradient[:, : degree + 1, 0]
    spacing = (interval[1] - interval[0]) / count
    physical = interval[0] + spacing * (np.arange(count)[:, None] + nodes)
    dofs = degree * np.arange(count)[:, None] + np.arange(degree + 1)
    return physical, weights, basis, derivative, dofs, spacing


def _line_operators(data: tuple, values: FloatArray) -> tuple:
    """Assemble weighted one-dimensional mass, stiffness and load integrals."""
    _, weights, basis, derivative, dofs, spacing = data
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


def _positive(values: list[tuple[FloatArray, ...]]) -> None:
    """Check coefficient positivity at tensor Gauss points, with bounded temporary memory."""
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
        _line_data(mesh.nx, (x0, x1), degree, order),
        _line_data(cast(int, mesh.ny), (y0, y1), degree, order),
    )
    values = [
        tuple(
            _factor(factor, axis[0].ravel()).reshape(axis[0].shape)
            for factor, axis in zip(term, axes, strict=True)
        )
        for term in permeability.terms
    ]
    _positive(values)
    base = tuple(_line_operators(axis, np.ones_like(axis[0])) for axis in axes)
    mass = sparse.kron(base[1][0], base[0][0], format="csr")
    matrix = sparse.csr_matrix(mass.shape)
    for pair in values:
        x, y = (_line_operators(axis, value) for axis, value in zip(axes, pair, strict=True))
        matrix = (
            matrix + sparse.kron(y[0], x[1], format="csr") + sparse.kron(y[1], x[0], format="csr")
        )
    load = np.zeros(mass.shape[0])
    for term in source.terms:
        vectors = [
            _line_operators(axis, _factor(factor, axis[0].ravel()).reshape(axis[0].shape))[2]
            for factor, axis in zip(term, axes, strict=True)
        ]
        load += np.kron(vectors[1], vectors[0])
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or not np.isfinite(load).all():
        raise ValueError("assembled separable diffusion and load must remain finite")
    return matrix, mass, load


def solve_separable_diffusion(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 2,
    permeability: SeparableField = _UNIT,
    source: SeparableField = _ZERO,
    dirichlet: Any = 0.0,
    quadrature_order: int = 6,
    solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> ConformingQuadrilateralSolution:
    """Solve a classical continuous Qk problem with full strong Dirichlet data.

    This is a conforming global solve, with no MHM skeleton or local condensation.
    Boundary values may be inhomogeneous physical XY callbacks. Natural or mixed
    boundary conditions use :func:`pymhm.conforming.solve_conforming_quadrilateral`.
    """
    matrix, _, load = separable_diffusion_operators(
        mesh, degree, permeability=permeability, source=source, order=quadrature_order
    )
    nx, ny = mesh.nx * degree, cast(int, mesh.ny) * degree
    fixed = np.unique(
        np.r_[
            np.arange(nx + 1),
            ny * (nx + 1) + np.arange(nx + 1),
            (nx + 1) * np.arange(ny + 1),
            nx + (nx + 1) * np.arange(ny + 1),
        ]
    )
    coordinates = (
        np.array(mesh.bounds)[[0, 2]]
        + np.column_stack((fixed % (nx + 1), fixed // (nx + 1))) * mesh.spacing / degree
    )
    pressure = np.zeros(matrix.shape[0])
    pressure[fixed] = scalar_values(dirichlet, coordinates)
    free = np.setdiff1d(np.arange(len(pressure)), fixed, assume_unique=True)
    if len(free):
        solved = solve_linear(
            matrix[free][:, free],
            (load - matrix @ pressure)[free],
            solver=solver,
            refinement_precision=refinement_precision,
        )
        pressure = pressure.astype(solved.dtype)
        pressure[free] = solved
    residual = float(
        np.linalg.norm((matrix @ pressure - load)[free])
        / max(
            np.linalg.norm(load[free]),
            np.linalg.norm((abs(matrix) @ abs(pressure))[free]),
            np.finfo(float).tiny,
        )
    )
    return ConformingQuadrilateralSolution(mesh, degree, pressure, permeability, residual)
