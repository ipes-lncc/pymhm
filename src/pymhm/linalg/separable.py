"""Matrix-free Cartesian Qk diffusion with a low-order refined AMG preconditioner.

The solved operator retains the explicitly separated coefficient and the Qk
Gauss rule. A Q1 problem on the same equidistant nodal grid supplies only the
preconditioner: it never replaces the high-order bilinear form or right-hand side.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.separable import (
    interval_nodal_quadrature as _line_data,
)
from pymhm.fem.scalar.separable import (
    interval_weighted_operators as _line_operators,
)
from pymhm.fem.scalar.separable import (
    require_positive_separated as _positive,
)
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError, _optional, _tolerances
from pymhm.materials.evaluation import scalar_values
from pymhm.materials.separable import SeparableField
from pymhm.materials.separable import factor_values as _factor
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution


@dataclass(frozen=True)
class TensorDiffusionOperator:
    """Store one-dimensional Galerkin factors and apply their exact Kronecker sum.

    Terms are ``(y_factor, x_factor)`` and vectors use x-fast ordering. Restricting
    both one-dimensional factors to their interior nodes imposes homogeneous
    strong Dirichlet conditions without constructing a two-dimensional matrix.
    """

    terms: tuple[tuple[sparse.csr_matrix, sparse.csr_matrix], ...]
    load: FloatArray
    node_shape: tuple[int, int]

    def apply(self, values: FloatArray) -> FloatArray:
        """Evaluate the matrix action, preserving an explicitly wider input dtype."""
        values = np.asarray(values)
        if values.shape != (int(np.prod(self.node_shape)),) or not np.isfinite(values).all():
            raise ValueError("tensor operator input has invalid shape or nonfinite entries")
        field = values.reshape(self.node_shape)
        result = np.zeros_like(field, dtype=np.result_type(values.dtype, float))
        for y, x in self.terms:
            result += y @ (x @ field.T).T
        return result.ravel()

    def interior(self) -> "TensorDiffusionOperator":
        """Remove the exterior nodal lines while retaining the physical load ordering."""
        terms = tuple((y[1:-1, 1:-1], x[1:-1, 1:-1]) for y, x in self.terms)
        return TensorDiffusionOperator(
            terms,
            self.load.reshape(self.node_shape)[1:-1, 1:-1].ravel(),
            (self.node_shape[0] - 2, self.node_shape[1] - 2),
        )

    def assemble(self) -> sparse.csr_matrix:
        """Materialize the same operator when a sparse matrix is explicitly required."""
        size = int(np.prod(self.node_shape))
        result = sparse.csr_matrix((size, size))
        for y, x in self.terms:
            result = result + sparse.kron(y, x, format="csr")
        result.eliminate_zeros()
        return result


def tensor_diffusion_operator(
    counts: tuple[int, int],
    bounds: tuple[float, float, float, float],
    degree: int,
    *,
    permeability: SeparableField,
    source: SeparableField,
    order: int,
) -> TensorDiffusionOperator:
    """Construct Qk factors without allocating mesh connectivity or a global matrix.

    ``counts`` are the two cell counts and ``bounds`` are ``(x0,x1,y0,y1)``.
    Positive scalar permeability is checked at the declared tensor Gauss points.
    """
    if len(counts) != 2:
        raise ValueError("counts must contain two positive integers")
    nx, ny = (positive_int(value, "cell count") for value in counts)
    raw = np.asarray(bounds)
    if np.iscomplexobj(raw) or raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError("bounds must contain four finite real coordinates")
    x0, x1, y0, y1 = np.asarray(raw, dtype=float)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bounds must have positive width and height")
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "quadrature order"), degree + 1)
    if not isinstance(permeability, SeparableField) or not permeability.terms:
        raise ValueError("permeability requires a nonempty SeparableField")
    if not isinstance(source, SeparableField):
        raise TypeError("source must be a SeparableField")
    axes = (_line_data(nx, (x0, x1), degree, order), _line_data(ny, (y0, y1), degree, order))
    evaluated = [
        tuple(
            _factor(factor, axis[0].ravel()).reshape(axis[0].shape)
            for factor, axis in zip(term, axes, strict=True)
        )
        for term in permeability.terms
    ]
    _positive(evaluated)
    terms: list[tuple[sparse.csr_matrix, sparse.csr_matrix]] = []
    for pair in evaluated:
        x, y = (_line_operators(axis, value) for axis, value in zip(axes, pair, strict=True))
        terms.extend(((y[0], x[1]), (y[1], x[0])))
    shape = (ny * degree + 1, nx * degree + 1)
    load = np.zeros(int(np.prod(shape)))
    for pair in source.terms:
        x, y = (
            _line_operators(axis, _factor(factor, axis[0].ravel()).reshape(axis[0].shape))[2]
            for factor, axis in zip(pair, axes, strict=True)
        )
        load += np.kron(y, x)
    if not np.isfinite(load).all() or any(
        not np.isfinite(matrix.data).all() for pair in terms for matrix in pair
    ):
        raise ValueError("tensor operator and load must remain finite")
    return TensorDiffusionOperator(tuple(terms), load, shape)


@dataclass(frozen=True)
class SeparableKrylovSolution(ConformingQuadrilateralSolution):
    """Conforming solution with the measured iteration count and true residual criterion."""

    iterations: int
    relative_equation_residual: float
    preconditioner_levels: int


def solve_separable_krylov(
    mesh: CartesianMacroMesh,
    *,
    degree: int,
    permeability: SeparableField,
    source: SeparableField,
    dirichlet: Any = 0.0,
    quadrature_order: int = 6,
    preconditioner_order: int = 2,
    rtol: float = 1e-10,
    maxiter: int = 1000,
    refinement_precision: Literal["double", "extended"] = "double",
) -> SeparableKrylovSolution:
    """Solve the original Qk operator with CG and a Q1 refined-grid AMG V-cycle.

    Both Galerkin matrices are symmetric positive definite after full exterior
    Dirichlet elimination, for positive sampled permeability. Symmetric AMG
    smoothing defines the CG preconditioner. This does not assert an arbitrary-
    contrast or arbitrary-degree condition-number bound. The independently
    accumulated true residual must satisfy ``||b-Au|| <= rtol*||b||``. Up to two
    correction solves reuse the hierarchy; the explicit extended mode retains
    correction digits in a wider long-double type and never changes this test.
    """
    if not isinstance(mesh, CartesianMacroMesh):
        raise TypeError("separable Krylov diffusion requires CartesianMacroMesh")
    _tolerances(rtol, 0.0)
    maxiter = positive_int(maxiter, "maxiter")
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise SolverUnavailableError("extended refinement requires a wider long-double type")
    full = tensor_diffusion_operator(
        (mesh.nx, cast(int, mesh.ny)),
        mesh.bounds,
        degree,
        permeability=permeability,
        source=source,
        order=quadrature_order,
    )
    ny, nx = full.node_shape
    dtype = np.longdouble if refinement_precision == "extended" else float
    pressure = np.zeros(full.node_shape, dtype=dtype)
    fixed = np.unique(
        np.r_[
            np.arange(nx),
            (ny - 1) * nx + np.arange(nx),
            nx * np.arange(ny),
            nx * np.arange(ny) + nx - 1,
        ]
    )
    points = (
        np.array(mesh.bounds)[[0, 2]]
        + np.column_stack((fixed % nx, fixed // nx)) * mesh.spacing / degree
    )
    pressure.ravel()[fixed] = scalar_values(dirichlet, points)
    if min(nx, ny) <= 2:
        return SeparableKrylovSolution(mesh, degree, pressure.ravel(), permeability, 0.0, 0, 0.0, 0)
    operator = full.interior()
    rhs = (
        (full.load.astype(dtype) - full.apply(pressure.ravel()))
        .reshape(full.node_shape)[1:-1, 1:-1]
        .ravel()
    )
    size = len(rhs)
    low = (
        tensor_diffusion_operator(
            (mesh.nx * degree, cast(int, mesh.ny) * degree),
            mesh.bounds,
            1,
            permeability=permeability,
            source=SeparableField(()),
            order=preconditioner_order,
        )
        .interior()
        .assemble()
    )
    pyamg = _optional("pyamg", "Install pyamg for low-order refined AMG preconditioning.")
    hierarchy = pyamg.smoothed_aggregation_solver(
        low,
        symmetry="hermitian",
        smooth=("jacobi", {"weighting": "local"}),
        presmoother=("gauss_seidel", {"sweep": "symmetric"}),
        postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
    )
    preconditioner = hierarchy.aspreconditioner(cycle="V")
    linear = LinearOperator((size, size), matvec=operator.apply, dtype=float)
    iterations = 0

    def count_iteration(_: FloatArray) -> None:
        """Count all CG iterations, including explicitly checked correction solves."""
        nonlocal iterations
        iterations += 1

    result = np.zeros(size, dtype=dtype)
    residual = rhs.copy()
    norm_rhs = np.linalg.norm(rhs)
    threshold = rtol * norm_rhs
    for _ in range(3):
        correction, status = cg(
            linear,
            np.asarray(residual, dtype=float),
            M=preconditioner,
            rtol=max(rtol, np.finfo(float).eps),
            atol=0.0,
            maxiter=maxiter,
            callback=count_iteration,
        )
        if status != 0:
            raise LinearSolveError(f"separable CG failed to converge (info={status})")
        result += correction
        precision = np.longdouble if np.finfo(np.longdouble).eps < np.finfo(float).eps else float
        residual = np.asarray(rhs, dtype=precision) - operator.apply(
            np.asarray(result, dtype=precision)
        )
        if np.linalg.norm(residual) <= threshold:
            break
    residual_norm = np.linalg.norm(residual)
    if not np.isfinite(residual_norm) or residual_norm > threshold:
        raise LinearSolveError(f"separable CG true residual {residual_norm} exceeds {threshold}")
    pressure[1:-1, 1:-1] = result.reshape(ny - 2, nx - 2)
    relative = float(residual_norm / norm_rhs) if norm_rhs else 0.0
    return SeparableKrylovSolution(
        mesh,
        degree,
        pressure.ravel(),
        permeability,
        relative,
        iterations,
        relative,
        len(hierarchy.levels),
    )
