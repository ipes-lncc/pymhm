"""Conforming Cartesian diffusion with finite sums of separable scalar fields.

Kronecker assembly is algebraically the same tensor Gauss Galerkin operator as
elementwise Qk assembly. It avoids evaluating two-dimensional coefficient arrays
and dense element matrices when a coefficient is explicitly given in separated
form. This module does not approximate a nonseparable field by a low-rank fit.
"""

from dataclasses import dataclass as dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss as leggauss
from scipy import sparse as sparse

from pymhm.core.validation import FloatArray as FloatArray
from pymhm.core.validation import positive_int as positive_int
from pymhm.fem.scalar.quadrilateral import qk_basis as qk_basis
from pymhm.fem.scalar.separable import (
    interval_nodal_quadrature,
    interval_weighted_operators,
    require_positive_separated,
)
from pymhm.fem.scalar.separable import (
    separable_diffusion_operators as separable_diffusion_operators,
)
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values
from pymhm.materials.separable import SeparableField as SeparableField
from pymhm.materials.separable import factor_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution

_UNIT = SeparableField(((1.0, 1.0),))
_ZERO = SeparableField(())


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
    boundary conditions use the compatibility function
    :func:`pymhm._legacy.models.darcy.conforming.solve_conforming_quadrilateral`.
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


_line_data = interval_nodal_quadrature
_line_operators = interval_weighted_operators
_positive = require_positive_separated
_factor = factor_values
