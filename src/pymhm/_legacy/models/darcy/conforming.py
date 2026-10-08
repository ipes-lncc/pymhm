"""Classical conforming Cartesian Qk diffusion for independently refined baselines."""

from dataclasses import dataclass as dataclass
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss as leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray as FloatArray
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.quadrilateral import qk_basis as qk_basis
from pymhm.fem.scalar.quadrilateral import qk_space, quadrilateral_operators
from pymhm.fem.traces.conforming import quadrilateral_boundary_data
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values as scalar_values
from pymhm.materials.evaluation import tensor_values as tensor_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import (
    ConformingQuadrilateralSolution as ConformingQuadrilateralSolution,
)


def solve_conforming_quadrilateral(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: int = 6,
    solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> ConformingQuadrilateralSolution:
    """Solve classical conforming diffusion with strong Dirichlet boundary data.

    Neumann entries are outward physical fluxes on boundary face indices of
    ``mesh``. The remaining exterior faces carry Dirichlet data. Pure Neumann
    problems retain the constant kernel through a physical mean constraint and
    reject incompatible total load. This solver has no MHM skeleton or local
    condensation; its entire conforming matrix is assembled and solved globally.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 1)
    matrix, mass, load = quadrilateral_operators(
        mesh, degree, permeability=permeability, source=source, order=order
    )
    _, nodes = qk_space(mesh, degree)
    boundary, prescribed = quadrilateral_boundary_data(
        mesh, degree, dirichlet=dirichlet, neumann=neumann, order=order
    )
    load += boundary
    fixed = np.zeros(len(nodes), dtype=bool)
    fixed[list(prescribed)] = True
    pressure = np.zeros(len(nodes))
    if fixed.any():
        pressure[list(prescribed)] = list(prescribed.values())
        free = np.flatnonzero(~fixed)
        if len(free):
            solved = solve_linear(
                matrix[free][:, free],
                (load - matrix @ pressure)[free],
                solver=solver,
                refinement_precision=refinement_precision,
            )
            pressure = pressure.astype(solved.dtype)
            pressure[free] = solved
    else:
        if not np.isfinite(mean_pressure):
            raise ValueError("mean pressure must be finite")
        if abs(load.sum()) > 1e-10 * max(float(np.sum(abs(load))), np.finfo(float).tiny):
            raise ValueError("pure-Neumann source and physical flux are incompatible")
        mean = np.asarray(mass @ np.ones(len(nodes)))
        augmented = sparse.bmat([[matrix, mean[:, None]], [mean[None], None]], format="csc")
        pressure = solve_linear(
            augmented,
            np.r_[load, mean_pressure * mesh.areas.sum()],
            solver=solver,
            refinement_precision=refinement_precision,
        )[:-1]
        free = np.arange(len(nodes))
    residual = float(
        np.linalg.norm((matrix @ pressure - load)[free])
        / max(
            np.linalg.norm(load[free]),
            np.linalg.norm((abs(matrix) @ abs(pressure))[free]),
            np.finfo(float).tiny,
        )
    )
    return ConformingQuadrilateralSolution(mesh, degree, pressure, permeability, residual)
