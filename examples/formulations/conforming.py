"""Classical Cartesian reference equations assembled from public numerical owners."""

from typing import Any, Literal

import numpy as np

from pymhm import Equation, MultiscaleProblem, SolverConfig, assemble
from pymhm.fem.scalar.quadrilateral import quadrilateral_operators
from pymhm.fem.scalar.separable import separable_diffusion_operators
from pymhm.fem.traces.conforming import quadrilateral_boundary_data
from pymhm.materials.separable import SeparableField
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution

_UNIT = SeparableField(((1.0, 1.0),))
_ZERO = SeparableField(())


def conforming_quadrilateral(
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
    separated: bool = False,
) -> ConformingQuadrilateralSolution:
    """Compose conforming energy, natural load, essential values and physical mean.

    The entire nodal vector is global and there are no condensed local fields.
    ``separated=True`` uses declared finite factor sums with exactly the same
    tensor Gauss rule as element assembly. It performs no rank approximation.
    This editable example controller uses the generic Equation/assemble API.
    """
    operation = separable_diffusion_operators if separated else quadrilateral_operators
    matrix, mass, load = operation(
        mesh, degree, permeability=permeability, source=source, order=quadrature_order
    )
    boundary, fixed = quadrilateral_boundary_data(
        mesh, degree, dirichlet=dirichlet, neumann=neumann, order=quadrature_order
    )
    load = load + boundary
    constraints = []
    if not fixed:
        if not np.isfinite(mean_pressure):
            raise ValueError("mean pressure must be finite")
        if abs(load.sum()) > 1e-10 * max(float(np.sum(abs(load))), np.finfo(float).tiny):
            raise ValueError("pure-Neumann source and physical flux are incompatible")
        moment = np.asarray(mass @ np.ones(len(load)))
        constraints = [(moment, mean_pressure * float(mesh.areas.sum()))]
    problem = MultiscaleProblem.from_global(
        Equation(matrix, load), len(load), fixed=fixed, constraints=tuple(constraints)
    )
    system = assemble(
        problem,
        solvers=SolverConfig(
            global_solver=solver, global_refinement_precision=refinement_precision
        ),
    )
    solution = system.solve()
    return ConformingQuadrilateralSolution(
        mesh, degree, solution.trace, permeability, solution.residual
    )


def conforming_separated(
    mesh: CartesianMacroMesh,
    *,
    permeability: SeparableField = _UNIT,
    source: SeparableField = _ZERO,
    **options: Any,
) -> ConformingQuadrilateralSolution:
    """Select explicit factor assembly in the same editable reference formulation."""
    return conforming_quadrilateral(
        mesh, permeability=permeability, source=source, separated=True, **options
    )
