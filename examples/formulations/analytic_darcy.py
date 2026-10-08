"""Declare metric-quadratic Darcy harmonics and their independent source potential.

The local polynomial energy, face moments and source integrals are public
numerical operations. The application explicitly selects the constant kernel,
normal-flux convention, weak boundary equation and physical pressure gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from pymhm import Equation, LocalEquations, MultiscaleProblem, assemble
from pymhm.core.assembly import SolverConfig
from pymhm.core.condensation import condense_local
from pymhm.core.contracts import LocalProblem
from pymhm.core.multiscale import MultiscaleSolution, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.metric import metric_quadratic_operators
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.triangle import scalar_operators
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_boundary_data
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.analytic import AnalyticDarcySolution


@dataclass(frozen=True)
class AnalyticDarcyDefinition:
    """Polynomial equations and the independently specified source discretization."""

    problem: MultiscaleProblem[int]
    mesh: TriangleMesh | TetraMesh
    source_degree: int
    pressure_integral: float | None


def analytic_equations(
    cell: int,
    *,
    mesh: TriangleMesh | TetraMesh,
    permeability: Any,
    source: Any,
    order: int,
    source_degree: int,
    source_refinement: int,
) -> LocalEquations:
    """Declare exact harmonic moments and a separate mean-zero Neumann response.

    The polynomial source response enters the global equation with all of its
    moments. It is subtracted during physical recovery and replaced by the full
    nodal source potential. Constant sources require no separate source field.
    All fine operators and the additional constrained solve run in this worker.
    """
    a, b, load, moments, space = metric_quadratic_operators(
        mesh, cell, permeability, source, quadrature_order=order
    )
    fine = mesh.submesh(cell, source_refinement if callable(source) else 1)
    potential = None
    if callable(source):
        operation = scalar_operators if isinstance(fine, TriangleMesh) else tetra_operators
        matrix, mass, force = operation(
            fine, source_degree, diffusion=space.permeability, source=source, order=order
        )
        constant = np.ones((len(force), 1))
        source_problem = LocalProblem(
            matrix,
            np.empty((len(force), 0)),
            force,
            np.empty(0, dtype=int),
            kernel=constant,
            constraints=mass @ constant,
        )
        potential = condense_local(source_problem).source
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        mesh.cell_faces[cell],
        kernel=np.eye(len(load))[:, :1],
        moments=moments,
        metadata=(space, fine, potential, moments[:, 0]),
    )


def define_analytic_darcy(
    mesh: TriangleMesh | TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    source_degree: int = 2,
    source_refinement: int = 2,
    quadrature_order: int = 8,
) -> AnalyticDarcyDefinition:
    """Declare constant normal traces on affine 2D/3D simplices.

    The permeability is a constant SPD tensor. The harmonic span and its metric
    units are independent of the separate nodal source resolution. A pure
    Neumann boundary fixes the physical pressure integral; other boundaries
    determine the pressure level through their weak value moments.
    """
    degree = positive_int(source_degree, "source_degree")
    refinement = positive_int(source_refinement, "source_refinement")
    order = positive_int(quadrature_order, "quadrature_order", 3)
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    natural = {} if neumann is None else dict(neumann)
    if isinstance(mesh, TriangleMesh):
        boundary, fixed = boundary_data(SkeletonSpace(mesh), dirichlet, natural, order=order)
        volume = float(mesh.areas.sum())
    elif isinstance(mesh, TetraMesh):
        boundary, fixed = tetra_boundary_data(TriangularSkeleton(mesh), dirichlet, natural, order)
        volume = float(mesh.volumes.sum())
    else:
        raise TypeError("analytic Darcy requires an affine triangle or tetrahedron mesh")
    provider = partial(
        analytic_equations,
        mesh=mesh,
        permeability=permeability,
        source=source,
        order=order,
        source_degree=degree,
        source_refinement=refinement,
    )
    problem = MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        len(boundary),
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    integral = mean_pressure * volume if set(natural) == set(mesh.boundary_faces) else None
    return AnalyticDarcyDefinition(problem, mesh, degree, integral)


def analytic_constraints(
    definition: AnalyticDarcyDefinition, system: MultiscaleSystem
) -> list[tuple[Any, float]]:
    """Build the physical mean from the actual metric-basis integrals."""
    if definition.pressure_integral is None:
        return []
    return [
        system.mean_constraint(
            [metadata[3] for metadata in system.local_metadata], definition.pressure_integral
        )
    ]


def recover_analytic_darcy(
    definition: AnalyticDarcyDefinition,
    system: MultiscaleSystem,
    solution: MultiscaleSolution,
) -> AnalyticDarcySolution:
    """Interpret harmonic plus full source fields without replacing source moments."""
    harmonic = tuple(
        field - response.source
        for field, response in zip(solution.fields, system.responses, strict=True)
    )
    return AnalyticDarcySolution(
        definition.mesh,
        tuple(metadata[0] for metadata in system.local_metadata),
        harmonic,
        tuple(metadata[1] for metadata in system.local_metadata),
        tuple(metadata[2] for metadata in system.local_metadata),
        definition.source_degree,
        solution,
    )


def analytic_darcy(
    mesh: TriangleMesh | TetraMesh, *, solver: str = "scipy", **options: Any
) -> AnalyticDarcySolution:
    """Execute this example's explicit equations with the generic assembly and solve API."""
    definition = define_analytic_darcy(mesh, **options)
    system = assemble(definition.problem, solvers=SolverConfig(global_solver=solver))
    solution = system.solve(constraints=analytic_constraints(definition, system))
    return recover_analytic_darcy(definition, system, solution)
