"""Analytical lowest-order Darcy harmonic lifts on affine simplices.

For macrocell-constant SPD permeability, the RT0 normal flux has an explicit
quadratic potential. A variable source contributes its full harmonic moments
to the global equations and a separate zero-mean Neumann source lifting.
"""

from dataclasses import dataclass as dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution as HybridSolution
from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray as FloatArray
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.metric import AnalyticDarcySpace as AnalyticDarcySpace
from pymhm.fem.scalar.metric import _rule as _rule
from pymhm.fem.scalar.metric import _volumes as _volumes
from pymhm.fem.scalar.metric import metric_quadratic_operators
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.operators import triangle_quadrature as triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.tetrahedron import tetra_tabulate as tetra_tabulate
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature as tetrahedron_quadrature
from pymhm.fem.scalar.triangle import scalar_operators
from pymhm.fem.scalar.triangle import tabulate as tabulate
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary
from pymhm.materials.evaluation import (
    scalar_values as scalar_values,
)
from pymhm.materials.evaluation import (
    scalar_values_3d as scalar_values_3d,
)
from pymhm.materials.evaluation import (
    tensor_values as tensor_values,
)
from pymhm.materials.evaluation import (
    tensor_values_3d as tensor_values_3d,
)
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.analytic import AnalyticDarcySolution as AnalyticDarcySolution


def analytic_darcy_local(
    mesh: TriangleMesh | TetraMesh,
    cell: int,
    permeability: Any = 1.0,
    source: Any = 0.0,
    *,
    quadrature_order: int = 8,
) -> tuple[LocalProblem, AnalyticDarcySpace]:
    """Compose exact metric-quadratic energy with an explicit constant kernel."""
    a, b, load, moments, space = metric_quadratic_operators(
        mesh, cell, permeability, source, quadrature_order=quadrature_order
    )
    kernel = np.eye(len(load))[:, :1]
    return LocalProblem(a, b, load, mesh.cell_faces[cell], kernel, moments), space


def solve_darcy_analytic(
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
    solver: str = "scipy",
) -> AnalyticDarcySolution:
    """Solve the lowest-order analytical MHM with optional variable-source reconstruction.

    Each face carries one constant normal-flux coefficient. K is one constant
    SPD tensor throughout the domain; different cell tensors can be supplied by
    assembling :func:`analytic_darcy_local` directly. Harmonic lifts are exact
    quadratics, independent of ``source_degree`` and ``source_refinement``.
    Constant sources have p_f=0. For callback sources, a separate continuous Pk
    zero-mean Neumann solve reconstructs p_f on each refined macrocell. The full
    source moments enter the global equations in both cases; replacing them by
    cell averages would change the method for a nonconstant source.
    """
    positive_int(source_degree, "source_degree", 1)
    positive_int(source_refinement, "source_refinement")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    local = [
        analytic_darcy_local(mesh, i, permeability, source, quadrature_order=quadrature_order)
        for i in range(len(mesh.cells))
    ]
    natural = {} if neumann is None else neumann
    if isinstance(mesh, TriangleMesh):
        boundary, fixed = boundary_data(
            SkeletonSpace(mesh), dirichlet, natural, order=quadrature_order
        )
    else:
        boundary, fixed = _boundary(TriangularSkeleton(mesh), dirichlet, natural, quadrature_order)
    system = HybridSystem([pair[0] for pair in local], boundary_load=boundary)
    moments = [problem.constraints[:, 0] for problem, _ in local]
    gauges = (
        [system.mean_constraint(moments, mean_pressure * _volumes(mesh).sum())]
        if set(natural) == set(mesh.boundary_faces)
        else None
    )
    result = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    harmonic = tuple(
        field - response.source
        for field, response in zip(result.fields, system.responses, strict=True)
    )
    source_meshes, sources = [], []
    for cell, (_, space) in enumerate(local):
        fine = mesh.submesh(cell, source_refinement if callable(source) else 1)
        values = None
        if callable(source):
            if isinstance(fine, TriangleMesh):
                matrix, mass, load = scalar_operators(
                    fine,
                    source_degree,
                    diffusion=space.permeability,
                    source=source,
                    order=quadrature_order,
                )
            else:
                matrix, mass, load = tetra_operators(
                    fine,
                    source_degree,
                    diffusion=space.permeability,
                    source=source,
                    order=quadrature_order,
                )
            kernel = np.ones((len(load), 1))
            values = (
                LocalProblem(
                    matrix,
                    np.empty((len(load), 0)),
                    load,
                    np.empty(0, dtype=int),
                    kernel,
                    mass @ kernel,
                )
                .condense()
                .source
            )
        source_meshes.append(fine)
        sources.append(values)
    return AnalyticDarcySolution(
        mesh,
        tuple(space for _, space in local),
        harmonic,
        tuple(source_meshes),
        tuple(sources),
        source_degree,
        result,
    )
