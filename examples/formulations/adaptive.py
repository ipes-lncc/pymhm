"""Example adaptive orchestration consuming an application equation factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from examples.formulations.darcy import DarcyDefinition, pressure_constraints, recover_darcy
from pymhm.adaptivity.darcy import AdaptiveDarcyResult, mark_dorfler
from pymhm.core.multiscale import assemble
from pymhm.core.validation import positive_int
from pymhm.estimators.darcy_energy import estimate_darcy_indicator
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.refinement import refine_triangles, transfer_skeleton
from pymhm.meshes.triangle import TriangleMesh


def refine_declared_pressure(
    mesh: TriangleMesh,
    definition_factory: Callable[[TriangleMesh, SkeletonSpace], DarcyDefinition],
    *,
    iterations: int = 2,
    theta: float = 0.5,
    tolerance: float = 0.0,
    maximum_cells: int = 10000,
    trace_space: FaceSpace | None = None,
    estimator_options: dict[str, Any] | None = None,
) -> AdaptiveDarcyResult:
    """Solve declared equations, estimate, mark and conformingly refine macrotriangles.

    The user callback owns the physical forms and data on each new mesh. This
    introductory orchestration supports homogeneous Dirichlet pressure, keeping
    boundary data independent of face ancestry. The shared Dörfler, estimator
    and mesh-transfer kernels own their numerical decisions. Iterations count
    actual solves; indicator changes do not imply contraction or optimality.
    """
    positive_int(iterations, "iterations")
    positive_int(maximum_cells, "maximum_cells")
    if not np.isreal(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    mark_dorfler([1.0], theta)
    space = FaceSpace.uniform(0) if trace_space is None else trace_space
    skeleton = SkeletonSpace(mesh, tuple(space for _ in mesh.faces))
    options: dict[str, Any] = {"convention": "energy", "degree": 1, "quadrature_order": 8}
    options.update({} if estimator_options is None else estimator_options)
    solutions, estimates, marks, refinements = [], [], [], []
    for step in range(iterations):
        declaration = definition_factory(mesh, skeleton)
        if declaration.formulation != "primal":
            raise ValueError("the adaptive energy indicator requires a primal declaration")
        if declaration.problem.fixed or np.any(declaration.problem.global_equation.L):
            raise ValueError("this introductory adaptive loop requires homogeneous Dirichlet data")
        system = assemble(declaration.problem)
        solution = recover_darcy(
            declaration,
            system,
            system.solve(constraints=pressure_constraints(declaration, system)),
        )
        estimate = estimate_darcy_indicator(solution, **options)
        solutions.append(solution)
        estimates.append(estimate)
        marked = mark_dorfler(estimate.local_squared, theta)
        marks.append(marked)
        if estimate.total <= tolerance or not np.any(marked):
            break
        if step + 1 == iterations:
            continue
        refined = refine_triangles(mesh, marked)
        if len(refined.mesh.cells) > maximum_cells:
            break
        skeleton = transfer_skeleton(skeleton, refined, new_face=space)
        mesh = refined.mesh
        refinements.append(refined)
    return AdaptiveDarcyResult(tuple(solutions), tuple(estimates), tuple(marks), tuple(refinements))
