"""Adaptive orchestration consuming an application's mathematical equation factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from examples.formulations.darcy import DarcyDefinition, pressure_constraints, recover_darcy
from pymhm.adaptivity.darcy import AdaptiveDarcyResult, solve_adaptive_darcy
from pymhm.core.multiscale import assemble
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import DarcySolution


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
    """Compose generic equation assembly with the shared Darcy adaptive policy.

    The user callback owns the physical forms on each new mesh. This introductory
    adapter supports homogeneous Dirichlet pressure. The package adaptive owner
    supplies marking, mesh ancestry, estimator validation and stopping rules;
    it receives the generic assembly callback through ``solve_step``.
    """
    space = FaceSpace.uniform(0) if trace_space is None else trace_space
    skeleton = SkeletonSpace(mesh, tuple(space for _ in mesh.faces))
    names = {
        "convention": "estimator_convention",
        "degree": "reconstruction_degree",
        "quadrature_order": "estimator_order",
        "ellipticity_lower_bound": "ellipticity_lower_bound",
        "backend": "estimator_backend",
        "workers": "estimator_workers",
    }
    raw = {} if estimator_options is None else dict(estimator_options)
    unknown = raw.keys() - names.keys()
    if unknown:
        raise TypeError(f"unsupported introductory estimator options: {sorted(unknown)}")
    options = {names[name]: value for name, value in raw.items()}

    def solve_step(
        current: TriangleMesh, *, skeleton: SkeletonSpace, **problem: Any
    ) -> DarcySolution:
        """Assemble and interpret the explicitly supplied homogeneous primal forms."""
        declaration = definition_factory(current, skeleton)
        if declaration.formulation != "primal":
            raise ValueError("the adaptive energy indicator requires a primal declaration")
        if declaration.problem.fixed or np.any(declaration.problem.global_equation.L):
            raise ValueError("this introductory adapter requires homogeneous Dirichlet data")
        system = assemble(declaration.problem)
        return recover_darcy(
            declaration,
            system,
            system.solve(constraints=pressure_constraints(declaration, system)),
        )

    return solve_adaptive_darcy(
        mesh,
        skeleton=skeleton,
        trace_space=space,
        solve_step=solve_step,
        iterations=iterations,
        theta=theta,
        tolerance=tolerance,
        maximum_cells=maximum_cells,
        **options,
    )
