"""Estimator-driven conforming tetrahedral MHM adaptation with physical boundary transfer."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.darcy.primal_3d import Darcy3DSolution, solve_darcy_3d
from pymhm.adaptivity.darcy import mark_dorfler
from pymhm.core.validation import FloatArray, positive_int
from pymhm.estimators.darcy_3d import Darcy3DEstimator, estimate_darcy_error_3d
from pymhm.fem.conditions import minimum_estimator_degree
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.refinement_3d import TetraRefinement, refine_tetrahedra
from pymhm.meshes.tetrahedron import TetraMesh


@dataclass(frozen=True)
class AdaptiveDarcy3DResult:
    """Solved 3D states, indicators, markings and exact refinement ancestry."""

    solutions: tuple[Darcy3DSolution, ...]
    estimators: tuple[Darcy3DEstimator, ...]
    marked: tuple[np.ndarray, ...]
    refinements: tuple[TetraRefinement, ...]

    @property
    def totals(self) -> FloatArray:
        """Return the selected indicator at each completed state."""
        return np.array([estimate.total for estimate in self.estimators])


def solve_adaptive_darcy_3d(
    mesh: TetraMesh,
    *,
    iterations: int = 5,
    theta: float = 0.5,
    tolerance: float = 0.0,
    maximum_cells: int = 10000,
    trace_degree: int = 0,
    trace_subdivisions: int = 1,
    reconstruction_degree: int = 1,
    estimator_order: int = 7,
    estimator_convention: Literal["energy", "published"] = "energy",
    ellipticity_lower_bound: Any = None,
    on_state: Callable[[int, Darcy3DSolution, Darcy3DEstimator], None] | None = None,
    **problem: Any,
) -> AdaptiveDarcy3DResult:
    """Solve, estimate, mark and bisect tetrahedral edge stars.

    The marking policy is original, using the dimension-independent estimator of
    [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073) and Dörfler bulk sets. It is not
    a reproduction of a historical 3D mesh sequence and implies no contraction or optimality
    guarantee. Physical Neumann data and per-cell ellipticity bounds follow their exact parent
    entities. The callback persists each solved state before refinement.
    """
    positive_int(iterations, "iterations")
    positive_int(maximum_cells, "maximum cells")
    if np.iscomplexobj(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    mark_dorfler([1.0], theta)
    parameters = dict(problem)
    if "skeleton" in parameters:
        raise ValueError("adaptive tetrahedral traces are selected by trace_degree/subdivisions")
    parameters.setdefault("degree", minimum_estimator_degree(trace_degree, 3))
    parameters.setdefault("quadrature_order", estimator_order)
    bound = ellipticity_lower_bound
    solutions, estimates, marks, refinements = [], [], [], []
    for step in range(iterations):
        skeleton = TriangularSkeleton(mesh, trace_subdivisions, degree=trace_degree)
        solution = solve_darcy_3d(mesh, skeleton=skeleton, **parameters)
        estimate = estimate_darcy_error_3d(
            solution,
            degree=reconstruction_degree,
            dirichlet=parameters.get("dirichlet", 0),
            neumann=parameters.get("neumann"),
            ellipticity_lower_bound=bound,
            convention=estimator_convention,
            quadrature_order=estimator_order,
        )
        solutions.append(solution)
        estimates.append(estimate)
        marked = mark_dorfler(estimate.local_squared, theta)
        marks.append(marked)
        if on_state is not None:
            on_state(step, solution, estimate)
        if estimate.total <= tolerance or not np.any(marked) or step + 1 == iterations:
            break
        refined = refine_tetrahedra(mesh, marked)
        if len(refined.mesh.cells) > maximum_cells:
            break
        natural = parameters.get("neumann")
        if natural is not None:
            parameters["neumann"] = {
                int(face): natural[int(parent)]
                for face, parent in enumerate(refined.face_parents)
                if int(parent) in natural
            }
        if bound is not None and np.ndim(bound):
            bound = np.asarray(bound)[refined.cell_parents]
        refinements.append(refined)
        mesh = refined.mesh
    return AdaptiveDarcy3DResult(
        tuple(solutions), tuple(estimates), tuple(marks), tuple(refinements)
    )
