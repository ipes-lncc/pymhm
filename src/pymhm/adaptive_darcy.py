"""Estimator-driven conforming macro refinement for primal Darcy problems."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np

from pymhm.darcy import DarcySolution, solve_darcy
from pymhm.estimator_spaces import minimum_estimator_degree
from pymhm.mesh import FaceSpace, FloatArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.refinement import TriangleRefinement, refine_triangles, transfer_skeleton
from pymhm.weighted_estimator import WeightedDarcyEstimator, estimate_darcy_indicator


def mark_dorfler(local_squared: Any, theta: float = 0.5) -> np.ndarray:
    """Select a minimal-cardinality deterministic bulk set of squared indicators.

    The selected sum is at least ``theta`` times the total sum. Ties follow
    original cell order. Zero indicators produce an empty set. Scaling before
    summation avoids overflow and preserves the marking for extreme units.
    """
    raw = np.asarray(local_squared)
    if (
        np.iscomplexobj(raw)
        or raw.ndim != 1
        or not len(raw)
        or not np.isfinite(raw).all()
        or np.any(raw < 0)
    ):
        raise ValueError("local_squared must be a nonempty finite nonnegative real vector")
    if not isinstance(theta, Real) or not np.isfinite(theta) or not 0 < float(theta) <= 1:
        raise ValueError("theta must belong to (0,1]")
    marked = np.zeros(len(raw), dtype=bool)
    maximum = float(np.max(raw))
    if maximum == 0:
        return marked
    normalized = raw / maximum
    order = np.argsort(-normalized, kind="stable")
    cumulative = np.cumsum(normalized[order])
    count = min(len(raw), int(np.searchsorted(cumulative, theta * cumulative[-1])) + 1)
    marked[order[:count]] = True
    return marked


@dataclass(frozen=True)
class AdaptiveDarcyResult:
    """Solved states, estimators, marking decisions and actual refinement maps."""

    solutions: tuple[DarcySolution, ...]
    estimators: tuple[WeightedDarcyEstimator, ...]
    marked: tuple[np.ndarray, ...]
    refinements: tuple[TriangleRefinement, ...]

    @property
    def totals(self) -> FloatArray:
        """Return the selected estimator or numerical indicator at every solved level."""
        return np.asarray([estimate.total for estimate in self.estimators])


def solve_adaptive_darcy(
    mesh: TriangleMesh,
    *,
    iterations: int = 5,
    theta: float = 0.5,
    tolerance: float = 0.0,
    maximum_cells: int = 10000,
    trace_degree: int = 0,
    trace_segments: int = 1,
    reconstruction_degree: int = 1,
    estimator_order: int = 8,
    estimator_convention: Literal["published", "energy"] = "energy",
    estimator_backend: Literal["serial", "thread", "process"] = "serial",
    estimator_workers: int | None = None,
    ellipticity_lower_bound: Any = None,
    skeleton: SkeletonSpace | None = None,
    local_mesh_factory: Callable[[TriangleMesh, int], TriangleMesh] | None = None,
    macro_refiner: Callable[[TriangleMesh, Any], TriangleRefinement] = refine_triangles,
    **problem: Any,
) -> AdaptiveDarcyResult:
    """Solve, estimate, mark and red-green-refine a primal Darcy macro mesh.

    This loop uses the weighted L09 energy decomposition and Dörfler bulk marking;
    it is an original adaptive policy, not a reproduction of a published marking
    schedule. All estimator assumptions remain required: polynomial represented
    boundary data, conforming fine triangulations, certified ellipticity and
    ``local degree >= trace degree + 2``. Local refinement and degree stay fixed.
    Neumann face data and macro-local ellipticity bounds follow exact ancestry.
    ``iterations`` counts solves. A proposed refinement exceeding
    ``maximum_cells`` is not solved. No contraction or optimality theorem is
    inferred from the loop, and indicator changes need not be monotone.
    ``local_mesh_factory(macro_mesh, cell_index)`` can construct material-fitted
    local partitions afresh at each level. Static ``local_meshes`` are rejected
    because their ancestry changes after refinement.
    ``macro_refiner`` defaults to red-green refinement. Passing
    ``refine_longest_edge`` selects conforming longest-edge propagation; either
    operation must provide cell/face ancestors for boundary and trace transfer.
    ``estimator_convention='published'`` selects the literal numerical terms
    (5.3)--(5.7); the default ``'energy'`` uses physical material weights.
    The printed convention does not imply a general-SPD energy upper bound.
    ``estimator_backend`` and ``estimator_workers`` select independent execution
    of local reconstruction and indicator terms separately from the PDE solver.
    """
    positive_int(iterations, "iterations")
    positive_int(maximum_cells, "maximum_cells")
    if not np.isreal(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    mark_dorfler([1.0], theta)
    space = FaceSpace.uniform(trace_degree, trace_segments)
    skeleton = (
        SkeletonSpace(mesh, tuple(space for _ in mesh.faces)) if skeleton is None else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("adaptive Darcy requires a scalar skeleton on the supplied mesh")
    parameters = dict(problem)
    if "local_meshes" in parameters:
        raise ValueError("use local_mesh_factory for adaptive local partitions")
    if parameters.get("formulation", "primal") != "primal":
        raise ValueError("the adaptive energy estimator requires primal Darcy")
    parameters.setdefault("degree", minimum_estimator_degree(trace_degree, 2))
    parameters.setdefault("quadrature_order", estimator_order)
    solutions, estimates, marks, refinements = [], [], [], []
    bound = ellipticity_lower_bound
    for step in range(iterations):
        if local_mesh_factory is not None:
            parameters["local_meshes"] = tuple(
                local_mesh_factory(mesh, cell) for cell in range(len(mesh.cells))
            )
        solution = solve_darcy(mesh, skeleton=skeleton, **parameters)
        estimate = estimate_darcy_indicator(
            solution,
            convention=estimator_convention,
            degree=reconstruction_degree,
            dirichlet=parameters.get("dirichlet", 0.0),
            neumann=parameters.get("neumann"),
            ellipticity_lower_bound=bound,
            quadrature_order=estimator_order,
            backend=estimator_backend,
            workers=estimator_workers,
        )
        solutions.append(solution)
        estimates.append(estimate)
        marked = mark_dorfler(estimate.local_squared, theta)
        marks.append(marked)
        if estimate.total <= tolerance or not np.any(marked):
            break
        if step + 1 == iterations:
            continue
        refined = macro_refiner(mesh, marked)
        if len(refined.mesh.cells) > maximum_cells:
            break
        skeleton = transfer_skeleton(skeleton, refined, new_face=space)
        natural = parameters.get("neumann")
        if natural is not None:
            parameters["neumann"] = {
                int(face): natural[int(refined.parent_faces[face])]
                for face in refined.mesh.boundary_faces
                if int(refined.parent_faces[face]) in natural
            }
        if bound is not None and np.asarray(bound).ndim == 1:
            bound = np.asarray(bound)[refined.parent_cells]
        mesh = refined.mesh
        refinements.append(refined)
    return AdaptiveDarcyResult(tuple(solutions), tuple(estimates), tuple(marks), tuple(refinements))
