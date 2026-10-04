"""Explicit local/macro refinement balance for the reconstructed Darcy estimator."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.darcy.primal import DarcySolution
from pymhm.adaptivity.darcy import AdaptiveDarcyResult, solve_adaptive_darcy
from pymhm.core.validation import positive_int
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.refinement import TriangleRefinement, refine_triangles, transfer_skeleton
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class BalancedDarcyResult:
    """Solved states and the declared refinement decisions, including budget termination."""

    result: AdaptiveDarcyResult
    local_refinements: tuple[int, ...]
    decisions: tuple[str, ...]
    stop_reason: str
    local_indicators: tuple[float, ...]


def solve_balanced_adaptive_darcy(
    mesh: TriangleMesh,
    *,
    iterations: int = 5,
    local_refinement: int = 1,
    maximum_local_refinement: int = 8,
    local_error_ratio: float = 1.0,
    theta: float = 0.5,
    tolerance: float = 0.0,
    maximum_cells: int = 10000,
    trace_degree: int = 0,
    trace_segments: int = 1,
    reconstruction_degree: int = 2,
    estimator_order: int = 8,
    estimator_convention: Literal["published", "energy"] = "energy",
    ellipticity_lower_bound: Any = None,
    skeleton: SkeletonSpace | None = None,
    local_mesh_factory: Callable[[TriangleMesh, int, int], TriangleMesh] | None = None,
    local_error_indicator: Callable[[DarcySolution], float] | None = None,
    on_state: Callable[[BalancedDarcyResult], None] | None = None,
    macro_refiner: Callable[[TriangleMesh, Any], TriangleRefinement] = refine_triangles,
    **problem: Any,
) -> BalancedDarcyResult:
    """Alternate globally uniform local refinement and marked macro refinement.

    Compare ``norm(eta_div + eta_osc)`` with ``local_error_ratio`` times
    ``sqrt(norm(eta_flux)**2 + norm(eta_nonconf)**2)``. When the first exceeds
    the second, double the local refinement everywhere; otherwise apply
    Dörfler marking and conforming red-green macro refinement. These are
    refinement proxies, not an exact separation of local and global errors.
    No estimator-contraction or complexity-optimality theorem is asserted.

    Uniform local refinement keeps shared fine boundary partitions compatible.
    A custom ``local_mesh_factory(mesh, cell, refinement)`` must likewise give
    a globally conforming partition, checked by potential recovery. It can fit
    Cartesian material interfaces while leaving the macro mesh unchanged.
    A local-resolution limit terminates the loop explicitly rather than
    substituting macro refinement for unresolved local indicators.
    ``local_error_indicator(solution)`` may supply a nonnegative alternative,
    such as an energy difference between nested local solves with unchanged
    trace data. Such a difference requires a saturation assumption to bound
    the actual local error; the policy does not assume that bound.
    ``on_state`` receives the complete accumulated result after each validated
    solve and refinement decision. It may save checkpoints; its exceptions are
    propagated. No incomplete solution is passed to the callback.
    ``macro_refiner`` selects the conforming macro refinement operation and its
    exact ancestry. The default is red-green; ``refine_longest_edge`` instead
    applies longest-edge propagation without green descendants.
    ``estimator_convention`` selects the literal published numerical terms or
    physical-energy material weights, without changing the finite-element PDE.
    """
    positive_int(iterations, "iterations")
    positive_int(local_refinement, "local_refinement")
    positive_int(maximum_local_refinement, "maximum_local_refinement")
    if maximum_local_refinement < local_refinement:
        raise ValueError("maximum_local_refinement must be at least local_refinement")
    if (
        not isinstance(local_error_ratio, Real)
        or not np.isfinite(local_error_ratio)
        or local_error_ratio <= 0
    ):
        raise ValueError("local_error_ratio must be finite and positive")
    parameters = dict(problem)
    space = FaceSpace.uniform(trace_degree, trace_segments)
    skeleton = (
        SkeletonSpace(mesh, tuple(space for _ in mesh.faces)) if skeleton is None else skeleton
    )
    solutions, estimates, marks, refinements, levels, decisions = [], [], [], [], [], []
    local_indicators = []
    bound = ellipticity_lower_bound
    reason = "iterations"
    while True:
        step = len(solutions)
        factory = (
            None
            if local_mesh_factory is None
            else lambda current, cell, level=local_refinement: local_mesh_factory(
                current, cell, level
            )
        )
        state = solve_adaptive_darcy(
            mesh,
            iterations=1,
            theta=theta,
            tolerance=tolerance,
            maximum_cells=maximum_cells,
            trace_degree=trace_degree,
            trace_segments=trace_segments,
            reconstruction_degree=reconstruction_degree,
            estimator_order=estimator_order,
            estimator_convention=estimator_convention,
            ellipticity_lower_bound=bound,
            skeleton=skeleton,
            local_mesh_factory=factory,
            local_refinement=local_refinement,
            **parameters,
        )
        solution, estimate, marked = state.solutions[0], state.estimators[0], state.marked[0]
        solutions.append(solution)
        estimates.append(estimate)
        marks.append(marked)
        levels.append(local_refinement)
        local = (
            float(np.linalg.norm(estimate.divergence_defect + estimate.oscillation))
            if local_error_indicator is None
            else local_error_indicator(solution)
        )
        if not isinstance(local, Real) or not np.isfinite(local) or local < 0:
            raise ValueError("local_error_indicator must return a finite nonnegative real scalar")
        local_indicators.append(float(local))
        other = np.hypot(
            np.linalg.norm(estimate.flux_defect), np.linalg.norm(estimate.nonconformity)
        )
        decision = "stop"
        if estimate.total <= tolerance or not np.any(marked):
            reason = "tolerance"
        elif step + 1 == iterations:
            reason = "iterations"
        elif local > local_error_ratio * other:
            if 2 * local_refinement > maximum_local_refinement:
                reason = "local_refinement_limit"
            else:
                local_refinement *= 2
                decision = "local"
        else:
            refined = macro_refiner(mesh, marked)
            if len(refined.mesh.cells) > maximum_cells:
                reason = "maximum_cells"
            else:
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
                decision = "macro"
        decisions.append(decision)
        result = BalancedDarcyResult(
            AdaptiveDarcyResult(
                tuple(solutions), tuple(estimates), tuple(marks), tuple(refinements)
            ),
            tuple(levels),
            tuple(decisions),
            reason,
            tuple(local_indicators),
        )
        if on_state is not None:
            on_state(result)
        if decision == "stop":
            break
    return result
