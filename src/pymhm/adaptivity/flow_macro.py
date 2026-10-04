"""Macroelement marking from the Stokes--Brinkman estimator in L14 Algorithm 1."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.flow.solver import solve_flow
from pymhm._legacy.models.vector import VectorSolution
from pymhm.core.validation import positive_int
from pymhm.estimators.flow import FlowEstimator, estimate_flow_error
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.longest_edge import refine_longest_edge
from pymhm.meshes.refinement import TriangleRefinement, refine_triangles
from pymhm.meshes.triangle import TriangleMesh


def mark_flow_cells(estimator: FlowEstimator, theta: float = 0.5) -> np.ndarray:
    """Mark eta_K >= theta*max(eta_K), where eta_K=eta_2,K+sum_F eta_1,F.

    The first-level contribution on each macroface is the square root of its
    summed segment contributions. Interior macrofaces contribute to both cells.
    Local indicators are unscaled; this is L14 Algorithm 1, not bulk marking.
    A zero estimator gives an empty marking set.
    """
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must lie strictly between zero and one")
    first = np.sqrt([value.sum() for value in estimator.face_squared])
    faces = estimator.solution.skeleton.mesh.cell_faces
    indicators = np.sqrt(estimator.local_squared) + first[faces].sum(axis=1)
    return (indicators >= theta * indicators.max()) & (indicators > 0)


@dataclass(frozen=True)
class AdaptiveFlowMacroResult:
    """Solved macro meshes, original indicators and conforming refinement ancestry."""

    solutions: tuple[VectorSolution, ...]
    estimators: tuple[FlowEstimator, ...]
    marked: tuple[np.ndarray, ...]
    refinements: tuple[TriangleRefinement, ...]
    stop_reason: str


def adapt_flow_macros(
    mesh: TriangleMesh,
    *,
    trace_degree: int = 0,
    local_refinement: int | tuple[int, ...] = 1,
    iterations: int = 4,
    theta: float = 0.5,
    tolerance: float = 0.0,
    maximum_cells: int = 10000,
    estimator_order: int = 8,
    macro_refiner: Literal["red-green", "longest-edge"] = "red-green",
    **solve_options: Any,
) -> AdaptiveFlowMacroResult:
    """Apply L14 Algorithm 1 with an explicitly selected conforming macro refinement.

    ``iterations`` counts refinement steps after the initial solve. Marked
    triangles use red--green closure by default. ``macro_refiner='longest-edge'``
    selects longest-edge propagation, preserving the same marking rule and
    controlling the minimum angle under repeated local refinement. Each
    child inherits its parent's local subdivision count, so no additional
    local refinement is induced. Local physical cells shrink only with their
    parent macrocell. Traces are unsplit discontinuous P_ell on every current
    macroface. This fully specified closure is an original realization of the
    published strategy; the paper does not provide its historical connectivity.

    The estimator assumes full Dirichlet Stokes/Brinkman data. Defaults are
    local P3/P3 USFEM and P0 traces. ``solve_options`` carries the same physical
    data to both solver and estimator; advection and prescribed tractions are
    excluded. A cell cap stops before solving an oversized mesh. No convergence
    rate or unit reliability constant is asserted by this adaptive loop.
    """
    iterations = positive_int(iterations, "iterations", 0)
    maximum_cells = positive_int(maximum_cells, "maximum_cells")
    if macro_refiner not in ("red-green", "longest-edge"):
        raise ValueError("macro_refiner must be red-green or longest-edge")
    refine = refine_triangles if macro_refiner == "red-green" else refine_longest_edge
    space = FaceSpace.uniform(trace_degree)
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must lie strictly between zero and one")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if len(mesh.cells) > maximum_cells:
        raise ValueError("initial macro mesh exceeds maximum_cells")
    if solve_options.get("traction") or solve_options.get("traction_components"):
        raise ValueError("L14 macro adaptivity requires full Dirichlet boundaries")
    advection = solve_options.get("advection", (0.0, 0.0))
    if callable(advection) or np.any(advection):
        raise ValueError("L14 macro adaptivity excludes advection")
    if any(key in solve_options for key in ("skeleton", "local_meshes")):
        raise ValueError("macro adaptivity builds its own skeleton and local meshes")
    raw = np.broadcast_to(local_refinement, (len(mesh.cells),))
    counts = tuple(positive_int(value, "local_refinement") for value in raw)
    options: dict[str, Any] = {"formulation": "usfem", "degree": 3}
    options.update(solve_options)
    physics = {
        key: options[key] for key in ("viscosity", "drag", "source", "dirichlet") if key in options
    }
    solutions, estimators, markings, refinements = [], [], [], []
    reason = "iterations"
    step = 0
    while True:
        skeleton = SkeletonSpace(mesh, tuple(space for _ in mesh.faces), 2)
        solution = solve_flow(mesh, skeleton=skeleton, local_refinement=counts, **options)
        estimate = estimate_flow_error(
            solution,
            full_dirichlet=True,
            variant="stokes-brinkman-2021",
            quadrature_order=estimator_order,
            **physics,
        )
        marked = mark_flow_cells(estimate, theta)
        solutions.append(solution)
        estimators.append(estimate)
        markings.append(marked)
        if estimate.total <= tolerance:
            reason = "tolerance"
            break
        if step == iterations:
            break
        refined = refine(mesh, marked)
        if len(refined.mesh.cells) > maximum_cells:
            reason = "cell_limit"
            break
        counts = tuple(counts[parent] for parent in refined.parent_cells)
        mesh = refined.mesh
        refinements.append(refined)
        step += 1
    return AdaptiveFlowMacroResult(
        tuple(solutions), tuple(estimators), tuple(markings), tuple(refinements), reason
    )
