"""Fixed-macro-mesh face and local refinement for the two-level flow estimators."""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.flow.solver import solve_flow
from pymhm.adaptivity.flow_local_mesh import refine_flow_local_meshes
from pymhm.adaptivity.transport import refine_skeleton_faces
from pymhm.core.validation import positive_int
from pymhm.estimators.flow import FlowEstimator, estimate_flow_error
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import VectorSolution


@dataclass(frozen=True)
class AdaptiveFlowResult:
    """Solved fields and indicators at each adaptive state, with an explicit stop reason."""

    solutions: tuple[VectorSolution, ...]
    estimators: tuple[FlowEstimator, ...]
    refinements: tuple[tuple[int, ...], ...]
    stop_reason: str
    local_refiner: str = "uniform"
    local_error_marking: str = "uniform"


def mark_flow_faces(
    estimator: FlowEstimator,
    theta: float = 0.5,
    *,
    variant: Literal["oseen-2021", "stokes-brinkman-2021"] = "oseen-2021",
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Apply face marking with local-error dominance.

    Use Algorithm 1 of [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8) or
    Algorithm 2 of [Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053).

    Both algorithms use eta_F=eta_1,F+sum eta_2,K, with eta_1,F=sqrt(sum_segments eta_1,Ftilde²).
    Mark faces above theta times the maximum and, within each marked face, every segment tied for
    its largest indicator. Refine incident local meshes when eta_1,F < sum eta_2,K. Values of
    eta_2,K are unscaled, exactly as in those algorithms.
    """
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must lie strictly between zero and one")
    if variant not in ("oseen-2021", "stokes-brinkman-2021"):
        raise ValueError("unknown flow marking variant")
    mesh = estimator.solution.skeleton.mesh
    first = np.sqrt([value.sum() for value in estimator.face_squared])
    local = np.sqrt(estimator.local_squared)
    combined = np.array(
        [
            first[face] + np.sum(local[cells[cells >= 0]])
            for face, cells in enumerate(mesh.face_cells)
        ]
    )
    chosen = (combined >= theta * combined.max()) & (combined > 0)
    marked, refine = [], np.zeros(len(mesh.cells), dtype=bool)
    for face, values in enumerate(estimator.face_squared):
        marked.append(chosen[face] & (values >= values.max()))
        neighbors = mesh.face_cells[face]
        neighbors = neighbors[neighbors >= 0]
        if chosen[face] and first[face] < local[neighbors].sum():
            refine[neighbors] = True
    return tuple(marked), refine


def _aligned_refinement(skeleton: SkeletonSpace, macro: int, current: int) -> int:
    """Find the uniform local grid containing every rational skeletal breakpoint."""
    refinement = current
    for face in skeleton.mesh.cell_faces[macro]:
        for point in skeleton.faces[face].breaks:
            fraction = Fraction(point).limit_denominator(2**20)
            if abs(float(fraction) - point) > 32 * np.finfo(float).eps:
                raise ValueError("adaptive local alignment requires rational face breakpoints")
            refinement = lcm(refinement, fraction.denominator)
    return refinement


def adapt_flow(
    mesh: TriangleMesh,
    *,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int | tuple[int, ...] = 1,
    iterations: int = 4,
    theta: float = 0.5,
    tolerance: float = 0.0,
    max_local_refinement: int = 64,
    estimator_variant: Literal["oseen-2021", "stokes-brinkman-2021"] = "oseen-2021",
    estimator_order: int = 8,
    local_refiner: Literal["uniform", "longest-edge"] = "uniform",
    max_local_cells: int = 65536,
    local_error_marking: Literal["uniform", "maximum"] = "uniform",
    solve_step: Callable[..., VectorSolution] | None = None,
    **solve_options: Any,
) -> AdaptiveFlowResult:
    """Solve, estimate and refine traces/local meshes with a fixed macro topology.

    Defaults are local P3/P3 Oseen with discontinuous P1 traces. Supply physical coefficients and
    forcing through solve_flow keyword arguments. Full Dirichlet data are required by these
    published estimators. The initial local meshes must resolve the supplied skeleton. Each
    subsequent local grid doubles on local-error dominance and is additionally refined to contain
    every new skeletal breakpoint, as in the matching submesh construction in Section 5.3 of
    [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8). This rational uniform-grid
    closure is explicit, not a minimal unstructured refinement. The maximum local resolution bounds
    memory consumption and returns a distinct stop reason; it never changes tolerances or solver
    choice. ``local_refiner='longest-edge'`` instead closes new boundary breakpoints with local
    longest-edge propagation; local-error dominance red-refines that local mesh once.
    ``max_local_cells`` limits this nonuniform route. Its ``refinements`` result is empty because a
    uniform subdivision count does not describe these meshes; actual meshes are stored in each
    solution. With longest-edge closure, ``local_error_marking='maximum'`` selects cells whose eta_2
    contribution is at least theta times the local maximum, within every local mesh triggered by the
    published macroface marking. Interior fine-face terms are split equally between their two
    neighboring cells. This within-local policy is explicit; it is not prescribed by
    [Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053).
    ``solve_step(mesh, skeleton=current_space, **problem_data)`` may assemble
    the user's own mathematical equations at each state. It receives the actual
    transferred boundary data and local partitions; its returned physical field
    must satisfy the estimator's existing hypotheses. The default retains the
    built-in solve. Callback exceptions propagate without changing refinement.
    """
    if solve_step is not None and not callable(solve_step):
        raise TypeError("solve_step must be callable or None")
    solve = solve_flow if solve_step is None else solve_step
    iterations = positive_int(iterations, "iterations", 0)
    limit = positive_int(max_local_refinement, "max_local_refinement")
    cell_limit = positive_int(max_local_cells, "max_local_cells")
    if local_refiner not in ("uniform", "longest-edge"):
        raise ValueError("local_refiner must be uniform or longest-edge")
    if local_error_marking not in ("uniform", "maximum"):
        raise ValueError("local_error_marking must be uniform or maximum")
    if local_error_marking == "maximum" and local_refiner != "longest-edge":
        raise ValueError("maximum local_error_marking requires longest-edge local_refiner")
    if "local_meshes" in solve_options:
        raise ValueError("adaptive flow constructs its own local_meshes")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must lie strictly between zero and one")
    if solve_options.get("traction") or solve_options.get("traction_components"):
        raise ValueError("adaptive flow requires full Dirichlet boundaries")
    skeleton = skeleton or SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    raw = np.broadcast_to(local_refinement, (len(mesh.cells),))
    refinement = tuple(positive_int(value, "local_refinement") for value in raw)
    if max(refinement) > limit:
        raise ValueError("initial refinement exceeds max_local_refinement")
    if any(
        _aligned_refinement(skeleton, cell, value) != value for cell, value in enumerate(refinement)
    ):
        raise ValueError("initial local refinement must contain every skeletal breakpoint")
    options: dict[str, Any] = {"formulation": "oseen", "degree": 3}
    options.update(solve_options)
    local_meshes = (
        tuple(mesh.submesh(cell, r) for cell, r in enumerate(refinement))
        if local_refiner == "longest-edge"
        else None
    )
    if local_meshes is not None and max(len(m.cells) for m in local_meshes) > cell_limit:
        raise ValueError("initial local mesh exceeds max_local_cells")
    physics = {
        key: options[key]
        for key in ("viscosity", "drag", "advection", "source", "dirichlet")
        if key in options
    }
    solutions, estimators, refinements = [], [], []
    reason = "iterations"
    iteration = 0
    while True:
        solution = solve(
            mesh,
            skeleton=skeleton,
            local_refinement=refinement,
            local_meshes=local_meshes,
            **options,
        )
        estimator = estimate_flow_error(
            solution,
            full_dirichlet=True,
            variant=estimator_variant,
            quadrature_order=estimator_order,
            **physics,
        )
        solutions.append(solution)
        estimators.append(estimator)
        if local_meshes is None:
            refinements.append(refinement)
        if estimator.total <= tolerance:
            reason = "tolerance"
            break
        if iteration == iterations:
            break
        marked, refine = mark_flow_faces(estimator, theta, variant=estimator_variant)
        updated = refine_skeleton_faces(skeleton, marked)
        if local_meshes is not None:
            fine_marked = (
                tuple(values >= theta**2 * values.max() for values in estimator.fine_squared)
                if local_error_marking == "maximum"
                else None
            )
            closed = refine_flow_local_meshes(
                updated, local_meshes, refine, cell_limit, fine_marked=fine_marked
            )
            if closed is None:
                reason = "resolution_limit"
                break
            local_meshes, skeleton = closed, updated
            iteration += 1
            continue
        next_refinement = tuple(
            _aligned_refinement(updated, cell, value * (2 if refine[cell] else 1))
            for cell, value in enumerate(refinement)
        )
        if max(next_refinement) > limit:
            reason = "resolution_limit"
            break
        skeleton, refinement = updated, next_refinement
        iteration += 1
    return AdaptiveFlowResult(
        tuple(solutions),
        tuple(estimators),
        tuple(refinements),
        reason,
        local_refiner,
        local_error_marking,
    )
