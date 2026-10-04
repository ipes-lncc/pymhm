"""Face-residual marking and fixed-macro-mesh RAD adaptivity from L11 Section 4."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from pymhm._legacy.models.transport.solver import ScalarSolution, solve_transport
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.scalar import prepare_scalar_trace


@dataclass(frozen=True)
class TransportBounds:
    """Declared global bounds for the L11 transport face indicator.

    ``diffusion`` bounds the largest eigenvalue of K, ``velocity`` bounds the
    Euclidean magnitude of beta, and ``effective_reaction`` bounds
    abs(c+div(beta)/2). Callers supply analytical bounds, not sampled maxima.
    """

    diffusion: float
    velocity: float
    effective_reaction: float

    def __post_init__(self) -> None:
        """Require finite positive diffusion and nonnegative remaining bounds."""
        values = np.array([self.diffusion, self.velocity, self.effective_reaction])
        if not np.isfinite(values).all() or self.diffusion <= 0 or np.any(values[1:] < 0):
            raise ValueError("coefficient bounds must be finite, diffusion positive, others >=0")

    def scale(self, skeleton: SkeletonSpace) -> float:
        """Compute max(sqrt(Kmax), diameter*beta_max, diameter²*c_eff_max)."""
        points = skeleton.mesh.points
        diameter = float(np.max(pdist(points[ConvexHull(points).vertices])))
        return max(
            float(np.sqrt(self.diffusion)),
            diameter * self.velocity,
            diameter**2 * self.effective_reaction,
        )


@dataclass(frozen=True)
class FaceIndicators:
    """One nonnegative L11 indicator per skeletal segment, including exterior zeros."""

    skeleton: SkeletonSpace
    values: tuple[FloatArray, ...]
    coefficient_scale: float

    @property
    def total(self) -> float:
        """Return the Euclidean norm of the segment indicators."""
        return float(np.linalg.norm(np.concatenate(self.values)))

    def mark(self, theta: float = 0.75) -> tuple[np.ndarray, ...]:
        """Mark eta >= theta*max(eta), stopping for a zero indicator field."""
        if not np.isfinite(theta) or not 0 < theta < 1:
            raise ValueError("theta must lie strictly between zero and one")
        maximum = max(float(np.max(value)) for value in self.values)
        return tuple((value >= theta * maximum) & (value > 0) for value in self.values)


def estimate_transport_faces(
    solution: ScalarSolution, bounds: TransportBounds, *, order: int | None = None
) -> FaceIndicators:
    """Integrate the jump estimator in L11 equations (4.3)--(4.4).

    On an interior macroface F, R_F=-jump(u)/2 and each segment Fhat receives
    ``C*norm(R_F,Fhat)/sqrt(length(F))``. The denominator is the original
    macroface length, not the refined segment length. Dirichlet conditions must
    be imposed strongly on the essential boundary. Prescribed natural faces are
    also supported: their multiplier variations vanish, so neither essential
    nor natural exterior faces contribute to this interior-jump indicator.
    ``solve_transport`` preserves the distinction between its Robin multiplier
    and a prescribed physical ``diffusive_flux``; the latter has its own
    half-advection boundary mass. In L11 section 5.1 the velocity is tangent to
    the natural walls, and the two flux conventions coincide there.

    This is a refinement indicator, not a computable constant-one upper bound
    or a maximum-principle certificate. Local-discretization errors and the
    approximation of nonrepresentable boundary data are not measured. A
    solution with weak Dirichlet imposition is rejected.

    Quadrature splits at both one-sided fine-edge partitions. The returned
    values therefore preserve broken macro traces and exactly integrate their
    polynomial squared jump for the default order.
    """
    if not solution.strong_dirichlet:
        raise ValueError("L11 face indicators require strong Dirichlet imposition")
    skeleton = solution.skeleton
    mesh = skeleton.mesh
    degree = solution.degree
    order = degree + 1 if order is None else positive_int(order, "order")
    gauss, weights = leggauss(order)
    scale = bounds.scale(skeleton)
    indicators = []
    for face, space in enumerate(skeleton.faces):
        first, second = mesh.face_cells[face]
        values = np.zeros(len(space.degrees))
        if second >= 0:
            traces = [
                prepare_scalar_trace(mesh, solution.local_meshes[cell], face, degree)
                for cell in (first, second)
            ]
            knots = [p for trace in traces for positions, _ in trace.pieces for p in positions]
            for segment, (left, right) in enumerate(
                zip(space.breaks[:-1], space.breaks[1:], strict=True)
            ):
                cuts = sorted({left, right, *(p for p in knots if left < p < right)})
                integral = 0.0
                for low, high in zip(cuts[:-1], cuts[1:], strict=True):
                    parameter = low + (gauss + 1) * (high - low) / 2
                    jump = traces[0].evaluate(solution.values[first], parameter)
                    jump -= traces[1].evaluate(solution.values[second], parameter)
                    # Physical face length cancels its original-macroface denominator.
                    integral += float(weights @ jump**2) * (high - low) / 8
                values[segment] = scale * np.sqrt(integral)
        indicators.append(values)
    return FaceIndicators(skeleton, tuple(indicators), scale)


def refine_skeleton_faces(skeleton: SkeletonSpace, marked: tuple[np.ndarray, ...]) -> SkeletonSpace:
    """Bisect marked segments, preserving degrees, continuity and vector components."""
    from pymhm.fem.traces.interval import FaceSpace

    if len(marked) != len(skeleton.faces):
        raise ValueError("provide one marking array per macroface")
    faces = []
    for space, mask in zip(skeleton.faces, marked, strict=True):
        flags = np.asarray(mask)
        if flags.dtype != bool or flags.shape != (len(space.degrees),):
            raise ValueError("marking arrays must be boolean with one entry per segment")
        breaks = [0.0]
        degrees = []
        for left, right, degree, flag in zip(
            space.breaks[:-1], space.breaks[1:], space.degrees, flags, strict=True
        ):
            if flag:
                breaks.append((left + right) / 2)
                degrees.append(degree)
            breaks.append(right)
            degrees.append(degree)
        faces.append(FaceSpace(tuple(breaks), tuple(degrees), space.continuous))
    return SkeletonSpace(skeleton.mesh, tuple(faces), skeleton.components)


@dataclass(frozen=True)
class AdaptiveTransportResult:
    """Solved adaptive states and indicators on their actual skeletal partitions."""

    solutions: tuple[ScalarSolution, ...]
    indicators: tuple[FaceIndicators, ...]
    stop_reason: str


def solve_adaptive_transport(
    skeleton: SkeletonSpace,
    bounds: TransportBounds,
    *,
    iterations: int = 5,
    theta: float = 0.75,
    tolerance: float = 0.0,
    max_trace_dofs: int | None = None,
    **options: Any,
) -> AdaptiveTransportResult:
    """Solve, estimate, mark and bisect faces while keeping macro/local meshes fixed.

    ``iterations`` counts solves, including the initial space. ``options`` are
    arguments of solve_transport; strong Dirichlet imposition is mandatory on
    the essential faces. Mixed data supplied through ``neumann`` (the Robin
    multiplier) or ``diffusive_flux`` retain their physical meaning at every
    state. Exterior faces remain unmarked because their data are prescribed.
    The local resolution must remain adequate as the trace space grows; the
    shared rank diagnostics reject incompatible enrichments. Neither local
    refinement nor a solver tolerance is changed automatically.
    """
    positive_int(iterations, "iterations")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if max_trace_dofs is not None:
        positive_int(max_trace_dofs, "max_trace_dofs")
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must lie strictly between zero and one")
    if options.get("dirichlet_enforcement", "strong") != "strong":
        raise ValueError("adaptive transport requires strong Dirichlet imposition")
    options["dirichlet_enforcement"] = "strong"
    solutions, indicators = [], []
    reason = "iterations"
    for iteration in range(iterations):
        solution = solve_transport(skeleton.mesh, skeleton=skeleton, **options)
        indicator = estimate_transport_faces(solution, bounds)
        solutions.append(solution)
        indicators.append(indicator)
        if indicator.total <= tolerance:
            reason = "tolerance"
            break
        if iteration + 1 < iterations:
            enriched = refine_skeleton_faces(skeleton, indicator.mark(theta))
            if max_trace_dofs is not None and enriched.size > max_trace_dofs:
                reason = "max_trace_dofs"
                break
            skeleton = enriched
    return AdaptiveTransportResult(tuple(solutions), tuple(indicators), reason)
