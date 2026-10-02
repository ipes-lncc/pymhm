"""Face-jump refinement indicators for the one-level Darcy MHM analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.darcy import DarcySolution
from pymhm.elements import scalar_values
from pymhm.mesh import FloatArray, positive_int
from pymhm.scalar_boundary import prepare_scalar_trace
from pymhm.weighted_estimator import _ellipticity


@dataclass(frozen=True)
class DarcyJumpEstimator:
    """L02 face indicators with the original cell-side multiplicity.

    Interior faces contribute once to each of their two incident cells. The
    published reliability theorem concerns exact local lifts; this quantity
    alone does not bound the error of finite-dimensional local solves.
    """

    face_squared: FloatArray
    local_squared: FloatArray
    coefficient_scale: float
    calibration: float

    @property
    def total(self) -> float:
        """Return the square root of the sum over all macrocell sides."""
        return float(np.sqrt(np.sum(self.local_squared)))


def estimate_darcy_jumps(
    solution: DarcySolution,
    *,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    ellipticity_lower_bound: Any = None,
    calibration: float = 1.0,
    order: int | None = None,
) -> DarcyJumpEstimator:
    """Evaluate Araya et al. (2013), equations (5.1)--(5.3).

    Use R_F=-jump(p)/2 on interior macrofaces, R_F=g_D-p on Dirichlet
    faces, and zero on Neumann faces. Then eta_F=c_l*c_min*||R_F||/sqrt(H_F),
    where H_F is the original macroface length and c_min is the square root
    of a certified lower eigenvalue bound for permeability. Integration splits
    at both fine-edge partitions and all skeletal breakpoints.

    ``calibration`` is c_l, not a proven constant-one reliability factor.
    The paper's Section 6 uses 3, 7, 18 and 50 for uniform trace degrees 0--3.
    Its theorem assumes exact local lifts, regularity and represented boundary
    data. Finite local errors require an additional estimator, such as the
    weighted flux-reconstruction estimator; no bound is asserted here.
    """
    if solution.formulation != "primal":
        raise ValueError("the jump indicator requires a primal pressure field")
    if not np.isreal(calibration) or not np.isfinite(calibration) or calibration <= 0:
        raise ValueError("calibration must be a finite positive real number")
    skeleton = solution.skeleton
    mesh = skeleton.mesh
    natural = {} if neumann is None else neumann
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    alpha = float(np.min(_ellipticity(solution, ellipticity_lower_bound)))
    quadrature = solution.degree + 1 if order is None else positive_int(order, "order")
    gauss, weights = leggauss(quadrature)
    indicators = np.zeros(len(mesh.faces))
    for face, space in enumerate(skeleton.faces):
        first, second = mesh.face_cells[face]
        if face in natural:
            continue
        incident = [first] if second < 0 else [first, second]
        traces = {
            cell: prepare_scalar_trace(mesh, solution.local_meshes[cell], face, solution.degree)
            for cell in incident
        }
        cuts = sorted(
            {
                *space.breaks,
                *(
                    position
                    for cell in incident
                    for positions, _ in traces[cell].pieces
                    for position in positions
                ),
            }
        )
        start, end = mesh.points[mesh.faces[face]]
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            parameter = left + (gauss + 1) * (right - left) / 2
            residual = traces[first].evaluate(solution.pressure[first], parameter)
            if second >= 0:
                residual -= traces[second].evaluate(solution.pressure[second], parameter)
                residual *= -0.5
            else:
                residual = (
                    scalar_values(dirichlet, start + parameter[:, None] * (end - start)) - residual
                )
            indicators[face] += (right - left) * float(weights @ residual**2) / 2
    indicators *= float(calibration) ** 2 * alpha
    return DarcyJumpEstimator(
        indicators,
        np.sum(indicators[mesh.cell_faces], axis=1),
        float(np.sqrt(alpha)),
        float(calibration),
    )
