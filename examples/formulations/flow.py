"""User-written velocity/pressure blocks in the historical continuous nodal spaces.

These application equations use Basix-backed scalar tabulation, generic element
scatter and trace integration. They do not call a physical solver or its local
factory. The finite-element coefficient order is velocity components interleaved
at each node, followed by scalar pressure nodes. The multiplier is negative
grad-grad pseudotraction, with the half-advection term for the skew form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import LocalEquations
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.inequalities import laplacian_inverse_bound
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.vector.flow import advection_contract, minimum_resistance, triangle_flow_operators
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import VectorSolution


@dataclass(frozen=True)
class VelocityPressureSpace:
    """Declare continuous scalar polynomial spaces and their vector trace layout."""

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    velocity_degree: int = 2
    pressure_degree: int = 1
    refinement: int = 4
    quadrature_order: int = 5


def p1_residual_weight(
    fine: TriangleMesh,
    viscosity: float,
    drag: float,
    *,
    tabulation: Any = None,
    resistance: Any = None,
) -> Any:
    """Return the declared P1 USFEM weight h²/(max(drag*h²,12*nu)+12*nu).

    P1 velocity has identically zero element Laplacian, so the inverse constant
    is exactly m=1/3. This application rule applies to constant scalar drag and
    constant positive viscosity; higher degrees and other material contracts
    require their own independently established inverse bound.
    """
    h = np.max(fine.lengths[fine.cell_faces], axis=1)
    return h**2 / (np.maximum(drag * h**2, 12 * viscosity) + 12 * viscosity)


def tensor_residual_weight(
    fine: TriangleMesh,
    viscosity: float,
    drag: Any,
    *,
    tabulation: Any,
    resistance: Any,
) -> Any:
    """Declare the tensor-USFEM weight using the shared physical inverse-bound owner.

    ``tabulation`` contains physical velocity gradients, Hessians and the Gaussian
    weights. The resistance's sampled maximum eigenvalue bounds each fine cell
    for this smooth manufactured application; no supremum or unresolved material
    interface is certified by sampling. The inverse constant itself comes from
    the pure kernel used by the established coefficient implementation.
    """
    gradient, hessian, weights = tabulation
    h = np.max(fine.lengths[fine.cell_faces], axis=1)
    m = laplacian_inverse_bound(gradient, hessian, weights, h)
    viscous = 4 * viscosity / m
    upper = np.linalg.eigvalsh(resistance)[..., -1].max(axis=1)
    return h**2 / (np.maximum(upper * h**2, viscous) + viscous)


def velocity_pressure_equations(
    cell: int,
    *,
    space: VelocityPressureSpace,
    source: Any,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0),
    residual_weight: Any = None,
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    formulation: str | None = None,
    stabilization: str = "tensor-2025",
    gamma_min: float | None = None,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
) -> LocalEquations:
    """Declare grad-grad flow, skew transport, resistance and residual operators.

    A residual callback explicitly supplies positive tau per fine cell. USFEM
    subtracts the full momentum residual product; Oseen declares opposite test
    convection and grad-div. Volume operators integrate the chosen forms, while
    this application declares B, C, translation kernel/coarse modes and moments.
    Pk/P(k-1) Taylor--Hood or Pk/Pk equal-order spaces remain explicit inputs.
    """
    if viscosity <= 0 or not np.isfinite(viscosity):
        raise ValueError("viscosity must be finite and positive")
    form = (
        ("usfem" if residual_weight is not None else "taylor-hood")
        if formulation is None
        else formulation
    )
    expected = space.velocity_degree - 1 if form == "taylor-hood" else space.velocity_degree
    if space.pressure_degree != expected:
        raise ValueError("pressure degree must match the declared flow formulation")
    beta, divergence, bound = advection_contract(
        advection, advection_divergence, advection_bound, stabilized=form == "oseen"
    )
    minimum = minimum_resistance(drag, gamma_min) if stabilization == "minimum-2017" else None
    fine = (
        space.mesh.submesh(cell, space.refinement) if local_meshes is None else local_meshes[cell]
    )
    forms = triangle_flow_operators(
        fine,
        degree=space.velocity_degree,
        formulation=form,
        viscosity=viscosity,
        drag=drag,
        beta=beta,
        source=source,
        order=max(space.quadrature_order, space.velocity_degree + 2),
        gamma_min=minimum,
        pointwise=stabilization == "pointwise-2017",
        residual_weight=residual_weight,
        beta_divergence=divergence,
        beta_bound=bound,
    )
    nv, size = len(forms.velocity_nodes), len(forms.load)
    b = np.zeros((size, len(space.skeleton.cell_dofs(cell))))
    b[: 2 * nv] = np.kron(
        trace_coupling(space.mesh, cell, fine, space.skeleton, space.velocity_degree), np.eye(2)
    )
    selector = sparse.eye(size, format="csr")
    return LocalEquations(
        forms.matrix,
        forms.load,
        b,
        -b.T,
        space.skeleton.cell_dofs(cell),
        kernel=forms.kernel if forms.pure else None,
        coarse_basis=None if forms.pure else forms.kernel,
        moments=forms.translation_moments,
        metadata=(
            fine,
            nv,
            forms.pressure_moments,
            forms.translation_moments,
            forms.resistance_moment,
            forms.absolute_resistance_moment,
            forms.zero_columns,
        ),
        field_data=(
            nodal_field(
                "velocity",
                fine,
                space.velocity_degree,
                components=2,
                reconstruction=selector[: 2 * nv],
            ),
            nodal_field("pressure", fine, space.pressure_degree, reconstruction=selector[2 * nv :]),
        ),
    )


def velocity_pressure_fields(
    system: MultiscaleSystem, solution: HybridSolution, space: VelocityPressureSpace
) -> VectorSolution:
    """Interpret the executed nodal coefficients without changing bases or gauges."""
    return VectorSolution(
        space.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(
            field[: 2 * record[1]].reshape(-1, 2)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        tuple(
            field[2 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        solution,
        space.velocity_degree,
        space.pressure_degree,
    )
