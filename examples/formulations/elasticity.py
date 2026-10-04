"""User-written P1/P1 Herrmann equations with physical rigid-motion moments.

This application is the constant-shear GaLS configuration of the introductory
nearly incompressible study. Basix tabulation, trace integration and element
scatter belong to shared FEM kernels; no physical solver or local factory is
called. Coefficients use displacement components interleaved at each node,
followed by pressure nodes, with exactly the declared global-centered rigid modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm import LocalEquations
from pymhm._legacy.models.elasticity.mixed_pressure import (
    ElasticitySolution,
    _strain_and_divergence,
)
from pymhm._legacy.models.elasticity.mixed_pressure import (
    _rigid as rigid_modes,
)
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class DisplacementPressureSpace:
    """Declare continuous P1/P1 locals and a two-component macroface space."""

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    refinement: int = 4
    quadrature_order: int = 8


def displacement_pressure_equations(
    cell: int,
    *,
    space: DisplacementPressureSpace,
    source: Any,
    lame_lambda: float,
    lame_mu: float = 1.0,
) -> LocalEquations:
    """Declare 2*mu*epsilon, div/pressure, compliance and negative GaLS residuals.

    For P1 displacement and constant shear, div(2*mu*epsilon(u)) vanishes on
    each fine triangle, so R(u,p)=-grad(p). The inverse-inequality constant is
    one and alpha=1/(4*mu), strictly below the sufficient bound 1/(2*mu).
    The source pairing adds alpha*h²*(R(v,q),f), with this residual sign.
    Infinite lambda means zero compliance. B uses the signed skeleton map and
    C=-B.T; its multiplier is negative outward Cauchy traction. The physical
    displacement and pressure gauge rows are left to the global application.
    """
    if lame_lambda <= 0 or np.isnan(lame_lambda):
        raise ValueError("this application needs positive lambda or its infinite limit")
    if lame_mu <= 0 or not np.isfinite(lame_mu):
        raise ValueError("this application needs constant positive finite shear")
    if space.skeleton.mesh is not space.mesh or space.skeleton.components != 2:
        raise ValueError("displacement traces need two components on this macro mesh")
    # These examples declare P1 traces and aligned r>=4, the matching-mesh bound.
    for face in space.skeleton.faces:
        intervals = np.diff(np.asarray(face.breaks)) * space.refinement
        if any(degree != 1 for degree in face.degrees) or np.any(intervals < 4):
            raise ValueError("this P1/P1 application requires four fine edges per P1 trace segment")
        if not np.allclose(intervals, np.round(intervals), atol=1e-12, rtol=0):
            raise ValueError("trace segments must align with fine edges")
    fine = space.mesh.submesh(cell, space.refinement)
    bary, weights = triangle_quadrature(max(space.quadrature_order, 3))
    dofs, nodes, basis, grad, hessian = tabulate(fine, 1, bary)
    nc, nq, ns, _ = grad.shape
    nv = len(nodes)
    udofs = (2 * dofs[:, :, None] + np.arange(2)).reshape(nc, 2 * ns)
    all_dofs = np.column_stack((udofs, 2 * nv + dofs))
    strain, divergence, strong = _strain_and_divergence(grad, hessian)
    mu = np.full((nc, nq), lame_mu)
    compliance = np.full((nc, nq), 0.0 if np.isinf(lame_lambda) else 1 / lame_lambda)
    blocks = np.zeros((nc, 3 * ns, 3 * ns))
    blocks[:, : 2 * ns, : 2 * ns] = np.einsum(
        "q,tqai,a,tqaj,tq,t->tij", weights, strain, [2.0, 2.0, 1.0], strain, mu, fine.areas
    )
    mixed = -np.einsum("q,tqi,qj,t->tij", weights, divergence, basis, fine.areas)
    blocks[:, : 2 * ns, 2 * ns :] = mixed
    blocks[:, 2 * ns :, : 2 * ns] = mixed.swapaxes(1, 2)
    blocks[:, 2 * ns :, 2 * ns :] = -np.einsum(
        "q,qi,qj,tq,t->tij", weights, basis, basis, compliance, fine.areas
    )
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    force = vector_values(source, physical.reshape(-1, 2)).reshape(nc, nq, 2)
    load_blocks = np.zeros((nc, 3 * ns))
    load_blocks[:, : 2 * ns] = np.einsum(
        "q,qi,tqa,t->tia", weights, basis, force, fine.areas
    ).reshape(nc, -1)
    alpha = 1 / (4 * lame_mu)
    h = np.max(fine.lengths[fine.cell_faces], axis=1)
    residual = np.concatenate((2 * mu[:, :, None, None] * strong, -grad.swapaxes(-1, -2)), axis=3)
    blocks -= np.einsum(
        "q,tqai,tqaj,t->tij", weights, residual, residual, alpha * h**2 * fine.areas
    )
    load_blocks += np.einsum(
        "q,tqai,tqa,t->ti", weights, residual, force, alpha * h**2 * fine.areas
    )
    size = 3 * nv
    a = assemble_element_blocks(blocks, all_dofs, all_dofs, (size, size))
    load = np.bincount(all_dofs.ravel(), weights=load_blocks.ravel(), minlength=size)
    b = np.zeros((size, len(space.skeleton.cell_dofs(cell))))
    b[: 2 * nv] = np.kron(trace_coupling(space.mesh, cell, fine, space.skeleton, 1), np.eye(2))
    center = space.mesh.points.mean(axis=0)
    kernel = np.zeros((size, 3))
    kernel[: 2 * nv] = rigid_modes(nodes, center).reshape(2 * nv, 3)
    local_moments = np.einsum(
        "q,qi,tqaj,t->tiaj",
        weights,
        basis,
        rigid_modes(physical.reshape(-1, 2), center).reshape(*physical.shape, 3),
        fine.areas,
    ).reshape(nc, 2 * ns, 3)
    moments = np.zeros_like(kernel)
    np.add.at(moments, udofs, local_moments)
    pressure_weights = np.zeros(size)
    np.add.at(pressure_weights, 2 * nv + dofs, fine.areas[:, None] * (weights @ basis))
    compliance_weights = np.zeros(size)
    np.add.at(
        compliance_weights,
        2 * nv + dofs,
        np.einsum("q,qi,tq,t->ti", weights, basis, compliance, fine.areas),
    )
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        space.skeleton.cell_dofs(cell),
        d=0,
        g=0,
        kernel=kernel,
        moments=moments,
        metadata=(
            fine,
            nv,
            pressure_weights,
            alpha,
            moments,
            compliance_weights,
            float(compliance.max()),
        ),
    )


def displacement_pressure_fields(
    system: MultiscaleSystem,
    solution: HybridSolution,
    space: DisplacementPressureSpace,
    lame_lambda: float,
    lame_mu: float = 1.0,
) -> ElasticitySolution:
    """Interpret executed P1 coefficients as displacement, Herrmann pressure and stress."""
    return ElasticitySolution(
        skeleton=space.skeleton,
        local_meshes=tuple(record[0] for record in system.local_metadata),
        values=tuple(
            field[: 2 * record[1]].reshape(-1, 2)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        pressure=tuple(
            field[2 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        hybrid=solution,
        degree=1,
        pressure_degree=1,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        formulation="gals",
        stabilization=tuple(record[3] for record in system.local_metadata),
    )
