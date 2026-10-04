"""Declare conservative SUPG equations from shared FEM and residual-scale kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm import LocalEquations
from pymhm._legacy.models.transport.rad import _streamline_scale
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.triangle import element_tabulate, trace_coupling
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values

from .scalar import ScalarDiscretization


def streamline_equations(
    cell: int,
    *,
    data: ScalarDiscretization,
    velocity: Any,
    velocity_divergence: Any,
    diffusion_divergence: Any,
    reaction: Any,
    source: Any,
) -> LocalEquations:
    """Write conservative skew Galerkin and its complete streamline residual pairing.

    The strong trial residual is ``-div(K grad(u))+beta.grad(u)+(c+div(beta))*u``
    and the test residual is ``beta.grad(v)``. Analytical divergence data are
    explicit inputs; sampled coefficients do not certify continuum bounds.
    The positive tau comes from the existing pure residual-scale owner. The
    trace multiplier is ``(-K grad(u)+beta*u/2).n`` with the fixed normal map.
    Global weak Dirichlet moments and physical constraints are caller choices.
    """
    fine = data.local_mesh(cell)
    bary, weights, material = material_triangle_quadrature(
        fine, data.diffusion, max(data.order, data.degree + 2)
    )
    dofs, nodes, basis, gradient, hessian = element_tabulate(fine, data.degree, bary)
    points = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    flat = points.reshape(-1, 2)
    nc, nq = weights.shape
    tensor = tensor_values(material, flat).reshape(nc, nq, 2, 2)
    beta = vector_values(velocity, flat).reshape(nc, nq, 2)
    div_beta = scalar_values(velocity_divergence, flat).reshape(nc, nq)
    c = scalar_values(reaction, flat).reshape(nc, nq)
    effective = c + div_beta / 2
    if np.any(effective < 0):
        raise ValueError("the declared conservative energy needs c+div(beta)/2 nonnegative")
    force = scalar_values(source, flat).reshape(nc, nq)
    blocks = np.einsum("tq,tqia,tqab,tqjb,t->tij", weights, gradient, tensor, gradient, fine.areas)
    blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, basis, basis, effective, fine.areas)
    streamline = np.einsum("tqa,tqia->tqi", beta, gradient)
    advection = np.einsum("tq,tqi,tqj,t->tij", weights, basis, streamline, fine.areas)
    blocks += (advection - advection.swapaxes(1, 2)) / 2
    load_blocks = np.einsum("tq,tqi,tq,t->ti", weights, basis, force, fine.areas)
    div_tensor = vector_values(diffusion_divergence, flat).reshape(nc, nq, 2)
    strong_diffusion = np.einsum("tqab,tqiab->tqi", tensor, hessian) + np.einsum(
        "tqa,tqia->tqi", div_tensor, gradient
    )
    strong = -strong_diffusion + streamline + (c + div_beta)[:, :, None] * basis
    tau = _streamline_scale(fine, tensor, beta, c + div_beta)
    blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, streamline, strong, tau, fine.areas)
    load_blocks += np.einsum("tq,tqi,tq,tq,t->ti", weights, streamline, force, tau, fine.areas)
    a = assemble_element_blocks(blocks, dofs, dofs, (len(nodes), len(nodes)))
    load = np.bincount(dofs.ravel(), weights=load_blocks.ravel(), minlength=len(nodes))
    moments = np.zeros(len(nodes))
    np.add.at(moments, dofs, fine.areas[:, None] * np.einsum("tq,tqi->ti", weights, basis))
    constant = np.ones((len(nodes), 1))
    b = trace_coupling(data.mesh, cell, fine, data.skeleton, data.degree)
    pure = not np.any(effective) and not np.any(beta)
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        d=0,
        g=0,
        kernel=constant if pure else None,
        coarse_basis=None if pure else constant,
        moments=moments[:, None],
        metadata=fine,
    )
