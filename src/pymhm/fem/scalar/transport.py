"""Conservative scalar volume forms and explicit strong-residual stabilization."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.stabilization import streamline_scale
from pymhm.fem.scalar.triangle import element_tabulate
from pymhm.fem.scalar.unusual import UnusualParameters, unusual_scale
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class ScalarTransportOperators:
    """Integrated A/load and physical nodal moments, with exact sampled constant-mode flags."""

    matrix: Any
    load: Any
    moments: Any
    nodes: Any
    pure_diffusion: bool
    zero_reaction_divergence: bool


def triangle_transport_operators(
    fine: TriangleMesh,
    *,
    degree: int,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0),
    velocity_divergence: Any = 0.0,
    diffusion_divergence: Any = (0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    stabilization: str = "galerkin",
    order: int = 6,
    unusual_parameters: UnusualParameters | None = None,
) -> ScalarTransportOperators:
    """Integrate conservative skew transport and Galerkin, SUPG or UNUSUAL residuals.

    Trial strong residual is -div(K grad(u))+beta.grad(u)+(c+div(beta))*u.
    SUPG tests against tau beta.grad(v); UNUSUAL subtracts tau times the complete
    reaction-diffusion residual pair with zero advection. The weak skew operator
    has c+div(beta)/2 and requires that sampled energy coefficient nonnegative.
    Matrices contain only volume and an explicitly supplied numerical-normal
    advection callback. Boundary enforcement, trace maps and kernels are caller
    declarations. Material-cut quadrature does not resolve interfaces for strong
    residual methods; those interfaces must fit the local approximation mesh.
    """
    if stabilization not in ("galerkin", "supg", "unusual"):
        raise ValueError("stabilization must be galerkin, supg or unusual")
    if unusual_parameters is not None and (
        stabilization != "unusual" or not isinstance(unusual_parameters, UnusualParameters)
    ):
        raise ValueError("unusual_parameters requires UNUSUAL stabilization")
    if stabilization == "unusual":
        if (
            callable(velocity)
            or np.any(velocity)
            or callable(velocity_divergence)
            or np.any(velocity_divergence)
        ):
            raise ValueError("published scalar UNUSUAL requires zero advection")
        parameters = unusual_parameters or UnusualParameters()
        if (
            callable(diffusion)
            and not isinstance(diffusion, CartesianCellField)
            and parameters.diffusion_lower is None
        ):
            raise ValueError("UNUSUAL variable diffusion requires diffusion_lower")
        if callable(reaction) and parameters.reaction_upper is None:
            raise ValueError("UNUSUAL variable reaction requires reaction_upper")
    bary, weights, material = material_triangle_quadrature(fine, diffusion, max(order, degree + 2))
    dofs, nodes, basis, gradient, hessian = element_tabulate(fine, degree, bary)
    physical = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    flat = physical.reshape(-1, 2)
    nt, nq = weights.shape
    tensor = tensor_values(material, flat).reshape(nt, nq, 2, 2)
    beta = vector_values(velocity, flat).reshape(nt, nq, 2)
    normal_pair = hasattr(velocity, "advection_boundary_matrix")
    if normal_pair and stabilization != "galerkin":
        raise ValueError("raw volume/numerical-normal velocity requires Galerkin stabilization")
    div_beta = (
        np.zeros((nt, nq))
        if normal_pair
        else scalar_values(velocity_divergence, flat).reshape(nt, nq)
    )
    c = scalar_values(reaction, flat).reshape(nt, nq)
    effective = c + div_beta / 2
    if np.any(effective < 0):
        raise ValueError("reaction+div(velocity)/2 must be nonnegative")
    force = scalar_values(source, flat).reshape(nt, nq)
    blocks = np.einsum("tq,tqia,tqab,tqjb,t->tij", weights, gradient, tensor, gradient, fine.areas)
    blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, basis, basis, effective, fine.areas)
    streamline = np.einsum("tqa,tqia->tqi", beta, gradient)
    advective = np.einsum("tq,tqi,tqj,t->tij", weights, basis, streamline, fine.areas)
    blocks += (
        -advective.swapaxes(1, 2) if normal_pair else (advective - advective.swapaxes(1, 2)) / 2
    )
    element_load = np.einsum("tq,tqi,tq,t->ti", weights, basis, force, fine.areas)
    if stabilization in ("supg", "unusual"):
        div_tensor = vector_values(diffusion_divergence, flat).reshape(nt, nq, 2)
        diffusion_residual = np.einsum("tqab,tqiab->tqi", tensor, hessian) + np.einsum(
            "tqa,tqia->tqi", div_tensor, gradient
        )
        strong = -diffusion_residual + streamline + (c + div_beta)[:, :, None] * basis
        if stabilization == "supg":
            tau = streamline_scale(fine, tensor, beta, c + div_beta)
            test_residual = streamline
        else:
            if isinstance(diffusion, CartesianCellField) and np.any(tensor != tensor[:, :1]):
                raise ValueError(
                    "UNUSUAL requires material interfaces aligned with local fine cells"
                )
            tau = unusual_scale(
                basis,
                gradient,
                diffusion_residual,
                weights,
                fine.lengths[fine.cell_faces].max(axis=1),
                tensor,
                c,
                fine.points[fine.cells].mean(axis=1),
                unusual_parameters or UnusualParameters(),
            )[:, None]
            test_residual = -strong
        blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, test_residual, strong, tau, fine.areas)
        element_load += np.einsum(
            "tq,tqi,tq,tq,t->ti", weights, test_residual, force, tau, fine.areas
        )
    matrix = assemble_element_blocks(blocks, dofs, dofs, (len(nodes), len(nodes)))
    if normal_pair:
        matrix += velocity.advection_boundary_matrix(degree, order)
    load = np.bincount(dofs.ravel(), weights=element_load.ravel(), minlength=len(nodes))
    moments = np.zeros(len(nodes))
    np.add.at(moments, dofs, fine.areas[:, None] * np.einsum("tq,tqi->ti", weights, basis))
    return ScalarTransportOperators(
        matrix,
        load,
        moments,
        nodes,
        not np.any(effective) and not np.any(beta),
        not np.any(c) and not np.any(div_beta),
    )
