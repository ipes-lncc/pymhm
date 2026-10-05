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

from pymhm import LocalEquations
from pymhm._legacy.models.flow.solver import _resistance_values
from pymhm._legacy.models.vector import VectorSolution
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.inequalities import laplacian_inverse_bound
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import element_tabulate, tabulate, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.triangle import TriangleMesh


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
) -> LocalEquations:
    """Declare grad-grad, skew-advection, drag and symmetric div/pressure blocks.

    With a residual-weight callback, subtract the full strong residual pairing
    ``tau*(-nu*Delta(u)+drag*u+grad(p),-nu*Delta(v)+drag*v+grad(q))`` and its
    source pairing. That contract requires equal velocity/pressure degrees and
    zero advection. The callable receives the fine mesh, viscosity, drag and
    tabulation/resistance keyword data, returning one weight per fine cell. Translation moments
    are physical velocity integrals, and pressure weights are physical integrals.
    B is signed trace integration and C=-B.T; D and g are explicitly zero.
    """
    if viscosity <= 0 or not np.isfinite(viscosity):
        raise ValueError("this application needs positive finite viscosity")
    beta = np.asarray(advection, dtype=float)
    if beta.shape != (2,) or not np.isfinite(beta).all():
        raise ValueError("this application declares constant two-dimensional advection")
    if residual_weight is not None and (
        space.velocity_degree != space.pressure_degree or np.any(beta)
    ):
        raise ValueError("the declared residual pairing requires equal degrees and zero advection")
    fine = space.mesh.submesh(cell, space.refinement)
    bary, weights = triangle_quadrature(max(space.quadrature_order, space.velocity_degree + 2))
    udofs, nodes, values, grad, hessian = tabulate(fine, space.velocity_degree, bary)
    bary_cells = np.broadcast_to(bary, (len(fine.cells), *bary.shape))
    basis = np.broadcast_to(values, (*bary_cells.shape[:2], values.shape[1]))
    if space.pressure_degree == space.velocity_degree:
        pdofs, pnodes, pbasis, pgrad = udofs, nodes, basis, grad
    else:
        pdofs, pnodes, pbasis, pgrad, _ = element_tabulate(fine, space.pressure_degree, bary_cells)
    nc, nq, ns = basis.shape
    np_local, nv, npres = pbasis.shape[-1], len(nodes), len(pnodes)
    vdofs = (2 * udofs[:, :, None] + np.arange(2)).reshape(nc, -1)
    dofs = np.column_stack((vdofs, 2 * nv + pdofs))
    qweights = np.broadcast_to(weights, (nc, nq))
    physical = np.einsum("tqi,tij->tqj", bary_cells, fine.points[fine.cells])
    convection = np.broadcast_to(beta, (nc, nq, 2))
    resistance = _resistance_values(drag, physical.reshape(-1, 2)).reshape(nc, nq, 2, 2)
    stiffness = np.einsum("tq,tqia,tqja,t->tij", qweights, grad, grad, fine.areas)
    advective = np.einsum("tq,tqi,tqa,tqja,t->tij", qweights, basis, convection, grad, fine.areas)
    scalar = viscosity * stiffness + (advective - advective.swapaxes(1, 2)) / 2
    blocks = np.zeros((nc, 2 * ns + np_local, 2 * ns + np_local))
    blocks[:, : 2 * ns, : 2 * ns] = np.einsum("tij,ab->tiajb", scalar, np.eye(2)).reshape(
        nc, 2 * ns, 2 * ns
    )
    blocks[:, : 2 * ns, : 2 * ns] += np.einsum(
        "tq,tqi,tqj,tqab,t->tiajb", qweights, basis, basis, resistance, fine.areas
    ).reshape(nc, 2 * ns, 2 * ns)
    divergence = grad.reshape(nc, nq, 2 * ns)
    cross = -np.einsum("tq,tqi,tqj,t->tij", qweights, divergence, pbasis, fine.areas)
    blocks[:, : 2 * ns, 2 * ns :] = cross
    blocks[:, 2 * ns :, : 2 * ns] = cross.swapaxes(1, 2)
    force = vector_values(source, physical.reshape(-1, 2)).reshape(nc, nq, 2)
    element_load = np.zeros((nc, 2 * ns + np_local))
    element_load[:, : 2 * ns] = np.einsum(
        "tq,tqi,tqa,t->tia", qweights, basis, force, fine.areas
    ).reshape(nc, -1)
    if residual_weight is not None:
        tau = np.asarray(
            residual_weight(
                fine,
                viscosity,
                drag,
                tabulation=(grad, hessian, weights),
                resistance=resistance,
            ),
            dtype=float,
        )
        if tau.shape != (nc,) or not np.isfinite(tau).all() or np.any(tau <= 0):
            raise ValueError("the residual weight must be positive on each fine cell")
        residual = np.zeros((nc, nq, 2, 2 * ns + np_local))
        laplacian = -viscosity * np.trace(hessian, axis1=-2, axis2=-1)
        for component in range(2):
            for coordinate in range(2):
                residual[:, :, component, coordinate : 2 * ns : 2] = (
                    resistance[:, :, component, coordinate, None] * basis
                    + (component == coordinate) * laplacian
                )
        residual[:, :, :, 2 * ns :] = pgrad.swapaxes(-1, -2)
        blocks -= np.einsum(
            "tq,tqai,tqaj,t->tij", qweights * tau[:, None], residual, residual, fine.areas
        )
        element_load -= np.einsum(
            "tq,tqai,tqa,t->ti", qweights * tau[:, None], residual, force, fine.areas
        )
    size = 2 * nv + npres
    a = assemble_element_blocks(blocks, dofs, dofs, (size, size))
    load = np.bincount(dofs.ravel(), weights=element_load.ravel(), minlength=size)
    b = np.zeros((size, len(space.skeleton.cell_dofs(cell))))
    b[: 2 * nv] = np.kron(
        trace_coupling(space.mesh, cell, fine, space.skeleton, space.velocity_degree), np.eye(2)
    )
    translations = np.zeros((size, 2))
    translations[: 2 * nv] = np.tile(np.eye(2), (nv, 1))
    velocity_weights = np.zeros(nv)
    np.add.at(
        velocity_weights,
        udofs,
        fine.areas[:, None] * np.einsum("tq,tqi->ti", qweights, basis),
    )
    moments = np.zeros_like(translations)
    moments[: 2 * nv] = (velocity_weights[:, None, None] * np.eye(2)).reshape(2 * nv, 2)
    pressure_weights = np.zeros(size)
    np.add.at(
        pressure_weights,
        2 * nv + pdofs,
        fine.areas[:, None] * np.einsum("tq,tqi->ti", qweights, pbasis),
    )
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        space.skeleton.cell_dofs(cell),
        d=0,
        g=0,
        kernel=translations if not np.any(resistance) and not np.any(beta) else None,
        coarse_basis=translations if np.any(resistance) or np.any(beta) else None,
        moments=moments,
        metadata=(fine, nv, pressure_weights),
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
