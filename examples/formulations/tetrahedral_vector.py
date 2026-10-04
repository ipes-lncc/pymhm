"""Declare tetrahedral mixed vector equations without physical solver dispatch.

The application uses existing tetrahedral tabulation, scalar trace integration,
element scatter, rigid-mode evaluation and physical inverse-bound kernels. Global
forms, boundary moments and gauges remain explicit in the calling notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import LocalEquations
from pymhm._legacy.models.elasticity.mixed_pressure_3d import GaLS3DSolution
from pymhm._legacy.models.elasticity.pressure_forms_3d import _strain_inverse_bound
from pymhm._legacy.models.elasticity.primal_3d import _KELVIN3, _boundary_vector, rigid_modes_3d
from pymhm._legacy.models.flow.solver_3d import Flow3DSolution
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


@dataclass(frozen=True)
class TetrahedralVectorSpace:
    """Declare continuous vector/scalar degrees, dyadic refinement and trace modes."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    vector_degree: int = 2
    pressure_degree: int = 1
    refinement: int = 2
    quadrature_order: int = 5


def vector_boundary_moments(
    skeleton: TriangularSkeleton, datum: Any, traction: dict[int, Any], order: int
) -> tuple[Any, dict[int, float]]:
    """Delegate Cartesian boundary moments to the existing scalar-trace projection owner.

    The datum is the weak vector Dirichlet field. Supplied physical outward
    tractions project to negative fixed multiplier coefficients. This operation
    selects no physical local operator or solution method.
    """
    return _boundary_vector(skeleton, datum, traction, order)


def _trace_blocks(
    cell: int, fine: TetraMesh, size: int, nv: int, data: TetrahedralVectorSpace
) -> tuple[Any, Any]:
    """Lift signed scalar trace integration to interleaved Cartesian vector coordinates."""
    b = np.zeros((size, 3 * len(data.skeleton.cell_dofs(cell))))
    b[: 3 * nv] = np.kron(
        tetra_trace_coupling(data.mesh, cell, fine, data.skeleton, data.vector_degree), np.eye(3)
    )
    indices = (3 * data.skeleton.cell_dofs(cell)[:, None] + np.arange(3)).ravel()
    return b, indices


def tetra_velocity_pressure_equations(
    cell: int, *, data: TetrahedralVectorSpace, source: Any, viscosity: float = 1.0
) -> LocalEquations:
    """Write grad-grad, symmetric pressure/divergence and negative trace-balance forms.

    This example declares Taylor–Hood P2/P1, zero drag and zero advection. Three
    translations are its local kernel and their moments integrate physical
    velocity. The pressure mean is a separately prescribed global constraint.
    No residual stabilization or three-dimensional Brinkman-extreme claim is made.
    """
    if data.vector_degree != 2 or data.pressure_degree != 1:
        raise ValueError("this tetrahedral velocity application declares P2/P1")
    fine = data.mesh.submesh(cell, data.refinement)
    bary, weights = tetrahedron_quadrature(max(data.quadrature_order, 4))
    udofs, nodes, basis, gradient, _ = tetra_element_tabulate(fine, 2, bary)
    pdofs, pnodes, pbasis, _ = tetra_tabulate(fine, 1, bary)
    nc, nq, ns, _ = gradient.shape
    nv, npres, ps = len(nodes), len(pnodes), pbasis.shape[1]
    vdofs = (3 * udofs[:, :, None] + np.arange(3)).reshape(nc, -1)
    dofs = np.column_stack((vdofs, 3 * nv + pdofs))
    points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    stiffness = np.einsum("q,tqia,tqja,t->tij", weights, gradient, gradient, fine.volumes)
    blocks = np.zeros((nc, 3 * ns + ps, 3 * ns + ps))
    blocks[:, : 3 * ns, : 3 * ns] = np.einsum(
        "tij,ab->tiajb", viscosity * stiffness, np.eye(3)
    ).reshape(nc, 3 * ns, 3 * ns)
    divergence = gradient.reshape(nc, nq, 3 * ns)
    cross = -np.einsum("q,tqi,qj,t->tij", weights, divergence, pbasis, fine.volumes)
    blocks[:, : 3 * ns, 3 * ns :] = cross
    blocks[:, 3 * ns :, : 3 * ns] = cross.swapaxes(1, 2)
    force = vector_values_3d(source, points.reshape(-1, 3)).reshape(nc, nq, 3)
    load_blocks = np.zeros((nc, 3 * ns + ps))
    load_blocks[:, : 3 * ns] = np.einsum(
        "q,qi,tqa,t->tia", weights, basis, force, fine.volumes
    ).reshape(nc, -1)
    size = 3 * nv + npres
    a = assemble_element_blocks(blocks, dofs, dofs, (size, size))
    load = np.bincount(dofs.ravel(), weights=load_blocks.ravel(), minlength=size)
    b, indices = _trace_blocks(cell, fine, size, nv, data)
    kernel = np.zeros((size, 3))
    kernel[: 3 * nv] = np.tile(np.eye(3), (nv, 1))
    velocity_weights = np.zeros(nv)
    np.add.at(velocity_weights, udofs, fine.volumes[:, None] * (weights @ basis))
    moments = np.zeros_like(kernel)
    moments[: 3 * nv] = (velocity_weights[:, None, None] * np.eye(3)).reshape(3 * nv, 3)
    pressure_weights = np.zeros(size)
    np.add.at(pressure_weights, 3 * nv + pdofs, fine.volumes[:, None] * (weights @ pbasis))
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        indices,
        d=0,
        g=0,
        kernel=kernel,
        moments=moments,
        metadata=(fine, nv, pressure_weights),
    )


def tetra_displacement_pressure_equations(
    cell: int,
    *,
    data: TetrahedralVectorSpace,
    source: Any,
    lame_lambda: float = np.inf,
    lame_mu: float = 1.0,
) -> LocalEquations:
    """Write GaLS P2/P2 with its physically computed tetrahedral inverse bound.

    Constant shear gives R(u,p)=div(2*mu*epsilon(u))-grad(p). The negative
    residual product and positive source pairing use alpha at half the sufficient
    bound, computed by the existing pure six-rigid-mode inverse-bound owner.
    It is computed on each physical tetrahedral local mesh and not inherited from
    a two-dimensional constant. Kernels and mass moments use the declared global
    volume center. Pressure is Herrmann pressure and B's multiplier is -sigma*n.
    """
    if data.vector_degree != 2 or data.pressure_degree != 2:
        raise ValueError("this displacement application declares GaLS P2/P2")
    if lame_mu <= 0 or not np.isfinite(lame_mu) or lame_lambda <= 0 or np.isnan(lame_lambda):
        raise ValueError("constant positive shear and positive/infinite lambda are required")
    fine = data.mesh.submesh(cell, data.refinement)
    vertices = data.mesh.points[data.mesh.cells[cell]]
    diameter = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max()
    center = (
        data.mesh.volumes @ data.mesh.points[data.mesh.cells].mean(axis=1) / data.mesh.volumes.sum()
    )
    bary, weights = tetrahedron_quadrature(max(data.quadrature_order, 4))
    dofs, nodes, basis, gradient, hessian = tetra_element_tabulate(fine, 2, bary)
    pdofs, pnodes, pbasis, pgradient = tetra_tabulate(fine, 2, bary)
    nc, nq, ns, _ = gradient.shape
    nv, npres, nps = len(nodes), len(pnodes), pbasis.shape[1]
    udofs = (3 * dofs[:, :, None] + np.arange(3)).reshape(nc, 3 * ns)
    all_dofs = np.column_stack((udofs, 3 * nv + pdofs))
    points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
    mu = np.full(points.shape[:2], lame_mu)
    compliance = np.full(mu.shape, 0.0 if np.isinf(lame_lambda) else 1 / lame_lambda)
    strain = np.einsum("aij,tqnj->tqani", _KELVIN3, gradient).reshape(*mu.shape, 6, 3 * ns)
    divergence = gradient.reshape(*mu.shape, 3 * ns)
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)
    strong = np.empty((*mu.shape, 3, 3 * ns))
    for axis in range(3):
        for component in range(3):
            strong[:, :, axis, component::3] = (
                hessian[:, :, :, axis, component] + (axis == component) * laplacian
            ) / 2
    blocks = np.zeros((nc, 3 * ns + nps, 3 * ns + nps))
    blocks[:, : 3 * ns, : 3 * ns] = np.einsum(
        "t,q,tqai,tqaj,tq->tij", fine.volumes, weights, strain, strain, 2 * mu
    )
    mixed = -np.einsum("t,q,tqi,qj->tij", fine.volumes, weights, divergence, pbasis)
    blocks[:, : 3 * ns, 3 * ns :] = mixed
    blocks[:, 3 * ns :, : 3 * ns] = mixed.swapaxes(1, 2)
    blocks[:, 3 * ns :, 3 * ns :] = -np.einsum(
        "t,q,qi,qj,tq->tij", fine.volumes, weights, pbasis, pbasis, compliance
    )
    force = vector_values_3d(source, points.reshape(-1, 3)).reshape(points.shape)
    load_blocks = np.zeros((nc, 3 * ns + nps))
    load_blocks[:, : 3 * ns] = np.einsum(
        "t,q,qi,tqa->tia", fine.volumes, weights, basis, force
    ).reshape(nc, 3 * ns)
    fine_vertices = fine.points[fine.cells]
    lengths = np.max(
        np.linalg.norm(fine_vertices[:, :, None] - fine_vertices[:, None, :], axis=-1), axis=(1, 2)
    )
    bound = _strain_inverse_bound(strain, strong, weights, lengths, float(diameter)) / (2 * lame_mu)
    alpha = bound / 2
    residual = np.concatenate(
        (2 * mu[:, :, None, None] * strong, -pgradient.swapaxes(-1, -2)), axis=-1
    )
    blocks -= np.einsum(
        "t,q,tqai,tqaj->tij", alpha * lengths**2 * fine.volumes, weights, residual, residual
    )
    load_blocks += np.einsum(
        "t,q,tqai,tqa->ti", alpha * lengths**2 * fine.volumes, weights, residual, force
    )
    size = 3 * nv + npres
    a = assemble_element_blocks(blocks, all_dofs, all_dofs, (size, size))
    load = np.bincount(all_dofs.ravel(), weights=load_blocks.ravel(), minlength=size)
    b, indices = _trace_blocks(cell, fine, size, nv, data)
    rigid = rigid_modes_3d(nodes, center).reshape(3 * nv, 6)
    mass = assemble_element_blocks(
        np.einsum("t,q,qi,qj->tij", fine.volumes, weights, basis, basis), dofs, dofs, (nv, nv)
    )
    kernel, moments = np.zeros((size, 6)), np.zeros((size, 6))
    kernel[: 3 * nv] = rigid
    moments[: 3 * nv] = sparse.kron(mass, sparse.eye(3)) @ rigid
    pressure_weights, compliance_weights = np.zeros(size), np.zeros(size)
    np.add.at(pressure_weights, 3 * nv + pdofs, fine.volumes[:, None] * (weights @ pbasis))
    np.add.at(
        compliance_weights,
        3 * nv + pdofs,
        np.einsum("t,q,qi,tq->ti", fine.volumes, weights, pbasis, compliance),
    )
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        indices,
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


def tetra_velocity_fields(
    system: MultiscaleSystem,
    solution: HybridSolution,
    data: TetrahedralVectorSpace,
    viscosity: float = 1.0,
) -> Flow3DSolution:
    """Interpret the declared coefficients in the unchanged grad-grad flow field record."""
    return Flow3DSolution(
        data.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(
            field[: 3 * record[1]].reshape(-1, 3)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        tuple(
            field[3 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        solution,
        data.vector_degree,
        data.pressure_degree,
        viscosity,
        (0.0, 0.0, 0.0),
        "taylor-hood",
    )


def tetra_displacement_fields(
    system: MultiscaleSystem,
    solution: HybridSolution,
    data: TetrahedralVectorSpace,
    lame_lambda: float = np.inf,
    lame_mu: float = 1.0,
) -> GaLS3DSolution:
    """Interpret displacement, Herrmann pressure and Cauchy stress in the executed basis."""
    return GaLS3DSolution(
        data.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(
            field[: 3 * record[1]].reshape(-1, 3)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        tuple(
            field[3 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        solution,
        data.vector_degree,
        data.pressure_degree,
        lame_lambda,
        lame_mu,
        "gals",
        tuple(record[3] for record in system.local_metadata),
    )
