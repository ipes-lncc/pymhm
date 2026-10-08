"""Conservative tetrahedral scalar transport integration with explicit derivatives.

These numerical forms expose diffusion, skew transport, reaction and full
SUPG residuals. They do not choose local kernels, macroface spaces or a solver.
"""

from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d, vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


def coefficient_derivatives_3d(
    points: FloatArray,
    diffusion: Any,
    velocity: Any,
    velocity_divergence: Any,
    diffusion_divergence: Any,
    stabilization: str,
) -> tuple[Any, Any]:
    """Require declared variable derivatives and reject contradictory constant data."""
    if callable(velocity) and velocity_divergence is None:
        raise ValueError("variable velocity requires velocity_divergence")
    if stabilization == "supg" and callable(diffusion) and diffusion_divergence is None:
        raise ValueError("SUPG with variable diffusion requires diffusion_divergence")
    div_beta = 0.0 if velocity_divergence is None else velocity_divergence
    div_tensor = (0.0, 0.0, 0.0) if diffusion_divergence is None else diffusion_divergence
    if not callable(velocity) and np.any(scalar_values_3d(div_beta, points)):
        raise ValueError("constant velocity must have zero divergence")
    if not callable(diffusion) and np.any(vector_values_3d(div_tensor, points)):
        raise ValueError("constant diffusion must have zero divergence")
    return div_beta, div_tensor


def tetra_transport_operators(
    mesh: TetraMesh,
    degree: int,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    order: int = 6,
) -> tuple[Any, FloatArray, FloatArray, bool]:
    """Assemble skew conservative RAD and optional full-residual SUPG.

    The associated boundary flux is ``(-K grad(u)+beta*u/2).n``. The source
    equation is ``-div(K grad(u))+div(beta*u)+reaction*u=f``. Coercivity requires
    ``reaction+div(beta)/2 >= 0``; the implementation checks integration points.
    Returned physical moments integrate the nodal functions over the domain.
    Galerkin requests first derivatives only; SUPG also tabulates its strong-residual Hessian.
    """
    if stabilization not in ("galerkin", "supg"):
        raise ValueError("stabilization must be galerkin or supg")
    velocity_divergence, diffusion_divergence = coefficient_derivatives_3d(
        mesh.points,
        diffusion,
        velocity,
        velocity_divergence,
        diffusion_divergence,
        stabilization,
    )
    bary, weights = tetrahedron_quadrature(max(positive_int(order, "order"), degree + 2))
    if stabilization == "supg":
        dofs, nodes, basis, gradient, hessian = tetra_element_tabulate(mesh, degree, bary)
    else:
        dofs, nodes, basis, gradient = tetra_tabulate(mesh, degree, bary)
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    shape = points.shape[:2]
    flat = points.reshape(-1, 3)
    tensor = tensor_values_3d(diffusion, flat).reshape(*shape, 3, 3)
    beta = vector_values_3d(velocity, flat).reshape(*shape, 3)
    div_beta = scalar_values_3d(velocity_divergence, flat).reshape(shape)
    c = scalar_values_3d(reaction, flat).reshape(shape)
    effective = c + div_beta / 2
    if np.any(effective < 0):
        raise ValueError("reaction+div(velocity)/2 must be nonnegative")
    force = scalar_values_3d(source, flat).reshape(shape)
    blocks = np.einsum("t,q,tqia,tqab,tqjb->tij", mesh.volumes, weights, gradient, tensor, gradient)
    blocks += np.einsum("t,q,qi,qj,tq->tij", mesh.volumes, weights, basis, basis, effective)
    streamline = np.einsum("tqa,tqia->tqi", beta, gradient)
    transport = np.einsum("t,q,qi,tqj->tij", mesh.volumes, weights, basis, streamline)
    blocks += (transport - transport.swapaxes(1, 2)) / 2
    load = np.einsum("t,q,qi,tq->ti", mesh.volumes, weights, basis, force)
    if stabilization == "supg":
        div_tensor = vector_values_3d(diffusion_divergence, flat).reshape(*shape, 3)
        strong = (
            -np.einsum("tqab,tqiab->tqi", tensor, hessian)
            - np.einsum("tqa,tqia->tqi", div_tensor, gradient)
            + streamline
            + (c + div_beta)[:, :, None] * basis[None]
        )
        corners = mesh.points[mesh.cells]
        diameter = np.max(
            np.linalg.norm(corners[:, :, None] - corners[:, None, :], axis=-1), axis=(1, 2)
        )[:, None]
        tau = 1 / np.sqrt(
            (2 * np.linalg.norm(beta, axis=-1) / diameter) ** 2
            + (4 * np.linalg.eigvalsh(tensor)[..., -1] / diameter**2) ** 2
            + (c + div_beta) ** 2
        )
        blocks += np.einsum("t,q,tqi,tqj,tq->tij", mesh.volumes, weights, streamline, strong, tau)
        load += np.einsum("t,q,tqi,tq,tq->ti", mesh.volumes, weights, streamline, force, tau)
    rows = np.broadcast_to(dofs[:, :, None], blocks.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], blocks.shape).ravel()
    matrix = sparse.coo_matrix(
        (blocks.ravel(), (rows, columns)), shape=(len(nodes), len(nodes))
    ).tocsc()
    assembled = np.zeros(len(nodes))
    moments = np.zeros(len(nodes))
    np.add.at(assembled, dofs, load)
    np.add.at(moments, dofs, mesh.volumes[:, None] * (weights @ basis))
    return matrix, assembled, moments, not np.any(beta) and not np.any(effective)
