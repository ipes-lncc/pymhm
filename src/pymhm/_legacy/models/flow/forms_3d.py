"""Three-dimensional grad-grad flow forms and residual stabilization on tetrahedra."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm._legacy.models.flow.solver import _laplacian_inverse_bound
from pymhm.core.validation import FloatArray, positive_int
from pymhm.core.validation import real_array as _real
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


def resistance_values_3d(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate finite symmetric nonnegative scalar or 3-by-3 resistance tensors."""
    values = _real(field(points) if callable(field) else field, "drag")
    if values.ndim == 0 or values.shape == (len(points),):
        values = np.broadcast_to(values, (len(points),))[:, None, None] * np.eye(3)
    try:
        values = np.broadcast_to(values, (len(points), 3, 3))
    except ValueError as exc:
        raise ValueError("drag must return scalars or 3-by-3 tensors") from exc
    scale = np.max(np.abs(values), axis=(1, 2))
    if np.any(
        np.max(np.abs(values - values.swapaxes(-1, -2)), axis=(1, 2)) > 1e-12 * scale
    ) or np.any(np.linalg.eigvalsh(values) < 0):
        raise ValueError("drag must be symmetric and nonnegative")
    return values


def _nonnegative(value: Any, name: str) -> float:
    """Validate a scalar physical bound without discarding complex or Boolean data."""
    raw = np.asarray(value)
    if (
        raw.ndim != 0
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
        or not np.isfinite(raw)
        or raw < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(raw)


def _minimum_resistance_3d(drag: Any, supplied: Any) -> float:
    """Resolve a declared global lower eigenvalue bound, never infer a callback infimum."""
    bound = None if supplied is None else _nonnegative(supplied, "gamma_min")
    if isinstance(drag, CartesianCellField):
        if len(drag.spacing) != 3:
            raise ValueError("3D Cartesian drag requires three spatial dimensions")
        values = drag.values.reshape((-1, *drag.values.shape[3:]))
        minimum = float(
            np.linalg.eigvalsh(resistance_values_3d(values, np.zeros((len(values), 3))))[
                ..., 0
            ].min()
        )
    elif callable(drag):
        if bound is None:
            raise ValueError("variable drag requires an explicit gamma_min for minimum-2017")
        return bound
    else:
        minimum = float(np.linalg.eigvalsh(resistance_values_3d(drag, np.zeros((1, 3))))[0, 0])
    if bound is not None and bound > minimum:
        raise ValueError("gamma_min exceeds the minimum material eigenvalue")
    return minimum if bound is None else bound


def flow_contract_3d(
    viscosity: Any,
    drag: Any,
    advection: Any,
    divergence: Any,
    bound: Any,
    formulation: str,
    stabilization: str,
    gamma_min: Any,
    degree: int | None,
) -> tuple[float, Any, Any, float | None, float | None, int]:
    """Validate the common physical and polynomial contracts for 3D flow forms."""
    nu = _nonnegative(viscosity, "viscosity")
    if nu == 0:
        raise ValueError("viscosity must be positive")
    if formulation not in ("taylor-hood", "usfem", "oseen"):
        raise ValueError("formulation must be taylor-hood, usfem or oseen")
    if stabilization not in ("tensor-2025", "minimum-2017", "pointwise-2017"):
        raise ValueError("unknown USFEM stabilization")
    if formulation != "usfem" and stabilization != "tensor-2025":
        raise ValueError("USFEM stabilization choices require formulation=usfem")
    if gamma_min is not None and stabilization != "minimum-2017":
        raise ValueError("gamma_min is only used with minimum-2017")
    k = (
        (2 if formulation == "taylor-hood" else 1)
        if degree is None
        else positive_int(degree, "degree")
    )
    if k > 4 or (formulation == "taylor-hood" and k < 2):
        raise ValueError("velocity degree must be 2--4 for Taylor-Hood or 1--4 for equal order")
    if callable(advection):
        if divergence is None:
            raise ValueError("callable advection requires advection_divergence")
        if formulation == "oseen" and bound is None:
            raise ValueError("stabilized callable advection requires advection_bound")
        beta, div_beta = advection, divergence
    else:
        beta = vector_values_3d(advection, np.zeros((1, 3)))[0]
        if divergence is not None and (
            callable(divergence) or np.any(scalar_values_3d(divergence, np.zeros((1, 3))))
        ):
            raise ValueError("constant advection has zero divergence")
        div_beta = 0.0
        bound = np.linalg.norm(beta) if bound is None else bound
    beta_bound = None if bound is None else _nonnegative(bound, "advection_bound")
    if formulation == "usfem" and (callable(beta) or np.any(beta)):
        raise ValueError("advection requires Taylor-Hood or the oseen formulation")
    if formulation == "oseen":
        if callable(drag) or np.ndim(drag) != 0:
            raise ValueError("stabilized Oseen requires constant scalar drag")
        _nonnegative(drag, "drag")
    minimum = _minimum_resistance_3d(drag, gamma_min) if stabilization == "minimum-2017" else None
    return nu, beta, div_beta, beta_bound, minimum, k


@dataclass(frozen=True)
class Flow3DOperators:
    """Assembled velocity-pressure algebra and physical moments on one local mesh.

    Velocity coefficients are interleaved Cartesian components at continuous Pk
    nodes, followed by scalar continuous pressure coefficients. Taylor--Hood uses
    P(k-1) pressure; stabilized forms use Pk. Material moments support exact-zero
    translation tests; sampled positive resistances are never classified by a
    small absolute threshold.
    """

    matrix: sparse.csc_matrix
    load: FloatArray
    velocity_nodes: FloatArray
    pressure_nodes: FloatArray
    velocity_moments: FloatArray
    pressure_moments: FloatArray
    resistance_moment: FloatArray
    absolute_resistance_moment: FloatArray
    zero_columns: Any


def tetra_flow_operators(
    mesh: TetraMesh,
    *,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0, 0.0),
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    source: Any = (0.0, 0.0, 0.0),
    degree: int | None = None,
    formulation: str = "taylor-hood",
    stabilization: str = "tensor-2025",
    gamma_min: float | None = None,
    order: int = 5,
) -> Flow3DOperators:
    """Assemble -nu*Delta(u)+beta.grad(u)+Gamma*u+grad(p)=f, div(u)=0.

    The pressure test equation is multiplied by -1. Advection uses its skew
    form minus div(beta)*u.v/2; the natural traction is
    (nu*grad(u)-p*I)n-beta.n*u/2. USFEM subtracts the full trial/test momentum
    residual product and its forcing term. Oseen uses the adjoint convection
    sign in the test residual, plus positive grad-div stabilization.

    The polynomial inverse bound m=min(1/3,C) is computed per tetrahedron from
    C*h^2*||Delta(v)||^2 <= ||grad(v)||^2. The three USFEM coefficient choices
    preserve the declared 2D conventions. For variable materials, Gaussian
    quadrature is used; discontinuities must be resolved by the local mesh.
    No tetrahedral Cartesian cut integration is implied.
    """
    nu, beta, div_beta, beta_bound, minimum, k = flow_contract_3d(
        viscosity,
        drag,
        advection,
        advection_divergence,
        advection_bound,
        formulation,
        stabilization,
        gamma_min,
        degree,
    )
    bary, weights = tetrahedron_quadrature(max(positive_int(order, "order"), k + 2))
    udofs, unodes, basis, gradient, hessian = tetra_element_tabulate(mesh, k, bary)
    pk = k - 1 if formulation == "taylor-hood" else k
    pdofs, pnodes, pbasis, pgradient = tetra_tabulate(mesh, pk, bary)
    nt, nq, ns, _ = gradient.shape
    nv, npres, ps = len(unodes), len(pnodes), pbasis.shape[1]
    vdofs = (3 * udofs[:, :, None] + np.arange(3)).reshape(nt, -1)
    dofs = np.column_stack((vdofs, 3 * nv + pdofs))
    physical = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    flat = physical.reshape(-1, 3)
    resistance = resistance_values_3d(drag, flat).reshape(nt, nq, 3, 3)
    velocity = vector_values_3d(beta, flat).reshape(nt, nq, 3)
    div_values = scalar_values_3d(div_beta, flat).reshape(nt, nq)
    if beta_bound is not None and np.any(
        np.linalg.norm(velocity, axis=-1) > beta_bound * (1 + 32 * np.finfo(float).eps)
    ):
        raise ValueError("advection_bound is smaller than a sampled advection magnitude")
    stiffness = np.einsum("q,tqia,tqja,t->tij", weights, gradient, gradient, mesh.volumes)
    convection = np.einsum("tqa,tqia->tqi", velocity, gradient)
    transport = np.einsum("q,qi,tqj,t->tij", weights, basis, convection, mesh.volumes)
    scalar = nu * stiffness + (transport - transport.swapaxes(1, 2)) / 2
    scalar -= np.einsum("q,tq,qi,qj,t->tij", weights, div_values / 2, basis, basis, mesh.volumes)
    blocks = np.zeros((nt, 3 * ns + ps, 3 * ns + ps))
    blocks[:, : 3 * ns, : 3 * ns] = np.einsum("tij,ab->tiajb", scalar, np.eye(3)).reshape(
        nt, 3 * ns, 3 * ns
    )
    blocks[:, : 3 * ns, : 3 * ns] += np.einsum(
        "q,qi,qj,tqab,t->tiajb", weights, basis, basis, resistance, mesh.volumes
    ).reshape(nt, 3 * ns, 3 * ns)
    divergence = gradient.reshape(nt, nq, 3 * ns)
    cross = -np.einsum("q,tqi,qj,t->tij", weights, divergence, pbasis, mesh.volumes)
    blocks[:, : 3 * ns, 3 * ns :] = cross
    blocks[:, 3 * ns :, : 3 * ns] = cross.swapaxes(1, 2)
    force = vector_values_3d(source, flat).reshape(nt, nq, 3)
    element_load = np.zeros((nt, 3 * ns + ps))
    element_load[:, : 3 * ns] = np.einsum(
        "q,qi,tqa,t->tia", weights, basis, force, mesh.volumes
    ).reshape(nt, -1)
    if formulation != "taylor-hood":
        vertices = mesh.points[mesh.cells]
        diameters = np.max(
            np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=-1), axis=(1, 2)
        )
        inverse = _laplacian_inverse_bound(gradient, hessian, weights, diameters)
        viscous = 4 * nu / inverse
        eigenvalues = np.linalg.eigvalsh(resistance)
        if formulation == "oseen":
            advective = cast(float, beta_bound) * diameters
            tau = (
                diameters**2
                / (np.maximum(float(drag) * diameters**2, viscous) + np.maximum(viscous, advective))
            )[:, None]
            grad_div = advective * np.minimum(1.0, advective / viscous)
            blocks[:, : 3 * ns, : 3 * ns] += np.einsum(
                "q,t,tqi,tqj,t->tij", weights, grad_div, divergence, divergence, mesh.volumes
            )
        else:
            if stabilization == "pointwise-2017":
                bound = eigenvalues[..., 0]
            elif minimum is None:
                bound = eigenvalues[..., -1].max(axis=1)[:, None]
            else:
                if np.any(eigenvalues[..., 0] < minimum):
                    raise ValueError("gamma_min exceeds a sampled material eigenvalue")
                bound = np.full((nt, 1), minimum)
            tau = diameters[:, None] ** 2 / (
                np.maximum(bound * diameters[:, None] ** 2, viscous[:, None]) + viscous[:, None]
            )
        residual = np.zeros((nt, nq, 3, 3 * ns + ps))
        laplacian = -nu * np.trace(hessian, axis1=-2, axis2=-1)
        for a in range(3):
            for c in range(3):
                residual[:, :, a, c : 3 * ns : 3] = (
                    resistance[:, :, a, c, None] * basis + (a == c) * laplacian
                )
        residual[:, :, :, 3 * ns :] = pgradient.swapaxes(-1, -2)
        trial = residual.copy()
        if formulation == "oseen":
            for component in range(3):
                trial[:, :, component, component : 3 * ns : 3] += convection
                residual[:, :, component, component : 3 * ns : 3] -= convection
        blocks -= np.einsum("q,tq,tqai,tqaj,t->tij", weights, tau, residual, trial, mesh.volumes)
        element_load -= np.einsum(
            "q,tq,tqai,tqa,t->ti", weights, tau, residual, force, mesh.volumes
        )
    size = 3 * nv + npres
    matrix = _assemble_blocks(blocks, dofs, size)
    load = np.bincount(dofs.ravel(), weights=element_load.ravel(), minlength=size)
    umoment, pmoment = np.zeros(nv), np.zeros(npres)
    np.add.at(umoment, udofs, mesh.volumes[:, None] * (weights @ basis))
    np.add.at(pmoment, pdofs, mesh.volumes[:, None] * (weights @ pbasis))
    return Flow3DOperators(
        matrix,
        load,
        unodes,
        pnodes,
        umoment,
        pmoment,
        np.einsum("q,tqab,t->ab", weights, resistance, mesh.volumes),
        np.einsum("q,tqab,t->ab", weights, np.abs(resistance), mesh.volumes),
        np.all(resistance == 0, axis=(0, 1, 2)),
    )
