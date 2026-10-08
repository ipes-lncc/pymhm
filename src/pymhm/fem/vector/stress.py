"""Public physical moment and volume integration owners; no solver dispatch."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.elasticity import compliance_products
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh

_DEFAULT_FAMILY = BDMFamily()


def mixed_elasticity_operators(
    mesh: TriangleMesh,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    family: BDMFamily = _DEFAULT_FAMILY,
    compliance: Any = None,
) -> tuple[Any, Any, Any, FloatArray]:
    """Assemble compliance, divergence, asymmetry and body-force moments."""
    bary, weights = triangle_quadrature(order)
    values, divergence = family.basis(mesh, bary)
    scalar_basis = reference_basis(family.polynomial_degree - 1, bary)[0]
    scalar_size = scalar_basis.shape[1]
    local_size = family.local_size
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    tensors = np.zeros((*values.shape[:2], 2 * local_size, 2, 2))
    tensors[:, :, 0::2, 0], tensors[:, :, 1::2, 1] = values, values
    if compliance is None:
        mu = scalar_values(lame_mu, points.reshape(-1, 2)).reshape(points.shape[:2])
        bulk = bulk_compliance(lame_lambda, mu.ravel(), points.reshape(-1, 2)).reshape(mu.shape)
        trace = np.trace(tensors, axis1=-2, axis2=-1)
        deviator = tensors - trace[..., None, None] * np.eye(2) / 2
        # The split avoids cancellation near the incompressible limit.
        mass = np.einsum(
            "q,tq,tqiab,tqjab,t->tij", weights, 1 / (2 * mu), deviator, deviator, mesh.areas
        )
        mass += np.einsum("q,tq,tqi,tqj,t->tij", weights, bulk / 2, trace, trace, mesh.areas)
    else:
        products, _, _ = compliance_products(compliance, points, tensors)
        mass = np.einsum("q,tqij,t->tij", weights, products, mesh.areas)
    div = np.zeros((len(mesh.cells), 2 * scalar_size, 2 * local_size))
    scalar_div = np.einsum("q,qi,tqj,t->tij", weights, scalar_basis, divergence, mesh.areas)
    div[:, 0::2, 0::2], div[:, 1::2, 1::2] = scalar_div, scalar_div
    asym = np.einsum(
        "q,qi,tqj,t->tij",
        weights,
        scalar_basis,
        tensors[..., 0, 1] - tensors[..., 1, 0],
        mesh.areas,
    )
    force = np.einsum(
        "q,qi,tqa,t->tia",
        weights,
        scalar_basis,
        vector_values(source, points.reshape(-1, 2)).reshape(*points.shape[:2], 2),
        mesh.areas,
    )
    stress_dofs = (2 * family.dofs(mesh)[:, :, None] + np.arange(2)).reshape(
        -1, 2 * family.local_size
    )
    nstress = 2 * family.size(mesh)
    nc = len(mesh.cells)
    return (
        assemble_element_blocks(mass, stress_dofs, stress_dofs, (nstress, nstress)),
        assemble_element_blocks(
            div,
            np.arange(2 * scalar_size * nc).reshape(nc, 2 * scalar_size),
            stress_dofs,
            (2 * scalar_size * nc, nstress),
        ),
        assemble_element_blocks(
            asym,
            np.arange(scalar_size * nc).reshape(nc, scalar_size),
            stress_dofs,
            (scalar_size * nc, nstress),
        ),
        force.ravel(),
    )


def bulk_compliance(lame_lambda: Any, mu: FloatArray, points: FloatArray) -> FloatArray:
    """Evaluate 1/[2(mu+lambda)], including a zero incompressible compliance."""
    raw = lame_lambda(points) if callable(lame_lambda) else lame_lambda
    if np.iscomplexobj(raw):
        raise ValueError("Lamé lambda must be real")
    lam = np.broadcast_to(np.asarray(raw, dtype=float), (len(points),))
    if np.any(np.isnan(lam)) or np.any(lam < 0) or np.any(mu <= 0):
        raise ValueError("Lamé lambda must be nonnegative and mu strictly positive")
    scale = np.maximum(mu, lam)
    return np.asarray((0.5 / scale) / (1 + np.minimum(mu, lam) / scale))


def stress_trace_moments(
    mesh: TriangleMesh,
    size: int,
    lame_lambda: Any,
    lame_mu: Any,
    order: int,
    family: BDMFamily = _DEFAULT_FAMILY,
    compliance: Any = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Integrate trace(sigma) and its physical bulk-compliance-weighted moment."""
    bary, weights = triangle_quadrature(order)
    basis, _ = family.basis(mesh, bary)
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
    # Row zero contributes sigma_xx; row one contributes sigma_yy.
    traces = basis.reshape(len(mesh.cells), len(bary), 2 * family.local_size)
    dofs = (2 * family.dofs(mesh)[:, :, None] + np.arange(2)).reshape(-1, 2 * family.local_size)
    plain = np.einsum("q,tqi,t->ti", weights, traces, mesh.areas)
    if compliance is None:
        mu = scalar_values(lame_mu, points)
        bulk = bulk_compliance(lame_lambda, mu, points).reshape(len(mesh.cells), -1)
        weighted = np.einsum("q,tq,tqi,t->ti", weights, bulk, traces, mesh.areas)
        scale = float(bulk.max())
    else:
        tensors = np.zeros((*basis.shape[:2], 2 * family.local_size, 2, 2))
        tensors[:, :, 0::2, 0], tensors[:, :, 1::2, 1] = basis, basis
        _, weighted_trace, scale = compliance_products(compliance, points, tensors)
        weighted = np.einsum("q,tqi,t->ti", weights, weighted_trace, mesh.areas)
    return (
        np.bincount(dofs.ravel(), weights=plain.ravel(), minlength=size),
        np.bincount(dofs.ravel(), weights=weighted.ravel(), minlength=size),
        scale,
    )


def traction_mapping(
    mesh: TriangleMesh,
    cell: int,
    fine: TriangleMesh,
    skeleton: SkeletonSpace,
    family: BDMFamily = _DEFAULT_FAMILY,
) -> FloatArray:
    """Apply the shared BDM normal-moment map to each stress row."""
    return np.asarray(np.kron(family.trace_map(mesh, cell, fine, skeleton), np.eye(2)), dtype=float)


def rigid_values(points: FloatArray, center: FloatArray) -> FloatArray:
    """Evaluate two translations and one centered rigid rotation."""
    values = np.zeros((*points.shape[:-1], 2, 3))
    values[..., 0, 0], values[..., 1, 1] = 1, 1
    values[..., 0, 2] = -(points[..., 1] - center[1])
    values[..., 1, 2] = points[..., 0] - center[0]
    return values


def displacement_rigid_moments(
    mesh: TriangleMesh, center: FloatArray, family: BDMFamily = _DEFAULT_FAMILY
) -> FloatArray:
    """Integrate discontinuous displacement basis against the three rigid motions."""
    degree = family.polynomial_degree - 1
    bary, weights = triangle_quadrature(degree + 2)
    basis = reference_basis(degree, bary)[0]
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    return np.einsum(
        "q,qi,tqak,t->tiak", weights, basis, rigid_values(points, center), mesh.areas
    ).reshape(-1, 3)
