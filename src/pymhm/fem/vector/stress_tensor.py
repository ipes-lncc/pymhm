"""Public physical moment and volume integration owners; no solver dispatch."""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.assembly import assemble_element_blocks as _scatter
from pymhm.fem.hdiv.tensor_rt import tensor_rt_basis, tensor_rt_dofs
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature
from pymhm.fem.vector.stress import bulk_compliance as _bulk_compliance
from pymhm.fem.vector.stress import rigid_values as _rigid_values
from pymhm.materials.elasticity import compliance_products
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh


def complete_rotation_basis(degree: int, points: FloatArray) -> FloatArray:
    """Tabulate total-degree Legendre products P_s, excluding the Q_s cross corners."""
    x, y = np.moveaxis(points, -1, 0)
    lx, ly = legendre_values(2 * x - 1, degree), legendre_values(2 * y - 1, degree)
    return np.stack(
        [lx[..., a] * ly[..., b] for b in range(degree + 1) for a in range(degree + 1 - b)], axis=-1
    )


def tensor_rigid_moments(
    mesh: CartesianMacroMesh, degree: int, center: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Project rigid displacement into Q_s and return its physical moment matrix."""
    points, weights = quadrilateral_quadrature(degree + 2)
    _, _, basis = tensor_rt_basis(mesh, 1, degree - 1, points)
    physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
    moments = np.einsum(
        "q,qi,tqak,t->tiak", weights, basis, _rigid_values(physical, center), mesh.areas
    )
    mass = mesh.areas[:, None] * np.einsum("q,qi,qi->i", weights, basis, basis)
    return (moments / mass[:, :, None, None]).reshape(-1, 3), moments.reshape(-1, 3)


def tensor_stress_operators(
    mesh: CartesianMacroMesh,
    degree: int,
    enrichment: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> tuple:
    """Assemble compliance, Q_s divergence, P_s asymmetry and physical force moments."""
    points, weights = quadrilateral_quadrature(order)
    values, divergence, displacement = tensor_rt_basis(mesh, degree, enrichment, points)
    rotation = complete_rotation_basis(degree + enrichment, points)
    physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
    width = values.shape[2]
    tensors = np.zeros((*values.shape[:2], 2 * width, 2, 2))
    tensors[:, :, 0::2, 0], tensors[:, :, 1::2, 1] = values, values
    trace = np.trace(tensors, axis1=-2, axis2=-1)
    if compliance is None:
        mu = scalar_values(lame_mu, physical.reshape(-1, 2)).reshape(physical.shape[:2])
        bulk = _bulk_compliance(lame_lambda, mu.ravel(), physical.reshape(-1, 2)).reshape(mu.shape)
        deviator = tensors - trace[..., None, None] * np.eye(2) / 2
        mass = np.einsum(
            "q,tq,tqiab,tqjab,t->tij", weights, 1 / (2 * mu), deviator, deviator, mesh.areas
        )
        mass += np.einsum("q,tq,tqi,tqj,t->tij", weights, bulk / 2, trace, trace, mesh.areas)
        weighted_trace = bulk[:, :, None] * trace
        compliance_scale = float(bulk.max())
    else:
        products, weighted_trace, compliance_scale = compliance_products(
            compliance, physical, tensors
        )
        mass = np.einsum("q,tqij,t->tij", weights, products, mesh.areas)
    scalar_div = np.einsum("q,qi,tqj,t->tij", weights, displacement, divergence, mesh.areas)
    nd, nr = displacement.shape[1], rotation.shape[1]
    div = np.zeros((len(mesh.cells), 2 * nd, 2 * width))
    div[:, 0::2, 0::2], div[:, 1::2, 1::2] = scalar_div, scalar_div
    asym = np.einsum(
        "q,qi,tqj,t->tij", weights, rotation, tensors[..., 0, 1] - tensors[..., 1, 0], mesh.areas
    )
    force = np.einsum(
        "q,qi,tqa,t->tia",
        weights,
        displacement,
        vector_values(source, physical.reshape(-1, 2)).reshape(*physical.shape[:2], 2),
        mesh.areas,
    )
    dofs = (2 * tensor_rt_dofs(mesh, degree, enrichment)[:, :, None] + np.arange(2)).reshape(
        -1, 2 * width
    )
    ns, nu, nrot = int(dofs.max()) + 1, 2 * nd * len(mesh.cells), nr * len(mesh.cells)
    plain = np.bincount(
        dofs.ravel(),
        weights=np.einsum("q,tqi,t->ti", weights, trace, mesh.areas).ravel(),
        minlength=ns,
    )
    weighted = np.bincount(
        dofs.ravel(),
        weights=np.einsum("q,tqi,t->ti", weights, weighted_trace, mesh.areas).ravel(),
        minlength=ns,
    )
    return (
        _scatter(mass, dofs, dofs, (ns, ns)),
        _scatter(div, np.arange(nu).reshape(len(mesh.cells), -1), dofs, (nu, ns)),
        _scatter(asym, np.arange(nrot).reshape(len(mesh.cells), -1), dofs, (nrot, ns)),
        force.ravel(),
        plain,
        weighted,
        compliance_scale,
    )
