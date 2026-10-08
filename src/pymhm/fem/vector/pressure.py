"""Herrmann elasticity volume forms and physical moments on triangular meshes."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.vector.elasticity import rigid_modes as _rigid
from pymhm.fem.vector.elasticity import strain_and_divergence as _strain_and_divergence
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.compliance import compressibility_values as _compressibility
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


def strain_inverse_bound(
    strain: FloatArray, strong: FloatArray, weights: FloatArray, h: FloatArray, diameter: float
) -> float:
    """Bound Eq. (4.4) by elementwise generalized eigenvalues modulo rigid motions.

    Taking the minimum with one prevents a refinement-dependent growth of the
    stabilization. No PDE solution or reference error enters this computation.
    """
    energy = np.einsum("q,tqai,a,tqaj->tij", weights, strain, [1, 1, 0.5], strain)
    residual = np.einsum("q,tqai,tqaj->tij", weights, strong, strong)
    maximum = 1.0
    for mass, second, length in zip(energy, residual, h, strict=True):
        eigenvalues, vectors = linalg.eigh(mass)
        selected = eigenvalues > 1e-11 * eigenvalues[-1]
        if np.count_nonzero(selected) != len(eigenvalues) - 3:
            raise ValueError("inverse inequality requires exactly 3 resolved kernel modes")
        whitening = vectors[:, selected] / np.sqrt(eigenvalues[selected])
        operator = length**2 * (mass / diameter**2 + second)
        maximum = max(maximum, float(linalg.eigvalsh(whitening.T @ operator @ whitening)[-1]))
    return 1 / maximum


@dataclass(frozen=True)
class ElasticityPressureOperators:
    """Displacement/pressure volume matrix, rigid modes and physical integral moments."""

    matrix: Any
    load: FloatArray
    displacement_nodes: FloatArray
    pressure_nodes: FloatArray
    kernel: FloatArray
    rigid_moments: FloatArray
    pressure_moments: FloatArray
    compliance_moments: FloatArray
    compliance_scale: float
    stabilization_alpha: float


def triangle_elasticity_pressure_operators(
    fine: TriangleMesh,
    *,
    degree: int,
    formulation: str,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    diameter: float,
    rigid_center: FloatArray,
    stabilization_alpha: float | None = None,
    lame_mu_gradient: Any = (0.0, 0.0),
    shear_bounds: tuple[float, float, float] | None = None,
) -> ElasticityPressureOperators:
    """Integrate 2*mu*epsilon, div/pressure, compliance and optional GaLS residual.

    Pk/Pk GaLS subtracts (alpha*h²*R(u,p),R(v,q)) and adds
    (alpha*h²*f,R(v,q)), R=div(2*mu*epsilon(u))-grad(p). Taylor--Hood
    instead uses Pk/P(k-1) with no residual. Three rigid modes and moments
    are evaluated about the explicitly supplied physical center. Lambda=infinity
    has zero compressibility. Variable GaLS shear requires declared valid bounds
    and analytical derivatives; quadrature checks do not certify between samples.
    """
    bary, weights = triangle_quadrature(order)
    dofs, nodes, basis, gradients, hessian = tabulate(fine, degree, bary)
    pressure_degree = degree if formulation == "gals" else degree - 1
    pdofs, pnodes, pbasis, pgradients, _ = tabulate(fine, pressure_degree, bary)
    nv, npres = len(nodes), len(pnodes)
    ns, nps = len(basis.T), len(pbasis.T)
    udofs = (2 * dofs[:, :, None] + np.arange(2)).reshape(len(fine.cells), 2 * ns)
    all_dofs = np.column_stack((udofs, 2 * nv + pdofs))
    strain, divergence, strong = _strain_and_divergence(gradients, hessian)
    size = 2 * nv + npres
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    flat = physical.reshape(-1, 2)
    mu = scalar_values(lame_mu, flat).reshape(len(fine.cells), len(bary))
    if np.any(mu <= 0):
        raise ValueError("Lame mu must be positive")
    epsilon = _compressibility(lame_lambda, flat).reshape(mu.shape)
    blocks = np.zeros((len(fine.cells), 2 * ns + nps, 2 * ns + nps))
    blocks[:, : 2 * ns, : 2 * ns] = np.einsum(
        "q,tqai,a,tqaj,tq,t->tij",
        weights,
        strain,
        [2.0, 2.0, 1.0],
        strain,
        mu,
        fine.areas,
    )
    mixed = -np.einsum("q,tqi,qj,t->tij", weights, divergence, pbasis, fine.areas)
    blocks[:, : 2 * ns, 2 * ns :] = mixed
    blocks[:, 2 * ns :, : 2 * ns] = mixed.swapaxes(1, 2)
    blocks[:, 2 * ns :, 2 * ns :] = -np.einsum(
        "q,qi,qj,tq,t->tij", weights, pbasis, pbasis, epsilon, fine.areas
    )
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    force = vector_values(source, physical.reshape(-1, 2)).reshape(len(fine.cells), -1, 2)
    element_load = np.zeros((len(fine.cells), 2 * ns + nps))
    element_load[:, : 2 * ns] = np.einsum(
        "q,qi,tqa,t->tia", weights, basis, force, fine.areas
    ).reshape(len(fine.cells), -1)
    alpha = 0.0
    if formulation == "gals":
        h = np.max(fine.lengths[fine.cell_faces], axis=1)
        shear_gradient = vector_values(lame_mu_gradient, flat).reshape(*mu.shape, 2)
        if shear_bounds is None:
            lower, upper, gradient_bound = float(mu.min()), float(mu.max()), 0.0
        else:
            lower, upper, gradient_bound = shear_bounds
            if (
                np.any(mu < lower * (1 - 1e-12))
                or np.any(mu > upper * (1 + 1e-12))
                or np.any(np.linalg.norm(shear_gradient, axis=-1) > gradient_bound * (1 + 1e-12))
            ):
                raise ValueError("shear_bounds do not bound the evaluated material coefficients")
        bound = (
            strain_inverse_bound(strain, strong, weights, h, diameter)
            * lower
            / (2 * (upper**2 + diameter**2 * gradient_bound**2))
        )
        alpha = bound / 2 if stabilization_alpha is None else stabilization_alpha
        if not np.isfinite(alpha) or not 0 < alpha < bound:
            raise ValueError(f"stabilization_alpha must lie strictly between 0 and {bound:g}")
        stress_divergence = 2 * mu[:, :, None, None] * strong
        stress_divergence[:, :, 0] += (
            2 * strain[:, :, 0] * shear_gradient[:, :, 0, None]
            + strain[:, :, 2] * shear_gradient[:, :, 1, None]
        )
        stress_divergence[:, :, 1] += (
            strain[:, :, 2] * shear_gradient[:, :, 0, None]
            + 2 * strain[:, :, 1] * shear_gradient[:, :, 1, None]
        )
        residual = np.concatenate((stress_divergence, -pgradients.swapaxes(-1, -2)), axis=3)
        blocks -= np.einsum(
            "q,tqai,tqaj,t->tij", weights, residual, residual, alpha * h**2 * fine.areas
        )
        element_load += np.einsum(
            "q,tqai,tqa,t->ti", weights, residual, force, alpha * h**2 * fine.areas
        )
    matrix = _assemble_blocks(blocks, all_dofs, size)
    load = np.bincount(all_dofs.ravel(), weights=element_load.ravel(), minlength=size)
    kernel = np.zeros((size, 3))
    kernel[: 2 * nv] = _rigid(nodes, rigid_center).reshape(2 * nv, 3)
    local_moments = np.einsum(
        "q,qi,tqaj,t->tiaj",
        weights,
        basis,
        _rigid(physical.reshape(-1, 2), rigid_center).reshape(*physical.shape, 3),
        fine.areas,
    ).reshape(len(fine.cells), 2 * ns, 3)
    constraints = np.zeros_like(kernel)
    np.add.at(constraints, udofs, local_moments)
    pressure_weights = np.zeros(size)
    np.add.at(pressure_weights, 2 * nv + pdofs, fine.areas[:, None] * (weights @ pbasis))
    compliance_weights = np.zeros(size)
    np.add.at(
        compliance_weights,
        2 * nv + pdofs,
        np.einsum("q,qi,tq,t->ti", weights, pbasis, epsilon, fine.areas),
    )
    return ElasticityPressureOperators(
        matrix,
        load,
        nodes,
        pnodes,
        kernel,
        constraints,
        pressure_weights,
        compliance_weights,
        float(epsilon.max()),
        alpha,
    )
