"""Isotropic displacement-pressure elasticity forms on affine tetrahedra.

Pressure is ``p=-lambda*div(u)``. The GaLS residual is the full divergence
of ``2*mu*sym(grad(u))-p*I``, including derivatives of a variable shear modulus.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray, positive_int
from pymhm.core.validation import real_array as _real
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D as _KELVIN3
from pymhm.fem.vector.elasticity_3d import rigid_modes_3d
from pymhm.fem.vector.inequalities_3d import strain_inverse_bound_3d as _strain_inverse_bound
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.compliance import compressibility_values as _compressibility
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


def elasticity_contract_3d(
    lame_lambda: Any,
    lame_mu: Any,
    lame_mu_gradient: Any,
    shear_bounds: Any,
    formulation: str,
    degree: int,
    stabilization_alpha: Any,
    points: FloatArray,
) -> tuple[Any, tuple[float, float, float] | None]:
    """Validate the finite/infinite compressibility and residual-coefficient contracts."""
    k = positive_int(degree, "degree")
    if (
        formulation not in ("gals", "taylor-hood")
        or k > 4
        or (formulation == "taylor-hood" and k < 2)
    ):
        raise ValueError("use GaLS degree 1--4 or Taylor-Hood degree 2--4")
    _compressibility(lame_lambda, points)
    if np.any(scalar_values_3d(lame_mu, points) <= 0):
        raise ValueError("Lame mu must be positive")
    if formulation != "gals" and stabilization_alpha is not None:
        raise ValueError("stabilization_alpha applies only to GaLS")
    if stabilization_alpha is not None:
        alpha = _real(stabilization_alpha, "stabilization_alpha")
        if alpha.ndim != 0 or alpha <= 0:
            raise ValueError("stabilization_alpha must be a positive scalar")
    if (
        callable(lame_mu)
        and formulation == "gals"
        and (lame_mu_gradient is None or shear_bounds is None)
    ):
        raise ValueError("variable shear in GaLS requires its gradient and shear_bounds")
    gradient = (0.0, 0.0, 0.0) if lame_mu_gradient is None else lame_mu_gradient
    if not callable(lame_mu) and np.any(vector_values_3d(gradient, points)):
        raise ValueError("a constant Lame mu must have zero gradient")
    bounds = None
    if shear_bounds is not None:
        values = _real(shear_bounds, "shear_bounds")
        if values.shape != (3,) or not 0 < values[0] <= values[1] or values[2] < 0:
            raise ValueError("shear_bounds requires positive lower, upper, nonnegative gradient")
        bounds = (float(values[0]), float(values[1]), float(values[2]))
    return gradient, bounds


@dataclass(frozen=True)
class ElasticityPressure3DOperators:
    """Mixed local operator, rigid modes and physical pressure/displacement moments."""

    matrix: sparse.csc_matrix
    load: FloatArray
    displacement_nodes: FloatArray
    pressure_nodes: FloatArray
    kernel: FloatArray
    rigid_moments: FloatArray
    pressure_moments: FloatArray
    compliance_moments: FloatArray
    compliance_scale: float
    stabilization_alpha: float
    stabilization_bound: float


def tetra_elasticity_pressure_operators(
    mesh: TetraMesh,
    *,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    lame_mu_gradient: Any = None,
    shear_bounds: tuple[float, float, float] | None = None,
    source: Any = (0.0, 0.0, 0.0),
    degree: int = 1,
    formulation: str = "gals",
    stabilization_alpha: float | None = None,
    macro_diameter: float | None = None,
    rigid_center: Any = (0.0, 0.0, 0.0),
    order: int = 6,
) -> ElasticityPressure3DOperators:
    """Assemble tetrahedral GaLS Pk/Pk or Taylor-Hood Pk/P(k-1) elasticity.

    The GaLS parameter is half the computed upper bound unless supplied:
    ``alpha < C_I*mu_lower/[2*(mu_upper**2+h_K**2*grad_mu_bound**2)]``.
    The inverse constant uses the physical tetrahedral strain/divergence forms,
    not a constant transferred from triangles. All six rigid modes are retained.
    Bounds for variable shear must be certified by the caller and are additionally
    checked at the integration points. Discontinuous materials require fitted
    tetrahedra; a callback does not automatically add interface cuts.
    """
    gradient_data, bounds = elasticity_contract_3d(
        lame_lambda,
        lame_mu,
        lame_mu_gradient,
        shear_bounds,
        formulation,
        degree,
        stabilization_alpha,
        mesh.points,
    )
    bary, weights = tetrahedron_quadrature(max(positive_int(order, "order"), degree + 2))
    dofs, nodes, basis, gradient, hessian = tetra_element_tabulate(mesh, degree, bary)
    pdegree = degree if formulation == "gals" else degree - 1
    pdofs, pnodes, pbasis, pgradient = tetra_tabulate(mesh, pdegree, bary)
    nv, ns, npres, nps = len(nodes), basis.shape[1], len(pnodes), pbasis.shape[1]
    udofs = (3 * dofs[:, :, None] + np.arange(3)).reshape(len(mesh.cells), 3 * ns)
    all_dofs = np.column_stack((udofs, 3 * nv + pdofs))
    points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
    flat = points.reshape(-1, 3)
    mu = scalar_values_3d(lame_mu, flat).reshape(points.shape[:2])
    if np.any(mu <= 0):
        raise ValueError("Lame mu must be positive at every integration point")
    compliance = _compressibility(lame_lambda, flat).reshape(mu.shape)
    strain = np.einsum("aij,tqnj->tqani", _KELVIN3, gradient).reshape(*mu.shape, 6, 3 * ns)
    divergence = gradient.reshape(*mu.shape, 3 * ns)
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)
    strong = np.empty((*mu.shape, 3, 3 * ns))
    for a in range(3):
        for component in range(3):
            strong[:, :, a, component::3] = (
                hessian[:, :, :, a, component] + (a == component) * laplacian
            ) / 2
    blocks = np.zeros((len(mesh.cells), 3 * ns + nps, 3 * ns + nps))
    blocks[:, : 3 * ns, : 3 * ns] = np.einsum(
        "t,q,tqai,tqaj,tq->tij", mesh.volumes, weights, strain, strain, 2 * mu
    )
    mixed = -np.einsum("t,q,tqi,qj->tij", mesh.volumes, weights, divergence, pbasis)
    blocks[:, : 3 * ns, 3 * ns :] = mixed
    blocks[:, 3 * ns :, : 3 * ns] = mixed.swapaxes(1, 2)
    blocks[:, 3 * ns :, 3 * ns :] = -np.einsum(
        "t,q,qi,qj,tq->tij", mesh.volumes, weights, pbasis, pbasis, compliance
    )
    force = vector_values_3d(source, flat).reshape(points.shape)
    element_load = np.zeros((len(mesh.cells), 3 * ns + nps))
    element_load[:, : 3 * ns] = np.einsum(
        "t,q,qi,tqa->tia", mesh.volumes, weights, basis, force
    ).reshape(len(mesh.cells), 3 * ns)
    alpha, bound = 0.0, 0.0
    if formulation == "gals":
        vertices = mesh.points[mesh.cells]
        lengths = np.max(
            np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=-1), axis=(1, 2)
        )
        diameter = float(lengths.max()) if macro_diameter is None else float(macro_diameter)
        if not np.isfinite(diameter) or diameter < float(lengths.max()) * (1 - 1e-12):
            raise ValueError("macro_diameter must bound every physical tetrahedron diameter")
        shear_gradient = vector_values_3d(gradient_data, flat).reshape(points.shape)
        lower, upper, grad_bound = (
            (float(mu.min()), float(mu.max()), 0.0) if bounds is None else bounds
        )
        if (
            np.any(mu < lower * (1 - 1e-12))
            or np.any(mu > upper * (1 + 1e-12))
            or np.any(np.linalg.norm(shear_gradient, axis=-1) > grad_bound * (1 + 1e-12))
        ):
            raise ValueError("shear_bounds do not bound the evaluated coefficients")
        bound = (
            _strain_inverse_bound(strain, strong, weights, lengths, diameter)
            * lower
            / (2 * (upper**2 + diameter**2 * grad_bound**2))
        )
        alpha = bound / 2 if stabilization_alpha is None else float(stabilization_alpha)
        if not alpha < bound:
            raise ValueError(f"stabilization_alpha must be strictly below {bound:g}")
        stress_divergence = 2 * mu[:, :, None, None] * strong + 2 * np.einsum(
            "aij,tqan,tqj->tqin", _KELVIN3, strain, shear_gradient
        )
        residual = np.concatenate((stress_divergence, -pgradient.swapaxes(-1, -2)), axis=-1)
        blocks -= np.einsum(
            "t,q,tqai,tqaj->tij", alpha * lengths**2 * mesh.volumes, weights, residual, residual
        )
        element_load += np.einsum(
            "t,q,tqai,tqa->ti", alpha * lengths**2 * mesh.volumes, weights, residual, force
        )
    size = 3 * nv + npres
    matrix = _assemble_blocks(blocks, all_dofs, size)
    load = np.bincount(all_dofs.ravel(), weights=element_load.ravel(), minlength=size)
    center = vector_values_3d(rigid_center, np.zeros((1, 3)))[0]
    rigid = rigid_modes_3d(nodes, center).reshape(3 * nv, 6)
    mass = _assemble_blocks(
        np.einsum("t,q,qi,qj->tij", mesh.volumes, weights, basis, basis), dofs, nv
    )
    kernel, moments = np.zeros((size, 6)), np.zeros((size, 6))
    kernel[: 3 * nv] = rigid
    moments[: 3 * nv] = sparse.kron(mass, sparse.eye(3)) @ rigid
    pressure_moments, compliance_moments = np.zeros(size), np.zeros(size)
    np.add.at(pressure_moments, 3 * nv + pdofs, mesh.volumes[:, None] * (weights @ pbasis))
    np.add.at(
        compliance_moments,
        3 * nv + pdofs,
        np.einsum("t,q,qi,tq->ti", mesh.volumes, weights, pbasis, compliance),
    )
    return ElasticityPressure3DOperators(
        matrix,
        load,
        nodes,
        pnodes,
        kernel,
        moments,
        pressure_moments,
        compliance_moments,
        float(compliance.max()),
        alpha,
        bound,
    )
