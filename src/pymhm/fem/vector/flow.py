"""Physical velocity-pressure volume forms on explicitly supplied triangular meshes."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.inequalities import laplacian_inverse_bound
from pymhm.fem.quadrature.material import cartesian_triangle_quadrature
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import element_tabulate as _element_tabulation
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.materials.resistance import resistance_values as _resistance_values
from pymhm.meshes.triangle import TriangleMesh


def minimum_resistance(drag: Any, supplied: float | None) -> float:
    """Determine and validate an explicit global lower eigenvalue bound for 2017 USFEM."""
    if supplied is not None:
        value = np.asarray(supplied)
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.number)
            or np.iscomplexobj(value)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("gamma_min must be a finite nonnegative scalar")
        supplied = float(value)
    if isinstance(drag, CartesianCellField):
        values = drag.values.reshape((-1, *drag.values.shape[len(drag.spacing) :]))
        minimum = float(
            np.min(
                np.linalg.eigvalsh(_resistance_values(values, np.zeros((len(values), 2))))[..., 0]
            )
        )
    elif callable(drag):
        if supplied is None:
            raise ValueError("minimum-2017 with a variable callback requires an explicit gamma_min")
        return supplied
    else:
        minimum = float(
            np.min(np.linalg.eigvalsh(_resistance_values(drag, np.zeros((1, 2))))[..., 0])
        )
    if supplied is not None and supplied > minimum:
        raise ValueError("gamma_min exceeds the minimum eigenvalue of the material")
    return minimum if supplied is None else supplied


def advection_contract(
    advection: Any, divergence: Any, bound: float | None, *, stabilized: bool
) -> tuple[Any, Any, float | None]:
    """Validate exact divergence data and an optional certified velocity bound.

    A callable needs its analytical divergence, including explicit zero for a
    solenoidal field. Stabilized Oseen also needs a supplied global upper bound
    on its Euclidean magnitude; quadrature sampling cannot certify a supremum.
    """
    if callable(advection):
        if divergence is None:
            raise ValueError("callable advection requires advection_divergence")
        if stabilized and bound is None:
            raise ValueError("stabilized callable advection requires advection_bound")
    else:
        advection = vector_values(advection, np.zeros((1, 2)))[0]
        if divergence is not None and (callable(divergence) or np.any(np.asarray(divergence))):
            raise ValueError("constant advection has zero divergence")
        divergence = 0.0
        if bound is None:
            bound = float(np.linalg.norm(advection))
    if bound is not None:
        value = np.asarray(bound)
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.number)
            or np.iscomplexobj(value)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("advection_bound must be a finite nonnegative scalar")
        bound = float(value)
    return advection, divergence, bound


@dataclass(frozen=True)
class FlowOperators:
    """Velocity first, pressure second; literal volume operator and physical moments."""

    matrix: Any
    load: FloatArray
    velocity_nodes: FloatArray
    pressure_nodes: FloatArray
    kernel: FloatArray
    translation_moments: FloatArray
    pressure_moments: FloatArray
    resistance_moment: FloatArray
    absolute_resistance_moment: FloatArray
    zero_columns: Any
    pure: bool


def triangle_flow_operators(
    fine: TriangleMesh,
    *,
    degree: int,
    formulation: str,
    viscosity: float,
    drag: Any,
    beta: Any,
    source: Any,
    order: int,
    gamma_min: float | None = None,
    pointwise: bool = False,
    residual_weight: Any = None,
    beta_divergence: Any = 0.0,
    beta_bound: float | None = None,
) -> FlowOperators:
    """Integrate grad-grad flow and the declared residual test/trial operators.

    Coordinates are interleaved continuous Pk velocity then continuous pressure,
    P(k-1) for Taylor--Hood or Pk for USFEM/Oseen. Pressure equations have negative
    divergence. Skew advection includes -div(beta)/2; Oseen residuals use opposite
    test/trial convection and positive grad-div. Cartesian material cut integration
    preserves physical resistance data; inverse bounds use independent Gaussian
    polynomial integration. No trace, boundary, kernel elimination or solve is chosen.
    """
    gauss_bary, gauss_weights = triangle_quadrature(max(order, degree + 2))
    uniform = tabulate(fine, degree, gauss_bary)
    if isinstance(drag, CartesianCellField):
        bary, weights, pixels = cartesian_triangle_quadrature(
            fine, drag, max(order, degree + 2), return_cells=True
        )
        udofs, nodes, basis, gradient, hessian = _element_tabulation(fine, degree, bary)
        material = drag.values[tuple(pixels.reshape(-1, 2).T)]
    else:
        bary = np.broadcast_to(gauss_bary, (len(fine.cells), *gauss_bary.shape))
        weights = np.broadcast_to(gauss_weights, bary.shape[:2])
        udofs, nodes, scalar_basis, gradient, hessian = uniform
        basis = np.broadcast_to(scalar_basis, (*bary.shape[:2], scalar_basis.shape[1]))
        material = drag
    pdegree = degree - 1 if formulation == "taylor-hood" else degree
    if pdegree == degree:
        pdofs, pnodes, pbasis, pgradient = udofs, nodes, basis, gradient
    else:
        pdofs, pnodes, pbasis, pgradient, _ = _element_tabulation(fine, pdegree, bary)
    ns, ps = basis.shape[-1], pbasis.shape[-1]
    nv, npres = len(nodes), len(pnodes)
    vdofs = (2 * udofs[:, :, None] + np.arange(2)).reshape(len(fine.cells), -1)
    dofs = np.column_stack((vdofs, 2 * nv + pdofs))
    physical = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    resistance = _resistance_values(material, physical.reshape(-1, 2)).reshape(
        *bary.shape[:2], 2, 2
    )
    velocity = vector_values(beta, physical.reshape(-1, 2)).reshape(*bary.shape[:2], 2)
    divergence_beta = scalar_values(beta_divergence, physical.reshape(-1, 2)).reshape(
        bary.shape[:2]
    )
    if beta_bound is not None and np.any(
        np.linalg.norm(velocity, axis=-1) > beta_bound * (1 + 32 * np.finfo(float).eps)
    ):
        raise ValueError("advection_bound is smaller than a sampled advection magnitude")
    stiffness = np.einsum("tq,tqia,tqja,t->tij", weights, gradient, gradient, fine.areas)
    advective = np.einsum("tq,tqi,tqa,tqja,t->tij", weights, basis, velocity, gradient, fine.areas)
    scalar = viscosity * stiffness + (advective - advective.swapaxes(1, 2)) / 2
    scalar -= np.einsum(
        "tq,tq,tqi,tqj,t->tij", weights, divergence_beta / 2, basis, basis, fine.areas
    )
    blocks = np.zeros((len(fine.cells), 2 * ns + ps, 2 * ns + ps))
    blocks[:, : 2 * ns, : 2 * ns] = np.einsum("tij,ab->tiajb", scalar, np.eye(2)).reshape(
        len(fine.cells), 2 * ns, 2 * ns
    )
    blocks[:, : 2 * ns, : 2 * ns] += np.einsum(
        "tq,tqi,tqj,tqab,t->tiajb", weights, basis, basis, resistance, fine.areas
    ).reshape(len(fine.cells), 2 * ns, 2 * ns)
    divergence = gradient.reshape(*bary.shape[:2], 2 * ns)
    cross = -np.einsum("tq,tqi,tqj,t->tij", weights, divergence, pbasis, fine.areas)
    blocks[:, : 2 * ns, 2 * ns :] = cross
    blocks[:, 2 * ns :, : 2 * ns] = cross.swapaxes(1, 2)
    force = vector_values(source, physical.reshape(-1, 2)).reshape(*bary.shape[:2], 2)
    element_load = np.zeros((len(fine.cells), 2 * ns + ps))
    element_load[:, : 2 * ns] = np.einsum(
        "tq,tqi,tqa,t->tia", weights, basis, force, fine.areas
    ).reshape(len(fine.cells), -1)
    if formulation in ("usfem", "oseen"):
        diameters = np.max(fine.lengths[fine.cell_faces], axis=1)
        # This geometric inverse constant integrates only polynomials. Material
        # interfaces must not affect its independently exact Gaussian evaluation.
        inverse = laplacian_inverse_bound(uniform[3], uniform[4], gauss_weights, diameters)
        viscous_scale = 4 * viscosity / inverse
        eigenvalues = np.linalg.eigvalsh(resistance)
        if formulation == "oseen":
            # Eq. (26), written without divisions by gamma or ||beta|| so
            # the published gamma=0 internal-layer example is also defined.
            bound = cast(float, beta_bound)
            advective_scale = bound * diameters
            gamma = resistance[0, 0, 0, 0]
            tau = (
                diameters**2
                / (
                    np.maximum(gamma * diameters**2, viscous_scale)
                    + np.maximum(viscous_scale, advective_scale)
                )
            )[:, None]
            grad_div = advective_scale * np.minimum(1.0, advective_scale / viscous_scale)
            blocks[:, : 2 * ns, : 2 * ns] += np.einsum(
                "tq,t,tqi,tqj,t->tij", weights, grad_div, divergence, divergence, fine.areas
            )
        elif pointwise:
            resistance_bound = eigenvalues[..., 0]
        elif gamma_min is None:
            resistance_bound = np.max(eigenvalues[..., -1], axis=1)[:, None]
        else:
            if np.any(eigenvalues[..., 0] < gamma_min):
                raise ValueError("gamma_min exceeds a sampled material eigenvalue")
            resistance_bound = np.full((len(fine.cells), 1), gamma_min)
        if formulation == "usfem":
            tau = diameters[:, None] ** 2 / (
                np.maximum(resistance_bound * diameters[:, None] ** 2, viscous_scale[:, None])
                + viscous_scale[:, None]
            )
        if residual_weight is not None:
            tau = np.asarray(
                residual_weight(
                    fine,
                    viscosity,
                    drag,
                    tabulation=(uniform[3], uniform[4], gauss_weights),
                    resistance=resistance,
                ),
                dtype=float,
            )
            if tau.shape == (len(fine.cells),):
                tau = tau[:, None]
            if (
                tau.shape not in ((len(fine.cells), 1), weights.shape)
                or not np.isfinite(tau).all()
                or np.any(tau <= 0)
            ):
                raise ValueError(
                    "residual weight must be positive on every fine cell/quadrature point"
                )
        laplacian = -viscosity * np.trace(hessian, axis1=-2, axis2=-1)
        residual = np.zeros((*bary.shape[:2], 2, 2 * ns + ps))
        for a in range(2):
            for c in range(2):
                residual[:, :, a, c : 2 * ns : 2] = (
                    resistance[:, :, a, c, None] * basis + (a == c) * laplacian
                )
        residual[:, :, :, 2 * ns :] = pgradient.swapaxes(-1, -2)
        trial_residual = residual.copy()
        if formulation == "oseen":
            convection = np.einsum("tqa,tqia->tqi", velocity, gradient)
            for component in range(2):
                trial_residual[:, :, component, component : 2 * ns : 2] += convection
                residual[:, :, component, component : 2 * ns : 2] -= convection
        blocks -= np.einsum(
            "tq,tqai,tqaj,t->tij", weights * tau, residual, trial_residual, fine.areas
        )
        element_load -= np.einsum("tq,tqai,tqa,t->ti", weights * tau, residual, force, fine.areas)
    size = 2 * nv + npres
    matrix = _assemble_blocks(blocks, dofs, size)
    load = np.bincount(dofs.ravel(), weights=element_load.ravel(), minlength=size)
    translations = np.zeros((size, 2))
    translations[: 2 * nv] = np.tile(np.eye(2), (nv, 1))
    moments = np.zeros(nv)
    np.add.at(moments, udofs, fine.areas[:, None] * np.einsum("tq,tqi->ti", weights, basis))
    constraints = np.zeros_like(translations)
    constraints[: 2 * nv] = (moments[:, None, None] * np.eye(2)).reshape(2 * nv, 2)
    pressure_weights = np.zeros(size)
    np.add.at(
        pressure_weights,
        2 * nv + pdofs,
        fine.areas[:, None] * np.einsum("tq,tqi->ti", weights, pbasis),
    )
    resistance_moment = np.einsum("tq,tqab,t->ab", weights, resistance, fine.areas)
    absolute_moment = np.einsum("tq,tqab,t->ab", weights, np.abs(resistance), fine.areas)
    zero_columns = np.all(resistance == 0, axis=(0, 1, 2))
    pure = not np.any(resistance) and not np.any(velocity) and not np.any(divergence_beta)
    return FlowOperators(
        matrix,
        load,
        nodes,
        pnodes,
        translations,
        constraints,
        pressure_weights,
        resistance_moment,
        absolute_moment,
        zero_columns,
        pure,
    )
