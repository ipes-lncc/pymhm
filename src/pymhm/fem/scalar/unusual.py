"""Coefficient and inverse-inequality contracts for scalar UNUSUAL stabilization."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from pymhm.core.validation import FloatArray
from pymhm.materials.evaluation import scalar_values


@dataclass(frozen=True)
class UnusualParameters:
    """Declare elementwise bounds for the reaction--diffusion residual method.

    ``diffusion_lower`` bounds the smallest eigenvalue of A from below and
    ``reaction_upper`` bounds the nonnegative reaction from above throughout
    each fine cell. Scalars are global bounds; callables receive fine-cell
    centroids and must return bounds valid on the *whole* corresponding cell.
    Smooth coefficient callbacks require these declarations. Constant and
    resolved Cartesian materials admit bounds obtained directly from their data.

    ``inverse_constant`` optionally prescribes m in Eq. (15) of Santiago,
    Valentin and Martins (2025). It must satisfy 0<m<=1/3 and the discrete
    elementwise coercivity check in :func:`unusual_scale`. Without an explicit
    value, a conservative operator-inverse value up to 1/3 is computed. Thus constant-A
    P1 uses exactly the published m=1/3. Higher-degree and variable-A constants
    depend on the physical element and operator, not only the degree.
    """

    diffusion_lower: Any = None
    reaction_upper: Any = None
    inverse_constant: float | None = None

    def __post_init__(self) -> None:
        """Reject inverse parameters outside the stated admissible interval."""
        value = self.inverse_constant
        if value is not None and (not np.isfinite(value) or not 0 < value <= 1 / 3):
            raise ValueError("UNUSUAL inverse_constant must lie in (0, 1/3]")


def unusual_scale(
    basis: FloatArray,
    gradient: FloatArray,
    diffusion_residual: FloatArray,
    weights: FloatArray,
    diameters: FloatArray,
    tensor: FloatArray,
    reaction: FloatArray,
    centers: FloatArray,
    parameters: UnusualParameters,
) -> FloatArray:
    """Compute elementwise Eq. (15), including its zero-reaction limit.

    Arrays have leading axes (cell, quadrature point); gradients additionally
    have (basis, dimension), and diffusion_residual is div(A grad(phi)). Bounds
    are checked at quadrature points. The automatic inverse parameter obeys
    m*h²*||div(A grad(v))||² <= A_min²*||grad(v)||² on the local polynomial space,
    with both norms defined by the supplied positive integration rule. Adequate
    quadrature and valid bounds between samples remain caller obligations.
    An explicit m is accepted only when the negative residual is strictly
    smaller than the physical diffusion--reaction energy for every polynomial
    on each cell (modulo constants when reaction vanishes). This generalized
    eigenvalue check admits stable explicit choices beyond the conservative
    automatic inverse bound without claiming those bounds are identical.

    Eq. (15) is evaluated as m*h²/[max(sigma_max*m*h²,2*A_min)+2*A_min],
    avoiding division by zero when sigma vanishes. This is the continuous
    diffusion limit of that equation, with no artificial reaction floor.
    """
    lower_sample = np.linalg.eigvalsh(tensor)[..., 0].min(axis=1)
    upper_sample = reaction.max(axis=1)
    lower = (
        lower_sample
        if parameters.diffusion_lower is None
        else scalar_values(parameters.diffusion_lower, centers)
    )
    upper = (
        upper_sample
        if parameters.reaction_upper is None
        else scalar_values(parameters.reaction_upper, centers)
    )
    tolerance = 128 * np.finfo(float).eps
    if np.any(lower <= 0) or np.any(lower > lower_sample * (1 + tolerance)):
        raise ValueError(
            "UNUSUAL diffusion_lower must be positive and bound all sampled eigenvalues"
        )
    if np.any(upper < 0) or np.any(upper * (1 + tolerance) < upper_sample):
        raise ValueError("UNUSUAL reaction_upper must bound all sampled nonnegative reactions")
    energy = np.einsum("tq,tqia,tqja->tij", weights, gradient, gradient)
    residual = np.einsum("tq,tqi,tqj->tij", weights, diffusion_residual, diffusion_residual)
    # Removing one nodal basis eliminates the sole constant-gradient null mode.
    constants = np.full(len(gradient), 1 / 3)
    for cell, (stiffness, strong) in enumerate(zip(energy, residual, strict=True)):
        maximum = max(0.0, float(linalg.eigvalsh(strong[1:, 1:], stiffness[1:, 1:])[-1]))
        if maximum > 0:
            constants[cell] = min(1 / 3, lower[cell] ** 2 / (diameters[cell] ** 2 * maximum))
    if parameters.inverse_constant is not None:
        constants[:] = parameters.inverse_constant
    numerator = constants * diameters**2
    tau = numerator / (np.maximum(upper * numerator, 2 * lower) + 2 * lower)
    if parameters.inverse_constant is not None:
        # Represent constants exactly instead of summing derivative roundoff.
        values = basis.copy()
        derivatives = gradient.copy()
        diffusion = diffusion_residual.copy()
        values[:, :, 0] = 1.0
        derivatives[:, :, 0] = 0.0
        diffusion[:, :, 0] = 0.0
        physical = np.einsum("tq,tqia,tqab,tqjb->tij", weights, derivatives, tensor, derivatives)
        physical += np.einsum("tq,tq,tqi,tqj->tij", weights, reaction, values, values)
        strong = reaction[:, :, None] * values - diffusion
        negative = np.einsum("tq,tqi,tqj->tij", weights, strong, strong)
        for cell in range(len(tau)):
            start = 0 if np.any(reaction[cell]) else 1
            energy_cell = physical[cell, start:, start:]
            residual_cell = negative[cell, start:, start:]
            scale = 1 / np.sqrt(np.diag(energy_cell))
            eigenvalue = linalg.eigvalsh(
                residual_cell * scale[:, None] * scale[None, :],
                energy_cell * scale[:, None] * scale[None, :],
            )[-1]
            if tau[cell] * eigenvalue >= 1 - tolerance:
                raise ValueError("UNUSUAL inverse_constant violates the element coercivity bound")
    return tau
