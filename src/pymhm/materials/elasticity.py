"""Finite elastic constitutive tensors in orthonormal Kelvin coordinates.

These material evaluations are shared by static and dynamic vector forms and
field reconstruction. Positivity is checked at supplied points; callbacks still
require their own global material-bound premises.
"""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D
from pymhm.materials.evaluation import scalar_values, scalar_values_3d


def constitutive_values(
    material: Any, points: FloatArray, *, lame_lambda: Any = 1.0, lame_mu: Any = 1.0
) -> FloatArray:
    """Evaluate and validate stiffness in orthonormal Kelvin coordinates.

    Constant or callable matrices may have shape (3,3) or (2,2,2,2), optionally
    preceded by the point axis. Positivity is checked at supplied points;
    quadrature sampling does not certify uniform ellipticity everywhere.
    With material=None, finite nonnegative lambda and positive mu define the
    isotropic plane-strain tensor. A primal tensor with infinite bulk modulus
    is not supported; use the mixed incompressible formulations instead.
    """
    if material is None:
        lam, mu = scalar_values(lame_lambda, points), scalar_values(lame_mu, points)
        if np.any(lam < 0) or np.any(mu <= 0):
            raise ValueError("Lame mu must be positive and lambda nonnegative")
        result = np.zeros((len(points), 3, 3))
        result[:, 0, 0] = result[:, 1, 1] = lam + 2 * mu
        result[:, 0, 1] = result[:, 1, 0] = lam
        result[:, 2, 2] = 2 * mu
        return result
    raw = material(points) if callable(material) else material
    if np.iscomplexobj(raw):
        raise ValueError("constitutive tensor must be real")
    values = np.asarray(raw, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("constitutive tensor must be finite")
    if values.shape[-4:] == (2, 2, 2, 2):
        values = np.broadcast_to(values, (len(points), 2, 2, 2, 2))
        scale = np.max(np.abs(values), axis=(1, 2, 3, 4))
        for transpose in (
            values.swapaxes(1, 2),
            values.swapaxes(3, 4),
            values.transpose(0, 3, 4, 1, 2),
        ):
            if np.any(
                np.max(np.abs(values - transpose), axis=(1, 2, 3, 4))
                > 64 * np.finfo(float).eps * scale
            ):
                raise ValueError("constitutive tensor requires major and minor symmetries")
        values = np.einsum("aij,nijkl,bkl->nab", KELVIN_BASIS_2D, values, KELVIN_BASIS_2D)
    else:
        if values.shape[-2:] != (3, 3):
            raise ValueError(
                "constitutive tensor requires Kelvin (3,3) or Cartesian (2,2,2,2) shape"
            )
        values = np.broadcast_to(values, (len(points), 3, 3))
    if np.any(
        np.max(np.abs(values - values.swapaxes(1, 2)), axis=(1, 2))
        > 64 * np.finfo(float).eps * np.max(np.abs(values), axis=(1, 2))
    ):
        raise ValueError("Kelvin constitutive matrix must be symmetric")
    if np.any(np.linalg.eigvalsh(values)[:, 0] <= 0):
        raise ValueError("constitutive tensor must be positive definite on symmetric strains")
    return values


KELVIN_BASIS_2D = np.array(
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0.0]],
    ]
)


def constitutive_values_3d(
    material: Any, points: FloatArray, *, lame_lambda: Any = 1.0, lame_mu: Any = 1.0
) -> FloatArray:
    """Validate general SPD Kelvin/Cartesian stiffness, or finite isotropic Lamé fields."""
    if material is None:
        lam, mu = scalar_values_3d(lame_lambda, points), scalar_values_3d(lame_mu, points)
        if np.any(lam < 0) or np.any(mu <= 0):
            raise ValueError("Lame lambda must be nonnegative and mu positive")
        identity = np.r_[np.ones(3), np.zeros(3)]
        return 2 * mu[:, None, None] * np.eye(6) + lam[:, None, None] * np.outer(identity, identity)
    raw = material(points) if callable(material) else material
    if np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("constitutive tensor must be finite and real")
    values = np.asarray(raw, dtype=float)
    if values.shape[-4:] == (3, 3, 3, 3):
        values = np.broadcast_to(values, (len(points), 3, 3, 3, 3))
        scale = np.max(np.abs(values), axis=(1, 2, 3, 4))
        for permutation in (
            values.swapaxes(1, 2),
            values.swapaxes(3, 4),
            values.transpose(0, 3, 4, 1, 2),
        ):
            if np.any(
                np.max(np.abs(values - permutation), axis=(1, 2, 3, 4))
                > 64 * np.finfo(float).eps * scale
            ):
                raise ValueError("constitutive tensor requires major and minor symmetries")
        values = np.einsum("aij,nijkl,bkl->nab", KELVIN_BASIS_3D, values, KELVIN_BASIS_3D)
    else:
        if values.shape[-2:] != (6, 6):
            raise ValueError(
                "constitutive tensor requires Kelvin (6,6) or Cartesian (3,3,3,3) shape"
            )
        values = np.broadcast_to(values, (len(points), 6, 6))
    if np.any(
        np.max(np.abs(values - values.swapaxes(1, 2)), axis=(1, 2))
        > 64 * np.finfo(float).eps * np.max(np.abs(values), axis=(1, 2))
    ):
        raise ValueError("constitutive Kelvin matrix must be symmetric")
    if np.any(np.linalg.eigvalsh(values)[:, 0] <= 0):
        raise ValueError("constitutive tensor must be positive definite on symmetric strains")
    return values


def stress_compliance_values(material: Any, points: FloatArray) -> FloatArray:
    """Evaluate a full Cartesian compliance, checking its sampled ellipticity.

    Accept a constant or callback returning ``(4,4)`` or ``(2,2,2,2)``,
    optionally preceded by the evaluation-point axis. The operator is real,
    self-adjoint, positive definite and commutes with matrix transposition.
    Its skew extension must be supplied explicitly. A symmetric-strain Kelvin
    matrix alone does not define this full-tensor operator. Sampling does not
    prove a coefficient's uniform ellipticity between quadrature points.
    """
    raw = material(points) if callable(material) else material
    if np.iscomplexobj(raw):
        raise ValueError("stress compliance must be real")
    values = np.asarray(raw, dtype=float)
    if values.shape[-4:] == (2, 2, 2, 2):
        values = values.reshape(*values.shape[:-4], 4, 4)
    if values.shape[-2:] != (4, 4):
        raise ValueError("stress compliance requires Cartesian (4,4) or (2,2,2,2) shape")
    values = np.broadcast_to(values, (len(points), 4, 4))
    if not np.isfinite(values).all():
        raise ValueError("stress compliance must be finite")
    tolerance = 64 * np.finfo(float).eps * np.max(np.abs(values), axis=(1, 2))
    if np.any(np.max(np.abs(values - values.swapaxes(1, 2)), axis=(1, 2)) > tolerance):
        raise ValueError("stress compliance must be self-adjoint")
    permutation = np.array([0, 2, 1, 3])
    if np.any(
        np.max(np.abs(values - values[:, permutation][:, :, permutation]), axis=(1, 2)) > tolerance
    ):
        raise ValueError("stress compliance must preserve symmetric and skew tensors")
    if np.any(np.linalg.eigvalsh(values)[:, 0] <= 0):
        raise ValueError("stress compliance must be positive definite on full tensors")
    return values


def compliance_products(
    material: Any, points: FloatArray, tensors: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    """Return basis compliance products, tr(A sigma) and a positive gauge scale.

    ``tensors`` has shape ``(cells, quadrature, basis, 2, 2)``. The physical
    displacement identity is integral(tr(A sigma)) = integral(g dot n),
    including anisotropic couplings between normal and shear stresses.
    """
    shape = tensors.shape[:2]
    compliance = stress_compliance_values(material, points.reshape(-1, 2)).reshape(*shape, 4, 4)
    basis = tensors.reshape(*tensors.shape[:3], 4)
    action = np.einsum("tqab,tqib->tqia", compliance, basis)
    products = np.einsum("tqia,tqja->tqij", action, basis)
    weighted_trace = action[..., 0] + action[..., 3]
    identity = np.array([1.0, 0.0, 0.0, 1.0])
    scale = float(np.max(np.einsum("a,tqab,b->tq", identity, compliance, identity)) / 2)
    return products, weighted_trace, scale
