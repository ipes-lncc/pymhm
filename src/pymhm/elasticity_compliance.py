"""Anisotropic compliance on full, weakly symmetric two-dimensional stresses.

Cartesian coordinates are row-major ``(xx, xy, yx, yy)``. A compliance must
preserve symmetric tensors and be positive definite also on skew tensors;
the latter is the explicit extension needed by weak-symmetry mixed methods.
"""

from typing import Any

import numpy as np

from pymhm.mesh import FloatArray


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
