"""Physical strain inverse constants on affine tetrahedra modulo rigid motions."""

import numpy as np
from scipy import linalg

from pymhm.core.validation import FloatArray


def strain_inverse_bound_3d(
    strain: FloatArray,
    strong: FloatArray,
    weights: FloatArray,
    lengths: FloatArray,
    diameter: float,
) -> float:
    """Compute a safe Eq. 4.4 constant on physical tetrahedra modulo six rigid motions."""
    energy = np.einsum("q,tqai,tqaj->tij", weights, strain, strain)
    residual = np.einsum("q,tqai,tqaj->tij", weights, strong, strong)
    maximum = 1.0
    for mass, second, length in zip(energy, residual, lengths, strict=True):
        eigenvalues, vectors = linalg.eigh(mass)
        selected = eigenvalues > 1e-11 * eigenvalues[-1]
        if np.count_nonzero(selected) != len(eigenvalues) - 6:
            raise ValueError("inverse inequality requires exactly 6 resolved kernel modes")
        whitening = vectors[:, selected] / np.sqrt(eigenvalues[selected])
        operator = length**2 * (mass / diameter**2 + second)
        maximum = max(maximum, float(linalg.eigvalsh(whitening.T @ operator @ whitening)[-1]))
    return 1 / maximum
