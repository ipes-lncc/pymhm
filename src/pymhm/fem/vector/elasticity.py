"""Engineering-strain and rigid-motion tabulation for planar elasticity.

Nodal vector components are interleaved. The engineering shear is du_x/dy+
 du_y/dx; its elastic Gram weight is one half relative to normal strains.
"""

import numpy as np

from pymhm.core.validation import FloatArray


def rigid_modes(points: FloatArray, center: FloatArray) -> FloatArray:
    """Evaluate both translations and the infinitesimal counterclockwise rotation."""
    modes = np.zeros((len(points), 2, 3))
    modes[:, :, :2] = np.eye(2)
    modes[:, 0, 2] = -(points[:, 1] - center[1])
    modes[:, 1, 2] = points[:, 0] - center[0]
    return modes


def strain_and_divergence(
    gradients: FloatArray, hessian: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tabulate engineering strain, vector divergence and divergence of symmetric strain."""
    nt, nq, ns, _ = gradients.shape
    strain = np.zeros((nt, nq, 3, 2 * ns))
    strain[:, :, 0, 0::2] = gradients[:, :, :, 0]
    strain[:, :, 1, 1::2] = gradients[:, :, :, 1]
    strain[:, :, 2, 0::2] = gradients[:, :, :, 1]
    strain[:, :, 2, 1::2] = gradients[:, :, :, 0]
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)
    strong = np.empty((nt, nq, 2, 2 * ns))
    for a in range(2):
        for c in range(2):
            strong[:, :, a, c::2] = (hessian[:, :, :, a, c] + (a == c) * laplacian) / 2
    return strain, gradients.reshape(nt, nq, 2 * ns), strong
