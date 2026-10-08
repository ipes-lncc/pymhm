"""Physical positive residual scales for scalar finite-element forms."""

from typing import Any

import numpy as np

from pymhm.meshes.triangle import TriangleMesh


def streamline_scale(fine: TriangleMesh, tensor: Any, beta: Any, strong_reaction: Any) -> Any:
    """Return the common positive residual time scale for spatial and time-step forms."""
    h = np.max(fine.lengths[fine.cell_faces], axis=1)[:, None]
    magnitude = np.linalg.norm(beta, axis=-1)
    diffusivity = np.linalg.eigvalsh(tensor)[..., -1]
    return 1 / np.sqrt(
        (2 * magnitude / h) ** 2 + (4 * diffusivity / h**2) ** 2 + strong_reaction**2
    )
