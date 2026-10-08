"""Physical evaluation records for independently assembled conforming baselines."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.cartesian import CartesianMacroMesh


@dataclass(frozen=True)
class ConformingQuadrilateralSolution:
    """One global continuous Qk field, with physical evaluation at arbitrary points."""

    mesh: CartesianMacroMesh
    degree: int
    pressure: FloatArray
    permeability: Any
    residual: float

    def evaluate(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return pressure and gradient without interpolation or material averaging."""
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
            raise ValueError("evaluation points must be finite real XY pairs")
        points = np.asarray(raw, dtype=float)
        origin = self.mesh.points[0]
        coordinates = (points - origin) / self.mesh.spacing
        counts = np.array([self.mesh.nx, self.mesh.ny])
        tolerance = 64 * np.finfo(float).eps * counts
        if np.any(coordinates < -tolerance) or np.any(coordinates > counts + tolerance):
            raise ValueError("evaluation points lie outside the rectangular mesh")
        indices = np.minimum(np.maximum(coordinates.astype(int), 0), counts - 1)
        reference = np.clip(coordinates - indices, 0, 1)
        width = self.mesh.nx * self.degree + 1
        offsets = np.array(
            [j * width + i for j in range(self.degree + 1) for i in range(self.degree + 1)]
        )
        node_ids = (indices[:, 1] * width + indices[:, 0])[:, None] * self.degree + offsets
        basis, gradients = qk_basis(self.degree, reference)
        values = self.pressure[node_ids]
        return np.einsum("qi,qi->q", basis, values), np.einsum(
            "qi,qia->qa", values, gradients / self.mesh.spacing
        )

    def physical_flux(self, points: FloatArray) -> FloatArray:
        """Evaluate -K grad(p) using the coefficient's declared pointwise convention."""
        gradient = self.evaluate(points)[1]
        return -np.einsum("qab,qb->qa", tensor_values(self.permeability, points), gradient)
