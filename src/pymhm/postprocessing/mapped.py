"""Physical mapped-RT field records in the literal executed reference coordinates."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray
from pymhm.fem.hdiv.mapped import mapped_rt_basis, mapped_rt_dofs, tensor_legendre_values
from pymhm.fem.hdiv.mapped_forms import HexSkeleton, quadrature_slices
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.hexahedron import HexMesh, QuadratureOrder, cube_quadrature


@dataclass(frozen=True)
class MappedRTDarcySolution:
    """Mapped H(div) flux, discontinuous scalar-pullback pressure and condensed macro traces."""

    skeleton: HexSkeleton
    local_meshes: tuple[HexMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    permeability: Any
    source: Any
    quadrature_order: QuadratureOrder
    physical_residuals: FloatArray

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate pressure, physical flux and exact Piola divergence in every local hexahedron."""
        mesh = self.local_meshes[cell]
        basis, div, p = mapped_rt_basis(mesh, self.degree, points)
        coefficients = self.flux[cell][mapped_rt_dofs(mesh, self.degree)]
        return (
            self.pressure[cell] @ p.T,
            np.einsum("tqia,ti->tqa", basis, coefficients),
            np.einsum("tqi,ti->tq", div, coefficients),
        )

    def errors(self, pressure: Any, flux: Any, order: QuadratureOrder = 6) -> dict[str, float]:
        """Integrate physical pressure/vector-flux L2 errors with independent quadrature."""
        points, weights = cube_quadrature(order)
        errors = np.zeros(2)
        for cell, mesh in enumerate(self.local_meshes):
            for part in quadrature_slices(mesh, self.degree, len(points)):
                physical, _, det = mesh.geometry(points[part])
                p, q, _ = self.evaluate(cell, points[part])
                pe = scalar_values_3d(pressure, physical.reshape(-1, 3)).reshape(det.shape)
                qe = vector_values_3d(flux, physical.reshape(-1, 3)).reshape(*det.shape, 3)
                errors += [
                    np.sum(det * weights[part] * (p - pe) ** 2),
                    np.sum(det * weights[part] * np.sum((q - qe) ** 2, axis=2)),
                ]
        return dict(pressure_l2=float(np.sqrt(errors[0])), flux_l2=float(np.sqrt(errors[1])))

    def equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return every pressure-tested physical divergence/source defect, including cell mass."""
        points, weights = cube_quadrature(self.quadrature_order)
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            moments = np.zeros((len(mesh.cells), (self.degree + 1) ** 3))
            for part in quadrature_slices(mesh, self.degree, len(points)):
                physical, _, det = mesh.geometry(points[part])
                divergence = self.evaluate(cell, points[part])[2]
                f = scalar_values_3d(self.source, physical.reshape(-1, 3)).reshape(det.shape)
                p = tensor_legendre_values(self.degree, points[part])
                moments += np.einsum("tq,qi,tq->ti", det * weights[part], p, divergence - f)
            result.append(moments)
        return tuple(result)
