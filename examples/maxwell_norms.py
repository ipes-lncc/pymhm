"""Cached physical quadrature for exact staggered Maxwell cavity error norms."""

from typing import Any

import numpy as np

from examples.maxwell_data import CavityMode
from pymhm.fem.vector.curl import physical_basis, physical_points, quadrature


class MaxwellNorms:
    """Cache geometry and analytical spatial modes, preserving independent DG sides."""

    def __init__(self, solution: Any, mode: CavityMode, order: int = 8) -> None:
        """Tabulate independent error quadrature once for a fixed spatial discretization."""
        self.mode = mode
        points, weights, bases, gradients = [], [], [], []
        for local in solution.locals:
            bary, weight, _ = quadrature(local.mesh, 1, order)
            basis, gradient = physical_basis(local.mesh, local.degree, bary)
            points.append(physical_points(local.mesh, bary))
            weights.append(weight)
            bases.append(basis)
            gradients.append(gradient)
        self.points = np.concatenate(points)
        self.weights = np.concatenate(weights)
        self.basis, self.gradient = np.concatenate(bases), np.concatenate(gradients)
        self.electric_shape = mode.electric_shape(self.points.reshape(-1, mode.dimension)).reshape(
            *self.weights.shape, -1
        )
        self.magnetic_shape = mode.magnetic_shape(self.points.reshape(-1, mode.dimension)).reshape(
            *self.weights.shape, mode.dimension
        )

    def measure(self, solution: Any) -> dict[str, float]:
        """Integrate actual field/curl errors at E's and H's respective time coordinates."""
        dimension = self.mode.dimension
        e = np.concatenate(solution.electric).reshape(
            len(self.weights), -1, 1 if dimension == 2 else 3
        )
        h = np.concatenate(solution.magnetic).reshape(len(self.weights), -1, dimension)
        electric = np.einsum("tqi,tia->tqa", self.basis, e)
        magnetic = np.einsum("tqi,tia->tqa", self.basis, h)
        de = np.einsum("tqia,tic->tqca", self.gradient, e)
        dh = np.einsum("tqia,tic->tqca", self.gradient, h)
        if dimension == 2:
            curl_e = np.stack((de[..., 0, 1], -de[..., 0, 0]), axis=-1)
            curl_h = (dh[..., 1, 0] - dh[..., 0, 1])[..., None]
        else:
            curl_e = np.stack(
                (
                    de[..., 2, 1] - de[..., 1, 2],
                    de[..., 0, 2] - de[..., 2, 0],
                    de[..., 1, 0] - de[..., 0, 1],
                ),
                axis=-1,
            )
            curl_h = np.stack(
                (
                    dh[..., 2, 1] - dh[..., 1, 2],
                    dh[..., 0, 2] - dh[..., 2, 0],
                    dh[..., 1, 0] - dh[..., 0, 1],
                ),
                axis=-1,
            )
        cosine = np.cos(self.mode.omega * solution.electric_time)
        sine = np.sin(self.mode.omega * solution.magnetic_time)

        def norm(values: np.ndarray) -> float:
            """Integrate a physical vector square without subtracting large norm terms."""
            return float(np.sqrt(np.sum(self.weights * np.sum(values**2, axis=-1))))

        e_error = norm(electric - cosine * self.electric_shape)
        h_error = norm(magnetic - sine * self.magnetic_shape)
        ec_error = norm(curl_e + self.mode.omega * cosine * self.magnetic_shape)
        hc_error = norm(curl_h + self.mode.omega * sine * self.electric_shape)
        return {
            "electric_l2": e_error,
            "magnetic_l2": h_error,
            "electric_curl_l2": ec_error,
            "magnetic_curl_l2": hc_error,
            "combined_l2": float(np.hypot(e_error, h_error)),
            "combined_hcurl": float(np.sqrt(e_error**2 + h_error**2 + ec_error**2 + hc_error**2)),
        }
