"""Polynomial field interchange and physical norms for the HPC4e comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hpc4e_data import HPC4EData

from pymhm.elasticity_tensor_rt import _rotation_basis
from pymhm.quadrilateral import CartesianMacroMesh, quadrilateral_quadrature
from pymhm.tensor_rt import tensor_rt_basis


@dataclass(frozen=True)
class RectangularElasticityField:
    """Canonical per-cell RT stress, Q displacement and P rotation on a Cartesian grid.

    All quantities are dimensionless under the common HPC4e scaling. Cells run
    fastest in x. RT coefficients use outward normals of an isolated reference
    rectangle, including the Piola scale of the physical fine-cell dimensions.
    """

    nx: int
    ny: int
    degree: int
    enrichment: int
    bounds: tuple[float, float, float, float]
    stress: np.ndarray
    displacement: np.ndarray
    rotation: np.ndarray

    @classmethod
    def load(cls, path: Path) -> RectangularElasticityField:
        """Read MHM or classical coefficients without an assembled system.

        Classical RTk/Qk/Pk archives omit the independent interior enrichment;
        their canonical coefficient shapes correspond to enrichment zero.
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                int(data["nx"]),
                int(data["ny"]),
                int(data["degree"]),
                int(data["enrichment"]) if "enrichment" in data else 0,
                tuple(float(v) for v in data["bounds"]),
                data["stress"],
                data["displacement"],
                data["rotation"],
            )

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, ...]:
        """Evaluate one-sided displacement, full stress, divergence and rotation."""
        x0, x1, y0, y1 = self.bounds
        spacing = np.array([(x1 - x0) / self.nx, (y1 - y0) / self.ny])
        scaled = (np.asarray(points) - [x0, y0]) / spacing
        shape = np.array([self.nx, self.ny])
        tol = 32 * np.finfo(float).eps * shape
        if (
            scaled.ndim != 2
            or scaled.shape[1] != 2
            or not np.isfinite(scaled).all()
            or np.any(scaled < -tol)
            or np.any(scaled > shape + tol)
        ):
            raise ValueError("evaluation points must lie in the closed rectangular domain")
        indices = np.minimum(np.maximum(scaled, 0).astype(np.int64), shape - 1)
        reference = np.clip(scaled - indices, 0, 1)
        ids = indices[:, 0] + self.nx * indices[:, 1]
        cell = CartesianMacroMesh(bounds=(0, spacing[0], 0, spacing[1]))
        vectors, divergence, scalar = tensor_rt_basis(cell, self.degree, self.enrichment, reference)
        rotation = _rotation_basis(self.degree + self.enrichment, reference)
        return (
            np.einsum("qi,qia->qa", scalar, self.displacement[ids]),
            np.einsum("qib,qia->qab", vectors[0], self.stress[ids]),
            np.einsum("qi,qia->qa", divergence[0], self.stress[ids]),
            np.einsum("qi,qi->q", rotation, self.rotation[ids]),
        )


def compare_fields(
    approximation: RectangularElasticityField,
    reference: RectangularElasticityField,
    material: HPC4EData,
    order: int = 4,
) -> dict[str, float]:
    """Integrate L2 and compliance norms on the common, material-aligned fine grid."""
    if approximation.bounds != reference.bounds:
        raise ValueError("physical domains must agree")
    nx, ny = max(approximation.nx, reference.nx), max(approximation.ny, reference.ny)
    for fx, fy in (
        (approximation.nx, approximation.ny),
        (reference.nx, reference.ny),
        material.young.shape,
    ):
        if nx % fx or ny % fy:
            raise ValueError("integration cells must resolve both solutions and material pixels")
    lam, mu, _ = material.fields()
    x0, x1, y0, y1 = reference.bounds
    spacing = np.array([(x1 - x0) / nx, (y1 - y0) / ny])
    points, weights = quadrilateral_quadrature(order)
    totals = np.zeros((2, 5), dtype=np.longdouble)
    for start in range(0, nx * ny, 256):
        ids = np.arange(start, min(start + 256, nx * ny))
        origins = np.column_stack((ids % nx, ids // nx)) * spacing + [x0, y0]
        physical = (origins[:, None] + points * spacing).reshape(-1, 2)
        target, actual = reference.evaluate(physical), approximation.evaluate(physical)
        shear, bulk = mu(physical), lam(physical)
        for index, values in enumerate(
            (tuple(a - b for a, b in zip(actual, target, strict=True)), target)
        ):
            u, sigma, div, rot = values
            trace = np.trace(sigma, axis1=-2, axis2=-1)
            deviator = sigma - trace[:, None, None] * np.eye(2) / 2
            density = np.column_stack(
                (
                    np.sum(u**2, axis=-1),
                    np.sum(sigma**2, axis=(-2, -1)),
                    np.sum(div**2, axis=-1),
                    rot**2,
                    np.sum(deviator**2, axis=(-2, -1)) / (2 * shear)
                    + trace**2 / (4 * (shear + bulk)),
                )
            )
            weighted = density.reshape(len(ids), len(points), 5) * weights[None, :, None]
            totals[index] += np.sum(weighted, axis=(0, 1), dtype=np.longdouble) * np.prod(spacing)
    errors, norms = np.sqrt(totals)
    report = {"quadrature_order": order}
    for name, error, norm in zip(
        ("displacement", "stress", "divergence", "rotation", "compliance"),
        errors,
        norms,
        strict=True,
    ):
        report[name + "_difference_dimensionless"] = float(error)
        report[name + "_reference_norm_dimensionless"] = float(norm)
        report[name + "_relative"] = float(error / norm) if norm else 0.0 if error == 0 else np.inf
    return report
