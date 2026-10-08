"""Exact scalar energy and face moments in a metric-quadratic simplex basis.

The scalar basis has constant, mean-zero affine and mean-zero metric-quadratic
coordinates. These finite-dimensional operators can be composed with arbitrary
trial/test trace equations; no global method, gauge or solver is selected.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature
from pymhm.materials.evaluation import (
    scalar_values,
    scalar_values_3d,
    tensor_values,
    tensor_values_3d,
)
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def _volumes(mesh: Any) -> FloatArray:
    """Read cell measures in the two supported simplex dimensions."""
    return mesh.areas if isinstance(mesh, TriangleMesh) else mesh.volumes


def _rule(dimension: int, order: int) -> tuple[FloatArray, FloatArray]:
    """Use positive, unit-measure simplex quadrature in the physical dimension."""
    return triangle_quadrature(order) if dimension == 2 else tetrahedron_quadrature(order)


@dataclass(frozen=True)
class AnalyticDarcySpace:
    """Constant, mean-zero affine and mean-zero metric-quadratic local basis."""

    center: FloatArray
    covariance: FloatArray
    permeability: FloatArray

    def evaluate(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate basis and gradients at physical points, preserving metric units."""
        points = np.asarray(points)
        d = len(self.center)
        if (
            np.iscomplexobj(points)
            or points.ndim != 2
            or points.shape[1] != d
            or not np.isfinite(points).all()
        ):
            raise ValueError("points must be finite physical coordinates of the simplex dimension")
        x = points - self.center
        inverse = np.linalg.inv(self.permeability)
        quadratic = (
            np.einsum("qi,ij,qj->q", x, inverse, x) - np.einsum("ij,ji", inverse, self.covariance)
        ) / 2
        basis = np.column_stack((np.ones(len(points)), x, quadratic))
        gradient = np.zeros((len(points), d + 2, d))
        gradient[:, 1 : d + 1] = np.eye(d)
        gradient[:, -1] = x @ inverse
        return basis, gradient

    def integrate_rt0(
        self, pressure_mean: float, centroid_flux: Any, divergence: float
    ) -> FloatArray:
        """Integrate an RT0 field into the unique potential with its prescribed mean.

        The field is ``q(c) + divergence/d * (x-c)``. The result contains the
        coefficients in this space, with ``-K grad(p)=q`` identically. A supplied
        classical RT0 pressure mean is preserved; this operation does not add
        the variable-source moment correction of the analytical MHM equations.
        """
        flux = np.asarray(centroid_flux)
        if (
            flux.shape != self.center.shape
            or np.iscomplexobj(flux)
            or not np.isfinite(flux).all()
            or not np.isfinite([pressure_mean, divergence]).all()
        ):
            raise ValueError(
                "RT0 integration requires a finite mean, divergence and centroid vector"
            )
        return np.r_[
            pressure_mean, -np.linalg.solve(self.permeability, flux), -divergence / len(self.center)
        ]


def metric_quadratic_operators(
    mesh: TriangleMesh | TetraMesh,
    cell: int,
    permeability: Any = 1.0,
    source: Any = 0.0,
    *,
    quadrature_order: int = 8,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, AnalyticDarcySpace]:
    """Build exact harmonic lift equations and all variable-source moments.

    Permeability is one constant SPD tensor on this macrocell. The local trial
    span is ``{1, x_i-center_i, (x-center).K^-1.(x-center)/2}``. Its gradients
    generate exactly the RT0 vector space after multiplication by K. Basis
    functions other than the constant have zero physical mean. The source
    response in this small polynomial space is used for the global load only;
    a full variable-source potential requires a separate local Neumann solve.
    """
    positive_int(cell, "cell", 0)
    if not isinstance(mesh, (TriangleMesh, TetraMesh)) or cell >= len(mesh.cells):
        raise ValueError("provide a valid cell of a triangular or tetrahedral mesh")
    order = positive_int(quadrature_order, "quadrature_order", 3)
    vertices = mesh.points[mesh.cells[cell]]
    d = vertices.shape[1]
    tensor = (tensor_values if d == 2 else tensor_values_3d)(permeability, vertices[:1])[0]
    if callable(permeability) or np.asarray(permeability).shape not in ((), (d, d)):
        raise ValueError("analytical harmonic lifts require a macrocell-constant SPD tensor")
    center = vertices.mean(axis=0)
    covariance = (vertices - center).T @ (vertices - center) / ((d + 1) * (d + 2))
    space = AnalyticDarcySpace(center, covariance, tensor)
    volume = _volumes(mesh)[cell]
    matrix = np.zeros((d + 2, d + 2))
    matrix[1 : d + 1, 1 : d + 1] = volume * tensor
    matrix[-1, -1] = volume * np.trace(np.linalg.solve(tensor, covariance))
    coupling = np.zeros((d + 2, d + 1))
    for side, face in enumerate(mesh.cell_faces[cell]):
        points = mesh.points[mesh.faces[face]]
        face_center = points.mean(axis=0)
        delta = face_center - center
        face_covariance = (points - face_center).T @ (points - face_center) / (d * (d + 1))
        means = np.r_[
            1.0,
            delta,
            0.5
            * np.trace(
                np.linalg.solve(tensor, face_covariance + np.outer(delta, delta) - covariance)
            ),
        ]
        measure = mesh.lengths[face] if isinstance(mesh, TriangleMesh) else mesh.areas[face]
        coupling[:, side] = mesh.signs[cell, side] * measure * means
    bary, weights = _rule(d, order)
    physical = bary @ vertices
    values = (scalar_values if d == 2 else scalar_values_3d)(source, physical)
    load = volume * space.evaluate(physical)[0].T @ (weights * values)
    kernel = np.eye(d + 2)[:, :1]
    return matrix, coupling, load, volume * kernel, space
