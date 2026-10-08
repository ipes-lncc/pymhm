"""Hydrodynamic material tensors and exact one-sided vector-field evaluation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.operators import rt0_evaluate
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.velocity import PolynomialDarcyVelocity, PrimalDarcyVelocity


class RT0VectorField:
    """Evaluate one macro-local RT0 field with an exact geometric membership test.

    A nearest-centroid query supplies candidates only; barycentric tests select
    the containing triangle and an exhaustive fallback handles skew meshes.
    Points on an internal fine edge choose one adjacent fine cell. Macro sides
    remain explicit because each instance owns only one macroelement.
    """

    def __init__(self, mesh: TriangleMesh, flux: Any) -> None:
        """Validate integrated face fluxes and cache affine RT0 coefficients."""
        raw = np.asarray(flux)
        if np.iscomplexobj(raw) or raw.shape != (len(mesh.faces),) or not np.isfinite(raw).all():
            raise ValueError("RT0 flux must contain one finite real integrated flux per fine face")
        self.mesh = mesh
        vertices = mesh.points[mesh.cells]
        self.centers = vertices.mean(axis=1)
        self.origins = vertices[:, 0]
        self.inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        self.tree = cKDTree(self.centers)
        self.center_values = rt0_evaluate(mesh, raw.astype(float), np.array([[1 / 3] * 3]))[:, 0]
        self.divergences = np.sum(raw[mesh.cell_faces] * mesh.signs, axis=1) / mesh.areas

    def locate(self, points: Any) -> IntArray:
        """Locate every point, rejecting exterior or nonfinite coordinates."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("evaluation points must have finite shape (n,2)")
        count = len(self.mesh.cells)
        candidates = np.asarray(self.tree.query(points, k=min(8, count))[1]).reshape(
            len(points), min(8, count)
        )
        result = np.full(len(points), -1, dtype=np.int64)
        tolerance = 64 * np.finfo(float).eps
        for row, (point, ids) in enumerate(zip(points, candidates, strict=True)):
            bary = np.einsum("tij,tj->ti", self.inverse[ids], point - self.origins[ids])
            inside = np.all(bary >= -tolerance, axis=1) & (bary.sum(axis=1) <= 1 + tolerance)
            if np.any(inside):
                result[row] = ids[np.flatnonzero(inside)[0]]
            else:
                bary = np.einsum("tij,tj->ti", self.inverse, point - self.origins)
                inside = np.all(bary >= -tolerance, axis=1) & (bary.sum(axis=1) <= 1 + tolerance)
                if not np.any(inside):
                    raise ValueError("evaluation point lies outside the supplied macroelement")
                result[row] = np.flatnonzero(inside)[0]
        return result

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate physical Darcy velocity q=a+b*x inside each fine cell."""
        cells = self.locate(points)
        return self.center_values[cells] + self.divergences[cells, None] / 2 * (
            points - self.centers[cells]
        )

    def divergence(self, points: FloatArray) -> FloatArray:
        """Return the exact piecewise-constant RT0 divergence."""
        return self.divergences[self.locate(points)]


@dataclass(frozen=True)
class HydrodynamicDispersion:
    """D=(alpha_m+alpha_t*|v|)I+(alpha_l-alpha_t)*v tensor v/|v|.

    The final term is defined as zero at zero velocity. Molecular diffusion is
    strictly positive; longitudinal and transverse dispersivities are nonnegative
    with alpha_l>=alpha_t. Coordinates, time and coefficient units must agree.
    """

    velocity: RT0VectorField | PolynomialDarcyVelocity | PrimalDarcyVelocity
    molecular: float
    longitudinal: float
    transverse: float

    def __post_init__(self) -> None:
        """Enforce ellipticity and the physical ordering of dispersivities."""
        values = [self.molecular, self.longitudinal, self.transverse]
        if (
            not np.isfinite(values).all()
            or self.molecular <= 0
            or not 0 <= self.transverse <= self.longitudinal
        ):
            raise ValueError("require molecular>0 and longitudinal>=transverse>=0, all finite")

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate the symmetric positive-definite dispersion tensor.

        Use equation (5.5) of
        [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499).
        """
        return self._tensor(self.velocity(points))

    def _tensor(self, velocity: FloatArray) -> FloatArray:
        """Evaluate dispersion from physical vectors using one declared zero convention."""
        magnitude = np.linalg.norm(velocity, axis=1)
        inverse = np.divide(1.0, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0)
        return (self.molecular + self.transverse * magnitude)[:, None, None] * np.eye(2) + (
            self.longitudinal - self.transverse
        ) * inverse[:, None, None] * velocity[:, :, None] * velocity[:, None, :]

    def divergence(self, points: FloatArray) -> FloatArray:
        """Differentiate D within each fine cell, without smoothing material jumps.

        Since grad(v)=b I in two dimensions, div(D)=(2 alpha_l-alpha_t)*b*v/|v|.
        Its value is set to zero at an isolated velocity zero, where |v| is not
        differentiable; this is an almost-everywhere derivative for broken SUPG.
        """
        velocity = self.velocity(points)
        magnitude = np.linalg.norm(velocity, axis=1)
        normalized = np.divide(
            velocity, magnitude[:, None], out=np.zeros_like(velocity), where=magnitude[:, None] > 0
        )
        if isinstance(self.velocity, RT0VectorField):
            return (
                (2 * self.longitudinal - self.transverse)
                * self.velocity.divergence(points)[:, None]
                / 2
                * normalized
            )
        gradient = self.velocity.gradient(points)
        derivative_magnitude = np.einsum("pij,pi->pj", gradient, normalized)
        directional = np.einsum("pij,pj->pi", gradient, normalized)
        longitudinal_derivative = np.einsum("pi,pi->p", normalized, directional)
        return self.transverse * derivative_magnitude + (self.longitudinal - self.transverse) * (
            directional
            + normalized * np.trace(gradient, axis1=1, axis2=2)[:, None]
            - normalized * longitudinal_derivative[:, None]
        )


class PrimalDispersion(HydrodynamicDispersion):
    """Integrate dispersion on the same material intersections as the primal Darcy solve."""

    velocity: PrimalDarcyVelocity

    def triangle_quadrature(
        self, mesh: TriangleMesh, order: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Retain exact cut material samples and unsmoothed fine pressure derivatives."""
        bary, weights, material = material_triangle_quadrature(mesh, self.velocity.material, order)
        bary = np.array(bary, copy=True)
        # Padded zero-weight points still reach coefficient callbacks. Move them
        # to a positive rule point of the same triangle without changing integrals.
        for cell in range(len(mesh.cells)):
            bary[cell, weights[cell] == 0] = bary[cell, np.flatnonzero(weights[cell] > 0)[0]]
        points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
        cells = np.repeat(np.arange(len(mesh.cells)), weights.shape[1])
        values = self.velocity.evaluate(points, cells, material_values=material)
        return bary, weights, self._tensor(values)


RT0DarcyVelocity = RT0VectorField
