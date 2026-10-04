"""Explicit positive quadrature for layered coefficients and circular radial loads.

The quadrature protocol retains the original triangular approximation space.
Returned weights are divided by original-cell area; a supported load may cover
only part of a cell. Material values carry arbitrary trailing scalar/vector/
tensor axes, so a two-dimensional Kelvin stiffness may have shape (3, 3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.meshes.fitting import _split, _triangles
from pymhm.meshes.triangle import TriangleMesh


@runtime_checkable
class TriangleQuadratureField(Protocol):
    """A field supplying points, area-normalized weights and flattened values.

    Arrays have shapes (cells, points, 3), (cells, points), and
    (cells * points, ...). Zero weights pad shorter cell rules. Quadrature may
    omit regions where the field is exactly zero; it never changes trial DOFs.
    """

    def triangle_quadrature(
        self, mesh: TriangleMesh, order: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return a positive integration rule for this field on the given mesh."""
        ...


@runtime_checkable
class TimeDependentTriangleField(Protocol):
    """A time-dependent load with a spatial quadrature provider at each time."""

    def at_time(self, time: float) -> TriangleQuadratureField:
        """Return the physical field at a finite time, without density weighting."""
        ...


@runtime_checkable
class SeparableTriangleField(Protocol):
    """Explicit spatial quadrature snapshot multiplied by a real scalar in time.

    This opt-in contract is f(t,x)=time_scale(t)*spatial_field(x). Both methods
    use the same physical force-density convention as ``at_time``. Preparation
    integrates the spatial snapshot through the original local basis/rule once;
    a changing support, material-weighted load or nonseparable callback must
    continue to use the ordinary ``at_time``/callback path.
    """

    def spatial_field(self) -> TriangleQuadratureField:
        """Return the declared time-independent spatial force-density provider."""
        ...

    def time_scale(self, time: float) -> float:
        """Return its finite real scalar amplitude at a finite real time."""
        ...


def triangle_field_quadrature(
    mesh: TriangleMesh, field: TriangleQuadratureField, order: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate a provider's cellwise coordinates, positive weights and values.

    Providers own their integration error controls. This check enforces the
    array/geometry contract, not accuracy for an arbitrary integrand. Inactive
    support has zero weight; a whole-cell material rule must additionally
    conserve cell area, as done by :class:`PolylineLayerField`.
    """
    if not isinstance(mesh, TriangleMesh):
        raise ValueError("field quadrature requires a triangular mesh")
    raw = field.triangle_quadrature(mesh, positive_int(order, "quadrature order"))
    if any(np.iscomplexobj(value) for value in raw):
        raise ValueError("field quadrature arrays must be real")
    bary, weights, values = (np.asarray(value, dtype=float) for value in raw)
    if (
        bary.ndim != 3
        or bary.shape[0] != len(mesh.cells)
        or bary.shape[2] != 3
        or bary.shape[1] == 0
        or weights.shape != bary.shape[:2]
        or values.ndim == 0
        or values.shape[0] != weights.size
        or not all(np.isfinite(value).all() for value in (bary, weights, values))
        or np.any(weights < 0)
        or np.min(bary) < -256 * np.finfo(float).eps
        or np.max(abs(bary.sum(axis=2) - 1)) > 256 * np.finfo(float).eps
    ):
        raise ValueError("invalid triangular field quadrature contract")
    return bary, weights, values


def _pack(
    mesh: TriangleMesh,
    rules: list[tuple[FloatArray, FloatArray, FloatArray]],
    tail: tuple[int, ...],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Pad physical cell rules and map them to original-cell barycentric coordinates."""
    size = max(1, max(len(weights) for _, weights, _ in rules))
    bary = np.full((len(mesh.cells), size, 3), 1 / 3)
    weights = np.zeros((len(mesh.cells), size))
    values = np.zeros((len(mesh.cells), size, *tail))
    for cell, (points, measure, samples) in enumerate(rules):
        count = len(measure)
        if count == 0:
            continue
        vertices = mesh.points[mesh.cells[cell]]
        active = measure > 0
        locations = np.where(active[:, None], points, vertices.mean(axis=0))
        coordinates = np.linalg.solve((vertices[1:] - vertices[0]).T, (locations - vertices[0]).T).T
        bary[cell, :count, 0] = 1 - coordinates.sum(axis=1)
        bary[cell, :count, 1:] = coordinates
        weights[cell, :count] = measure / mesh.areas[cell]
        values[cell] = samples[0]
        values[cell, :count] = samples
    return bary, weights, values.reshape(-1, *tail)


@dataclass(frozen=True)
class PolylineLayerField:
    """Piecewise-constant data separated by noncrossing two-dimensional polylines.

    ``abscissae`` is strictly increasing. ``heights`` has one row per interface
    and increases strictly down its first axis at every abscissa. Layers are
    ordered by increasing second coordinate; an interface point belongs to the
    layer with the smaller index. Outside the abscissa range, each interface
    continues at its endpoint height. ``values`` has one leading entry per
    layer and arbitrary trailing axes, with no implicit Kelvin/Voigt conversion.
    Positivity or tensor symmetry is checked by the physical assembly owner.
    """

    abscissae: Any
    heights: Any
    values: Any

    def __post_init__(self) -> None:
        """Validate and freeze the geometric partition and its finite real data."""
        if any(np.iscomplexobj(v) for v in (self.abscissae, self.heights, self.values)):
            raise ValueError("polyline layers require finite real data")
        x, heights, values = (
            np.array(v, dtype=float, copy=True) for v in (self.abscissae, self.heights, self.values)
        )
        if (
            x.ndim != 1
            or len(x) < 2
            or np.any(np.diff(x) <= 0)
            or heights.ndim != 2
            or heights.shape[1] != len(x)
            or len(heights) == 0
            or np.any(np.diff(heights, axis=0) <= 0)
            or values.ndim == 0
            or len(values) != len(heights) + 1
            or not all(np.isfinite(v).all() for v in (x, heights, values))
        ):
            raise ValueError("polyline layers require ordered abscissae and noncrossing interfaces")
        for name, value in zip(
            ("abscissae", "heights", "values"), (x, heights, values), strict=True
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __call__(self, points: Any) -> FloatArray:
        """Evaluate the declared layer convention at finite physical coordinate pairs."""
        if np.iscomplexobj(points):
            raise ValueError("layer sample points must be finite real pairs")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("layer sample points must be finite real pairs")
        levels = np.array([np.interp(points[:, 0], self.abscissae, row) for row in self.heights])
        return self.values[np.sum(points[:, 1] > levels, axis=0)]

    def _uniform_value(self, vertices: FloatArray) -> FloatArray | None:
        """Prove a whole triangle lies in one layer using exact polyline extrema.

        Endpoint and interior-knot heights bound every affine segment over the
        triangle's horizontal range. Comparing these bounds with its vertical
        range is conservative; inconclusive cells follow geometric partitioning.
        """
        left, right = vertices[:, 0].min(), vertices[:, 0].max()
        x = np.r_[left, self.abscissae[(self.abscissae > left) & (self.abscissae < right)], right]
        levels = np.array([np.interp(x, self.abscissae, row) for row in self.heights])
        lower = np.r_[-np.inf, levels.max(axis=1)]
        upper = np.r_[levels.min(axis=1), np.inf]
        slope = np.max(abs(np.diff(self.heights, axis=1) / np.diff(self.abscissae)))
        envelope = (
            64
            * np.finfo(float).eps
            * (np.max(abs(vertices)) + np.max(abs(levels)) + slope * np.max(abs(self.abscissae)))
        )
        contained = (vertices[:, 1].min() > lower + envelope) & (
            vertices[:, 1].max() < upper - envelope
        )
        indices = np.flatnonzero(contained)
        return np.asarray(self.values[indices[0]]) if len(indices) else None

    def triangle_quadrature(
        self, mesh: TriangleMesh, order: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Integrate on exact polygonal intersections with every relevant layer segment.

        Positive Duffy rules integrate polynomial factors without fitting or
        enriching the original trial space. Splits use coordinate-roundoff
        envelopes from the shared planar geometry owner.
        """
        rule, weight = triangle_quadrature(positive_int(order, "quadrature order"))
        rules = []
        for cell, ids in enumerate(mesh.cells):
            vertices = mesh.points[ids]
            uniform = self._uniform_value(vertices)
            if uniform is not None:
                rules.append(
                    (
                        rule @ vertices,
                        weight * mesh.areas[cell],
                        np.broadcast_to(uniform, (len(weight), *self.values.shape[1:])),
                    )
                )
                continue
            lo, hi = vertices[:, 0].min(), vertices[:, 0].max()
            pieces = [vertices]
            for x in self.abscissae[(self.abscissae > lo) & (self.abscissae < hi)]:
                pieces = [
                    child for piece in pieces for child in _split(piece, np.array([1.0, 0.0]), x)
                ]
            layers = []
            for piece in pieces:
                center = piece[:, 0].mean()
                left = int(np.searchsorted(self.abscissae, center) - 1)
                parts = [piece]
                for row in self.heights:
                    if left < 0 or left >= len(self.abscissae) - 1:
                        slope, offset = 0.0, float(row[0] if left < 0 else row[-1])
                    else:
                        slope = (row[left + 1] - row[left]) / (
                            self.abscissae[left + 1] - self.abscissae[left]
                        )
                        offset = row[left] - slope * self.abscissae[left]
                    parts = [
                        child
                        for part in parts
                        for child in _split(part, np.array([-slope, 1.0]), offset)
                    ]
                layers.extend(parts)
            points, measures, samples = [], [], []
            for piece in layers:
                value = self(piece.mean(axis=0)[None])[0]
                for triangle in _triangles(piece):
                    edge = triangle[1:] - triangle[0]
                    area = abs(edge[0, 0] * edge[1, 1] - edge[0, 1] * edge[1, 0]) / 2
                    points.append(rule @ triangle)
                    measures.append(weight * area)
                    samples.append(np.broadcast_to(value, (len(weight), *self.values.shape[1:])))
            rules.append((np.vstack(points), np.concatenate(measures), np.concatenate(samples)))
        result = _pack(mesh, rules, self.values.shape[1:])
        if np.max(abs(result[1].sum(axis=1) - 1)) > 512 * np.finfo(float).eps:
            raise ValueError("polyline intersection area is unresolved at the coordinate scale")
        return result


def _radial_bounds(vertices: FloatArray, angle: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Intersect rays from the origin with a counterclockwise triangle and unit disk."""
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    lower, upper = np.zeros(len(angle)), np.ones(len(angle))
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        edge = end - start
        numerator = edge[0] * start[1] - edge[1] * start[0]
        denominator = edge[0] * direction[:, 1] - edge[1] * direction[:, 0]
        ratio = np.divide(
            numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0
        )
        lower = np.maximum(lower, np.where(denominator > 0, ratio, 0))
        upper = np.minimum(upper, np.where(denominator < 0, ratio, 1))
        upper[(denominator == 0) & (numerator > 0)] = 0
    lower = np.clip(lower, 0, 1)
    return lower, np.maximum(lower, np.clip(upper, 0, 1))


def _angles(vertices: FloatArray) -> FloatArray:
    """Split at triangle vertex rays and actual edge intersections with the unit circle."""
    points = list(vertices)
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        edge = end - start
        length = np.linalg.norm(edge)
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]])
        distance = start @ normal
        height_squared = 1 - distance * distance
        if height_squared >= 0:
            foot = distance * normal
            for point in (
                foot - np.sqrt(height_squared) * tangent,
                foot + np.sqrt(height_squared) * tangent,
            ):
                parameter = (point - start) @ tangent / length
                if 0 < parameter < 1:
                    points.append(point)
    array = np.asarray(points)
    angle = np.mod(np.arctan2(array[:, 1], array[:, 0]), 2 * np.pi)
    return np.unique(np.r_[0.0, angle, 2 * np.pi])


def _angular_rule(
    vertices: FloatArray, left: float, right: float, order: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return angular Gaussian points and analytic ray endpoints in normalized geometry."""
    nodes, weights = leggauss(order)
    angle = (left + right) / 2 + (right - left) * nodes / 2
    lower, upper = _radial_bounds(vertices, angle)
    return angle, weights * (right - left) / 2, lower, upper


def _moments(
    rule: tuple[FloatArray, FloatArray, FloatArray, FloatArray], degree: int
) -> FloatArray:
    """Integrate dimensionless scalar and radial-vector monomials over a ray rule."""
    angle, weights, lower, upper = rule
    cosine, sine = np.cos(angle), np.sin(angle)
    result: list[float] = []
    for total in range(degree + 1):
        radial = weights * (upper ** (total + 2) - lower ** (total + 2)) / (total + 2)
        for first in range(total + 1):
            value = radial * cosine**first * sine ** (total - first)
            result.extend((value.sum(), value @ cosine, value @ sine))
    return np.asarray(result)


@dataclass(frozen=True)
class RadialDiskLoad:
    """A finite-radius radial force density with independent cut-disk quadrature.

    The field is ``amplitude*(x-center)/|x-center|`` for 0<distance<radius and
    zero elsewhere. Amplitude is force per volume, not acceleration; no density
    factor is applied. An optional scalar ``time_function`` is sampled by
    ``at_time``. Angular adaptivity controls all scalar and radial-vector
    monomials through degree 2*order-2 in coordinates normalized by radius.
    ``tolerance`` bounds the absolute estimated error of these dimensionless
    moments per original cell, before multiplication by radius squared.
    The difference of two Gaussian rules is an estimator, not a certified bound.
    """

    center: Any
    radius: float
    amplitude: float = 1.0
    time_function: Callable[[float], float] | None = None
    tolerance: float = 1e-12
    max_depth: int = 12
    _spatial_snapshot: RadialDiskLoad | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _sampled_scale: float = field(default=1.0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze a finite center and reject invalid support or integration parameters."""
        if np.iscomplexobj(self.center) or any(
            np.iscomplexobj(v) for v in (self.radius, self.amplitude, self.tolerance)
        ):
            raise ValueError("disk load parameters must be finite and real")
        center = np.array(self.center, dtype=float, copy=True)
        if (
            center.shape != (2,)
            or not np.isfinite(center).all()
            or not np.isfinite([self.radius, self.amplitude, self.tolerance]).all()
            or self.radius <= 0
            or self.tolerance <= 0
        ):
            raise ValueError("disk load needs a finite center, positive radius and tolerance")
        positive_int(self.max_depth, "angular subdivision depth")
        if self.time_function is not None and not callable(self.time_function):
            raise ValueError("time_function must be a scalar callable")
        center.setflags(write=False)
        object.__setattr__(self, "center", center)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore validated, read-only spatial coordinates after pickle/spawn."""
        self.__dict__.update(state)
        self.__post_init__()

    def at_time(self, time: float) -> RadialDiskLoad:
        """Freeze the physical field and its declared base-vector temporal scaling.

        Public amplitude and samples carry the physical product. The separable
        protocol retains the unscaled spatial provider and sampled scalar, so
        force assembly integrates the same spatial load before multiplying it.
        Replacing public data resets this private snapshot contract.
        """
        scale = self.time_scale(time)
        spatial = self.spatial_field()
        result = replace(spatial, amplitude=spatial.amplitude * scale, time_function=None)
        object.__setattr__(result, "_spatial_snapshot", spatial)
        object.__setattr__(result, "_sampled_scale", scale)
        return result

    def spatial_field(self) -> RadialDiskLoad:
        """Return the declared unscaled disk and its original quadrature controls."""
        if self._spatial_snapshot is not None:
            return self._spatial_snapshot
        return replace(self, time_function=None)

    def time_scale(self, time: float) -> float:
        """Evaluate a finite real scalar temporal factor without changing spatial support."""
        if np.iscomplexobj(time) or np.ndim(time) != 0 or not np.isfinite(time):
            raise ValueError("load time must be finite")
        scale = self._sampled_scale if self.time_function is None else self.time_function(time)
        if np.iscomplexobj(scale) or np.ndim(scale) != 0 or not np.isfinite(scale):
            raise ValueError("load time scale must be a finite real scalar")
        scale = float(scale)
        if not np.isfinite(scale):
            raise ValueError("load time scale must be finite in binary64")
        return scale

    def __call__(self, points: Any) -> FloatArray:
        """Evaluate the static amplitude; call at_time first for a temporal modulation."""
        if np.iscomplexobj(points):
            raise ValueError("disk sample points must be finite real pairs")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("disk sample points must be finite real pairs")
        displacement = points - self.center
        radius = np.linalg.norm(displacement, axis=1)
        active = (radius > 0) & (radius < self.radius)
        result = np.zeros_like(points)
        result[active] = self.amplitude * displacement[active] / radius[active, None]
        return result

    def triangle_quadrature(
        self, mesh: TriangleMesh, order: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Integrate exact circular intersections without polygonizing the source disk."""
        order = positive_int(order, "quadrature order")
        radial_nodes, radial_weights = leggauss(order)
        angular_order = max(8, order)
        rules = []
        for ids in mesh.cells:
            vertices = (mesh.points[ids] - self.center) / self.radius
            if np.any(vertices.min(axis=0) > 1) or np.any(vertices.max(axis=0) < -1):
                rules.append((np.empty((0, 2)), np.empty(0), np.empty((0, 2))))
                continue
            breaks = _angles(vertices)
            pending = [
                (float(a), float(b), 0) for a, b in zip(breaks[:-1], breaks[1:], strict=True)
            ]
            accepted = []
            while pending:
                left, right, depth = pending.pop()
                low = _angular_rule(vertices, left, right, angular_order)
                high = _angular_rule(vertices, left, right, 2 * angular_order)
                error = np.max(abs(_moments(high, 2 * order - 2) - _moments(low, 2 * order - 2)))
                budget = self.tolerance * (right - left) / (2 * np.pi)
                if error <= budget:
                    accepted.append(high)
                elif depth >= self.max_depth:
                    raise ValueError(
                        "disk angular quadrature did not reach its requested tolerance"
                    )
                else:
                    middle = (left + right) / 2
                    pending.extend(((left, middle, depth + 1), (middle, right, depth + 1)))
            points, weights, values = [], [], []
            for angle, angular_weights, lower, upper in accepted:
                radial = lower[:, None] + (upper - lower)[:, None] * (radial_nodes + 1) / 2
                measure = (
                    angular_weights[:, None]
                    * (upper - lower)[:, None]
                    * radial_weights
                    / 2
                    * radial
                )
                direction = np.column_stack((np.cos(angle), np.sin(angle)))
                points.append(
                    (self.center + self.radius * radial[..., None] * direction[:, None]).reshape(
                        -1, 2
                    )
                )
                weights.append((self.radius**2 * measure).ravel())
                values.append(
                    np.broadcast_to(
                        self.amplitude * direction[:, None], (*radial.shape, 2)
                    ).reshape(-1, 2)
                )
            measure = np.concatenate(weights)
            active = measure > 0
            rules.append((np.vstack(points)[active], measure[active], np.vstack(values)[active]))
        return _pack(mesh, rules, (2,))
