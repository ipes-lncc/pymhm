"""Executed displacement, velocity and force records for second-order equations.

These records retain physical bases and independent incident values. They do
not choose time integration, local elimination, boundary data or a gauge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.scalar.triangle import reference_values, tabulate
from pymhm.fem.vector.curl import physical_points
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D
from pymhm.materials.elasticity import KELVIN_BASIS_2D, constitutive_values, constitutive_values_3d
from pymhm.materials.evaluation import vector_values, vector_values_3d
from pymhm.materials.sources import (
    SeparableTriangleField,
    TimeDependentTriangleField,
    TriangleQuadratureField,
    triangle_field_quadrature,
)


@dataclass(frozen=True)
class ElastodynamicLocal:
    """Physical mass/stiffness, signed traction moments and continuous local Pk space."""

    mesh: Any
    degree: int
    mass: Any
    stiffness: Any
    coupling: FloatArray
    trace_dofs: IntArray
    dofs: IntArray
    nodes: FloatArray
    basis: FloatArray
    points: FloatArray
    weights: FloatArray
    density_values: FloatArray
    constitutive: Any
    lame_lambda: Any
    lame_mu: Any
    quadrature_order: int

    def load(self, field: Any, *, density_weighted: bool = False) -> FloatArray:
        """Integrate physical force density or a density-weighted initial field.

        Explicit triangular quadrature providers are supported for force loads.
        They are rejected for density-weighted initial projection, which would
        require a verified joint partition of both independent fields.
        """
        dimension = self.nodes.shape[1]
        evaluate = vector_values if dimension == 2 else vector_values_3d
        if isinstance(field, TriangleQuadratureField):
            if density_weighted:
                raise ValueError(
                    "custom source quadrature cannot define a density-weighted initial projection"
                )
            if dimension != 2:
                raise ValueError("custom triangular source quadrature requires two dimensions")
            bary, normalized, data = triangle_field_quadrature(
                self.mesh, field, self.quadrature_order
            )
            if not np.any(normalized):
                return np.zeros(self.mass.shape[0])
            basis = reference_values(self.degree, bary.reshape(-1, 3)).reshape(*bary.shape[:2], -1)
            points = physical_points(self.mesh, bary)
            values = evaluate(data, points.reshape(-1, dimension)).reshape(points.shape)
            weights = normalized * self.mesh.areas[:, None]
            blocks = np.einsum("tq,tqi,tqa->tia", weights, basis, values)
        else:
            values = evaluate(field, self.points.reshape(-1, dimension)).reshape(self.points.shape)
            coefficient = self.density_values if density_weighted else np.ones_like(self.weights)
            blocks = np.einsum("tq,tqi,tqa->tia", self.weights * coefficient, self.basis, values)
        indices = dimension * self.dofs[:, :, None] + np.arange(dimension)
        return np.bincount(indices.ravel(), weights=blocks.ravel(), minlength=self.mass.shape[0])

    def load_at_time(self, source: Any, time: float) -> FloatArray:
        """Evaluate a temporal source while preserving opt-in spatial quadrature.

        Ordinary callbacks retain the existing ``source(time, points)`` path.
        A provider with ``at_time`` returns a spatial quadrature field. Both
        paths supply physical force density, with no implicit density factor.
        """
        if isinstance(source, PreparedElastodynamicSource):
            return source.load_at_time(self, time)
        if isinstance(source, SeparableTriangleField):
            return _scale_source_load(self.load(source.spatial_field()), source.time_scale, time)
        if isinstance(source, TimeDependentTriangleField):
            return self.load(source.at_time(time))
        if isinstance(source, TriangleQuadratureField):
            return self.load(source)
        return self.load(
            source(time, self.points.reshape(-1, self.nodes.shape[1]))
            if callable(source)
            else source
        )


@dataclass(frozen=True)
class PreparedElastodynamicSource:
    """Read-only original spatial force vectors in their executed local nodal bases.

    Loads use binary64 and have no implicit density weighting. Preparation is
    explicit: vectors belong to the exact local objects that integrated the
    source. They cannot be reused on another stepper or approximation space.
    The temporal factor is evaluated at every original endpoint/substep time.
    This snapshot owns no native factorization and imposes no time update.
    """

    locals: tuple[ElastodynamicLocal, ...]
    loads: tuple[FloatArray, ...]
    time_function: Callable[[float], float]
    _indices: dict[int, int] = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and copy original nodal vectors without narrowing their dtype."""
        if (
            not self.locals
            or len(self.locals) != len(self.loads)
            or len({id(local) for local in self.locals}) != len(self.locals)
            or not callable(self.time_function)
        ):
            raise ValueError("provide distinct executed locals, matching loads and a time function")
        loads = []
        for local, raw in zip(self.locals, self.loads, strict=True):
            values = np.asarray(raw)
            if (
                local.nodes.shape[1] != 2
                or values.dtype != np.dtype(float)
                or values.shape != (local.mass.shape[0],)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "prepared loads require finite binary64 original planar nodal vectors"
                )
            owned = values.copy()
            owned.setflags(write=False)
            loads.append(owned)
        object.__setattr__(self, "loads", tuple(loads))
        object.__setattr__(self, "_indices", {id(local): i for i, local in enumerate(self.locals)})

    def load_at_time(self, local: ElastodynamicLocal, time: float) -> FloatArray:
        """Scale the original vector, rejecting a different basis/mesh and nonreal time data."""
        index = self._indices.get(id(local))
        if index is None or self.locals[index] is not local:
            raise ValueError("prepared source belongs to different executed local operators")
        return _scale_source_load(self.loads[index], self.time_function, time)


def _scale_source_load(
    spatial: FloatArray, time_function: Callable[[float], float], time: float
) -> FloatArray:
    """Apply the declared separable temporal factor after the spatial integration."""
    if np.iscomplexobj(time) or np.ndim(time) != 0 or not np.isfinite(time):
        raise ValueError("prepared source time must be finite and real")
    factor = time_function(time)
    if np.iscomplexobj(factor) or np.ndim(factor) != 0 or not np.isfinite(factor):
        raise ValueError("prepared source time scale must be a finite real scalar")
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("prepared source time scale must be finite in binary64")
    with np.errstate(over="ignore", invalid="ignore"):
        load = spatial * factor
    if not np.isfinite(load).all():
        raise ValueError("prepared source scaled load must remain finite")
    return load


def stress_from_gradient(
    local: ElastodynamicLocal, points: FloatArray, gradient: FloatArray
) -> FloatArray:
    """Apply the local Kelvin constitutive law to an already evaluated gradient.

    ``points`` has shape ``(fine cells, quadrature points, dimension)`` and
    ``gradient`` appends its derivative coordinate axis. The caller owns the
    one-sided cell evaluation; no spatial interpolation or averaging occurs.
    """
    dimension = local.nodes.shape[1]
    kelvin, evaluate = (
        (KELVIN_BASIS_2D, constitutive_values)
        if dimension == 2
        else (KELVIN_BASIS_3D, constitutive_values_3d)
    )
    stiffness = evaluate(
        local.constitutive,
        points.reshape(-1, dimension),
        lame_lambda=local.lame_lambda,
        lame_mu=local.lame_mu,
    )
    strain = np.einsum("aij,tqij->tqa", kelvin, gradient).reshape(-1, len(kelvin))
    return np.einsum("nab,nb,aij->nij", stiffness, strain, kelvin).reshape(
        *points.shape[:2], dimension, dimension
    )


@dataclass(frozen=True)
class ElastodynamicSolution:
    """Displacement and velocity at a common macro time, and slabwise negative traction."""

    skeleton: Any
    locals: tuple[ElastodynamicLocal, ...]
    displacement: tuple[FloatArray, ...]
    velocity: tuple[FloatArray, ...]
    trace: FloatArray
    time: float
    energy: float
    constraint_residual: float

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one-sided continuous local polynomials on every fine simplex."""
        index = positive_int(cell, "cell", 0)
        if index >= len(self.locals):
            raise ValueError("cell outside local meshes")
        local = self.locals[index]
        dimension = local.nodes.shape[1]
        result = (
            tabulate(local.mesh, local.degree, bary)
            if dimension == 2
            else tetra_tabulate(local.mesh, local.degree, bary)
        )
        dofs, _, basis = result[:3]
        return tuple(
            np.einsum("qi,tia->tqa", basis, values.reshape(-1, dimension)[dofs])
            for values in (self.displacement[index], self.velocity[index])
        )

    def l2_error(self, exact: Any, *, velocity: bool = False, order: int = 8) -> float:
        """Integrate physical displacement/velocity L2 error with independent quadrature."""
        total = 0.0
        for cell, local in enumerate(self.locals):
            dimension = local.nodes.shape[1]
            bary, weights = (
                triangle_quadrature(order) if dimension == 2 else tetrahedron_quadrature(order)
            )
            numerical = self.evaluate(cell, bary)[int(velocity)]
            points = np.einsum("qi,tij->tqj", bary, local.mesh.points[local.mesh.cells])
            evaluate = vector_values if dimension == 2 else vector_values_3d
            target = evaluate(exact, points.reshape(-1, dimension)).reshape(points.shape)
            measure = local.mesh.areas if dimension == 2 else local.mesh.volumes
            total += float(measure @ (np.sum((numerical - target) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))

    def gradient(self, cell: int, bary: FloatArray, *, velocity: bool = False) -> FloatArray:
        """Evaluate the full broken displacement or velocity gradient in physical coordinates."""
        index = positive_int(cell, "cell", 0)
        if index >= len(self.locals):
            raise ValueError("cell outside local meshes")
        local = self.locals[index]
        dimension = local.nodes.shape[1]
        forms = (
            tabulate(local.mesh, local.degree, bary)
            if dimension == 2
            else tetra_tabulate(local.mesh, local.degree, bary)
        )
        dofs, _, _, derivative = forms[:4]
        values = self.velocity[index] if velocity else self.displacement[index]
        return np.einsum("tqib,tia->tqab", derivative, values.reshape(-1, dimension)[dofs])

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate raw symmetric Cauchy stress; no global H(div) conformity is implied."""
        gradient = self.gradient(cell, bary)
        local = self.locals[cell]
        points = np.einsum("qi,tij->tqj", bary, local.mesh.points[local.mesh.cells])
        return stress_from_gradient(local, points, gradient)
