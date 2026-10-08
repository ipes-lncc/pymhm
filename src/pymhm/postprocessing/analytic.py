"""Executed metric-quadratic harmonic fields and independent nodal source fields."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.metric import AnalyticDarcySpace, _rule, _volumes
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField, FieldDefinition
from pymhm.postprocessing.nodal import nodal_field


@dataclass(frozen=True)
class AnalyticDarcySolution:
    """Analytical trace-driven pressure/RT0 flux plus a separate source potential."""

    mesh: TriangleMesh | TetraMesh
    spaces: tuple[AnalyticDarcySpace, ...]
    harmonic: tuple[FloatArray, ...]
    source_meshes: tuple[Any, ...]
    source_pressure: tuple[FloatArray | None, ...]
    source_degree: int
    hybrid: HybridSolution
    source_fields: tuple[FieldDefinition | None, ...] = ()

    def __post_init__(self) -> None:
        """Archive each executed nodal source basis alongside its coefficient vector."""
        if not self.source_fields:
            object.__setattr__(
                self,
                "source_fields",
                tuple(
                    None
                    if coefficients is None
                    else nodal_field("source_potential", mesh, self.source_degree)
                    for mesh, coefficients in zip(
                        self.source_meshes, self.source_pressure, strict=True
                    )
                ),
            )
        if len(self.source_fields) != len(self.source_pressure):
            raise ValueError("source fields must match the executed source coefficient vectors")

    def pressure_update(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return p0+p_lambda and its exact RT0 physical flux, without p_f."""
        positive_int(cell, "cell", 0)
        if cell >= len(self.spaces):
            raise ValueError("cell index outside mesh")
        basis, gradient = self.spaces[cell].evaluate(points)
        coefficients = self.harmonic[cell]
        return basis @ coefficients, -np.einsum(
            "ab,qib,i->qa", self.spaces[cell].permeability, gradient, coefficients
        )

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate the full pressure and flux on every source-mesh cell at barycentric points."""
        mesh = self.source_meshes[cell]
        points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
        d = points.shape[-1]
        p, q = self.pressure_update(cell, points.reshape(-1, d))
        p, q = p.reshape(points.shape[:2]), q.reshape(points.shape)
        coefficients = self.source_pressure[cell]
        if coefficients is not None:
            definition = self.source_fields[cell]
            if definition is None:
                raise ValueError("a source coefficient vector requires its executed field basis")
            field = DiscreteField(definition, coefficients)
            owners = np.repeat(np.arange(len(mesh.cells)), len(bary))
            value, gradient = field.values_and_gradient(points.reshape(-1, d), cells=owners)
            p += value.reshape(p.shape)
            q -= np.einsum(
                "ab,tqb->tqa", self.spaces[cell].permeability, gradient.reshape(points.shape)
            )
        return p, q

    def errors(self, pressure: Any, flux: Any, order: int = 8) -> tuple[float, float]:
        """Integrate pressure and physical flux errors on the actual source meshes."""
        bary, weights = _rule(self.mesh.points.shape[1], order)
        error = np.zeros(2)
        for cell, mesh in enumerate(self.source_meshes):
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            p, q = self.evaluate(cell, bary)
            flat = points.reshape(-1, points.shape[-1])
            dp = p - pressure(flat).reshape(p.shape)
            dq = q - flux(flat).reshape(q.shape)
            error += [
                float(_volumes(mesh) @ (dp**2 @ weights)),
                float(_volumes(mesh) @ (np.sum(dq**2, axis=-1) @ weights)),
            ]
        return float(np.sqrt(error[0])), float(np.sqrt(error[1]))
