"""Physical electromagnetic field records in explicitly supplied DG bases.

Electric/magnetic times remain distinct. These records perform field evaluation
and error integration and impose no evolution algorithm or boundary condition.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.vector.curl import (
    CurlOperators as MaxwellLocal,
)
from pymhm.fem.vector.curl import (
    TangentialTraceSpace as MaxwellSkeleton,
)
from pymhm.fem.vector.curl import (
    field_values,
    physical_basis,
    physical_points,
    quadrature,
)
from pymhm.meshes.cartesian import CartesianMacroMesh


@dataclass(frozen=True)
class MaxwellSolution:
    """Staggered physical fields: E at electric_time and H at magnetic_time.

    Cell-local cardinal coefficients interleave physical components. The trace
    is lambda=H cross n on canonical macrofaces. The broken DG curl is distinct
    from an H(curl)-conforming reconstruction.
    """

    skeleton: MaxwellSkeleton
    locals: tuple[MaxwellLocal, ...]
    electric: tuple[FloatArray, ...]
    magnetic: tuple[FloatArray, ...]
    trace: FloatArray
    electric_time: float
    magnetic_time: float
    time_step: float
    frequency_bound: float
    energy: float
    energy_balance_residual: float

    def sample(self, cell: int, barycentric: Any) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate incident DG polynomials on every local fine cell.

        Simplexes use barycentric coordinates; rectangles use (x,y) in [0,1]^2.
        Every incident value is retained independently at broken interfaces.
        """
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.locals):
            raise ValueError("cell outside macro mesh")
        local = self.locals[cell]
        bary = np.asarray(barycentric)
        rectangle = isinstance(local.mesh, CartesianMacroMesh)
        if (
            np.iscomplexobj(bary)
            or bary.ndim != 2
            or bary.shape[1] != (2 if rectangle else local.mesh.cells.shape[1])
            or not np.isfinite(bary).all()
            or np.any(bary < -1e-13)
            or (
                np.any(bary > 1 + 1e-13)
                if rectangle
                else not np.allclose(bary.sum(axis=1), 1, atol=1e-13, rtol=0)
            )
        ):
            raise ValueError("sample coordinates must lie in the fine reference cell")
        basis, _ = physical_basis(local.mesh, local.degree, bary)
        dimension = local.mesh.points.shape[1]
        electric_components = 1 if dimension == 2 else 3
        electric = self.electric[cell].reshape(len(local.mesh.cells), -1, electric_components)
        magnetic = self.magnetic[cell].reshape(len(local.mesh.cells), -1, dimension)
        points = physical_points(local.mesh, bary)
        return (
            points,
            np.einsum("tqi,tia->tqa", basis, electric),
            np.einsum("tqi,tia->tqa", basis, magnetic),
        )

    def l2_errors(self, electric: Any, magnetic: Any, order: int = 6) -> tuple[float, float]:
        """Integrate physical L² errors at the explicitly distinct staggered times."""
        totals = np.zeros(2)
        for local, e, h in zip(self.locals, self.electric, self.magnetic, strict=True):
            bary, weights, _ = quadrature(local.mesh, 1, order)
            basis, _ = physical_basis(local.mesh, local.degree, bary)
            points = physical_points(local.mesh, bary)
            dimension = points.shape[-1]
            entries = ((0, e, electric, 1 if dimension == 2 else 3), (1, h, magnetic, dimension))
            for j, values, exact, components in entries:
                coefficients = values.reshape(len(local.mesh.cells), -1, components)
                result = np.einsum("tqi,tia->tqa", basis, coefficients)
                truth = field_values(exact, points.reshape(-1, dimension), components)
                totals[j] += np.sum(
                    weights * np.sum((result - truth.reshape(result.shape)) ** 2, axis=-1)
                )
        return float(np.sqrt(totals[0])), float(np.sqrt(totals[1]))
