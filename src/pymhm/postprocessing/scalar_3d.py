"""Physical scalar and conservative flux fields on original polygonal macrofaces."""

from dataclasses import dataclass
from typing import Any

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.solutions import RAD3DSolution


@dataclass(frozen=True)
class PolyhedralRADSolution:
    """Broken tetrahedral physical fields with their original polygonal skeleton."""

    field: RAD3DSolution
    skeleton: PolygonalSkeleton3D
    hybrid: HybridSolution

    @property
    def local_meshes(self) -> tuple[TetraMesh, ...]:
        """Return each independent conforming tetrahedral local discretization."""
        return self.field.local_meshes

    @property
    def values(self) -> tuple[FloatArray, ...]:
        """Return the unreconciled local scalar coefficients."""
        return self.field.values

    @property
    def degree(self) -> int:
        """Return the continuous local tetrahedral polynomial degree."""
        return self.field.degree

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate scalar and full physical flux -K grad(u)+beta u without averaging."""
        return self.field.evaluate(cell, bary)

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate the scalar error through the common tetrahedral evaluator."""
        return self.field.l2_error(exact, order)

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Integrate the complete broken physical gradient error."""
        return self.field.h1_seminorm_error(exact_gradient, order)

    def flux_l2_error(self, exact_flux: Any, order: int = 7) -> float:
        """Integrate the full conservative physical flux error."""
        return self.field.flux_l2_error(exact_flux, order)
