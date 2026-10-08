"""Executed complex acoustic fields and independent physical error integration.

These records select no acoustic method: coefficients and original assembled
systems are supplied explicitly by a user-defined variational application.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.helmholtz import acoustic_quadrature, acoustic_space, complex_values
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.linalg.complex import complexify_vector
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh


def local_helmholtz_error_squared(
    fine: TriangleMesh | CartesianMacroMesh,
    coefficients: Any,
    degree: int,
    exact: Any,
    order: int = 8,
    *,
    derivative: bool = False,
) -> float:
    """Integrate one local complex pressure or physical-gradient squared error.

    Coefficients use the continuous local Pk/Qk nodal ordering from
    :func:`pymhm.fem.scalar.helmholtz.acoustic_space`. ``exact(points)`` returns
    pressure values or, with ``derivative=True``, two physical gradient
    components. The Hermitian absolute square is integrated using the declared
    physical Gauss order. No frequency weight or interface averaging is applied;
    summing these integrals gives the squared broken norm across macro cells.
    """
    degree = positive_int(degree, "degree")
    order = positive_int(order, "order")
    dofs, points, weights, basis, gradient, _ = acoustic_quadrature(fine, degree, 1.0, order)
    coefficients = np.asarray(coefficients)
    if (
        coefficients.dtype.kind not in "biufc"
        or coefficients.shape != (int(dofs.max()) + 1,)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("local Helmholtz coefficients must be a finite nodal vector")
    if derivative:
        truth = np.asarray(
            exact(points.reshape(-1, 2)) if callable(exact) else exact, dtype=complex
        )
        truth = np.broadcast_to(truth, (points.size // 2, 2))
        approximate = np.einsum("tqia,ti->tqa", gradient, coefficients[dofs])
        square = np.sum(abs(approximate - truth.reshape(points.shape)) ** 2, axis=-1)
    else:
        truth = complex_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
        approximate = np.einsum("tqi,ti->tq", basis, coefficients[dofs])
        square = abs(approximate - truth) ** 2
    if not np.isfinite(square).all():
        raise ValueError("exact fields must be finite")
    return float(np.sum(weights * square))


@dataclass(frozen=True)
class HelmholtzSolution:
    """Broken complex pressure and physical normal flux outside absorbing faces.

    ``pressure`` and ``trace`` expose complex coefficients. ``hybrid`` retains
    the exact interleaved real coordinates used by the shared solvers. On
    absorbing faces the stored multiplier is zero; physical outgoing flux is
    -i*omega*u/sqrt(kappa*rho)-g and is included in macro conservation.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[Any, ...]
    pressure: tuple[Any, ...]
    trace: Any
    hybrid: HybridSolution
    system: HybridSystem
    omega: float
    density: Any
    bulk_modulus: Any
    source: Any
    degree: int
    quadrature_order: int
    absorbing: dict[int, Any]
    pml_stretch: Any
    point_sources: tuple[FloatArray, ...]

    def sample(self, cell: int, reference: Any) -> tuple[FloatArray, Any, Any]:
        """Evaluate every fine cell at reference points without averaging interfaces.

        Triangles use barycentric coordinates (q,3), rectangles use coordinates
        (q,2) in [0,1]^2. Return physical points, pressure and gradient with a
        leading fine-cell axis, preserving every incident value independently.
        """
        positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell index outside mesh")
        fine = self.local_meshes[cell]
        raw = np.asarray(reference)
        dimension = 3 if isinstance(fine, TriangleMesh) else 2
        if (
            np.iscomplexobj(raw)
            or raw.ndim != 2
            or raw.shape[1] != dimension
            or not np.isfinite(raw).all()
            or np.any(raw < 0)
            or np.any(raw > 1)
        ):
            raise ValueError("reference points must be finite coordinates in the reference element")
        reference = np.asarray(raw, dtype=float)
        if isinstance(fine, TriangleMesh):
            if not np.allclose(reference.sum(axis=1), 1, rtol=0, atol=1e-14):
                raise ValueError("barycentric coordinates must sum to one")
            dofs, _, basis, gradient, _ = tabulate(fine, self.degree, reference)
            points = np.einsum("qi,tia->tqa", reference, fine.points[fine.cells])
        else:
            dofs, _ = acoustic_space(fine, self.degree)
            basis, derivative = qk_basis(self.degree, reference)
            gradient = np.broadcast_to(
                derivative / fine.spacing, (len(fine.cells), *derivative.shape)
            )
            points = fine.points[fine.cells[:, 0], None, :] + reference * fine.spacing
        values = np.einsum("qi,ti->tq", basis, self.pressure[cell][dofs])
        gradients = np.einsum("tqia,ti->tqa", gradient, self.pressure[cell][dofs])
        return points, values, gradients

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the complex pressure error using the Hermitian absolute square."""
        return self._error(exact, order, False)

    def gradient_l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the physical complex gradient error, without a frequency weight."""
        return self._error(exact, order, True)

    def _error(self, exact: Any, order: int, derivative: bool) -> float:
        """Use independent physical quadrature for pressure or gradient norms."""
        order = positive_int(order, "order")
        total = 0.0
        for fine, coefficients in zip(self.local_meshes, self.pressure, strict=True):
            total += local_helmholtz_error_squared(
                fine, coefficients, self.degree, exact, order, derivative=derivative
            )
        return float(np.sqrt(total))

    def conservation_residuals(self) -> Any:
        """Return complex macro balances ∮q.n−omega²∫u/kappa−∫f−sum Q_j.

        Absorbing flux uses the discrete pressure trace and prescribed g. The
        result is evaluated through the assembled constant-test equations,
        including their physical material and face quadrature. With PML,
        q=-rho^-1*D*grad(u) and the mass integral contains d*u/kappa.
        """
        result = []
        for response, coefficients in zip(self.system.responses, self.hybrid.fields, strict=True):
            problem = response.problem
            residual = (
                problem.matrix @ coefficients
                + problem.coupling @ self.hybrid.trace[problem.trace_dofs]
                - problem.load
            )
            result.append(np.sum(complexify_vector(residual)))
        return np.asarray(result)
