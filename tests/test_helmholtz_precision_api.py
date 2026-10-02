"""Exercise explicit local correction precision through the public acoustic solver."""

import numpy as np
import pytest

from pymhm.helmholtz import solve_helmholtz
from pymhm.quadrilateral import CartesianMacroMesh


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="extended local correction requires a wider native NumPy type",
)
def test_acoustic_api_retains_corrected_local_fields():
    """Use nonhomogeneous exact linear pressure with original local equations."""
    omega = 1.2

    def pressure(points):
        """Return the manufactured complex linear pressure."""
        return 1 + points[:, 0] + 0.3j * points[:, 1]

    solution = solve_helmholtz(
        CartesianMacroMesh(2, 1),
        omega=omega,
        bulk_modulus=3.0,
        source=lambda points: -(omega**2) * pressure(points) / 3.0,
        dirichlet=pressure,
        absorbing=None,
        degree=3,
        local_refinement=2,
        local_refinement_precision="extended",
        quadrature_order=8,
    )
    for coefficients, response in zip(solution.pressure, solution.system.responses, strict=True):
        assert coefficients.dtype == np.dtype(np.clongdouble)
        assert response.lifts.dtype == np.dtype(np.longdouble)
    assert np.max(np.abs(solution.conservation_residuals())) < 1e-10


def test_acoustic_api_rejects_unsupported_local_precision():
    """Validation remains owned by LocalProblem rather than a case-specific fallback."""
    with pytest.raises(ValueError, match="double or extended"):
        solve_helmholtz(CartesianMacroMesh(), omega=1.2, local_refinement_precision="quadruple")
