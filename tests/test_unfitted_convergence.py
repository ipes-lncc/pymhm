"""Physical norm and configuration contracts of the published-curve comparison."""

import numpy as np
from numpy.testing import assert_allclose

from examples.unfitted_convergence import (
    error_norms,
    smooth_configurations,
    smooth_field,
    smooth_source,
)
from examples.unfitted_trace_family import ScalarTraceFamily
from pymhm.mesh import TriangleMesh


def test_distinct_pressure_gradient_flux_and_energy_norms_have_analytic_values():
    """Use a zero computed field and a nonzero exact quadratic with anisotropic K."""
    material = np.diag([2.0, 3.0])
    family = ScalarTraceFamily.prepare(
        TriangleMesh.unit_square(1),
        trace_degree=0,
        segments=1,
        local_degree=2,
        local_refinement=2,
        permeability=material,
    )
    solution, _ = family.solve(0, 1)

    def exact(points):
        """Provide pressure x²+2y² and its gradient independently."""
        x, y = points.T
        return x * x + 2 * y * y, np.column_stack((2 * x, 4 * y))

    result = error_norms(solution, exact, 5)
    for name, squared in zip(
        ("pressure", "gradient", "flux", "energy"), (13 / 9, 20 / 3, 160 / 3, 56 / 3), strict=True
    ):
        assert_allclose(result[f"{name}_absolute"], np.sqrt(squared), rtol=2e-14)
        assert result[f"{name}_relative"] == 1


def test_manufactured_derivatives_and_distinct_printed_trace_sweeps():
    """Check analytic gradients by complex steps and preserve all 27 distinct settings."""
    points = np.array([[0.13, 0.27], [0.39, 0.83]])
    values, gradient = smooth_field(points)
    assert_allclose(smooth_source(points), 8 * np.pi**2 * values)
    for axis in range(2):
        perturbed = points.astype(complex)
        perturbed[:, axis] += 1e-30j
        assert_allclose(smooth_field(perturbed)[0].imag / 1e-30, gradient[:, axis], rtol=3e-15)
    cases = smooth_configurations(32)
    assert len(cases) == len(set(cases)) == 27
    assert (4, 4) in cases and (3, 32) in cases and (4, 8) not in cases
    assert len(smooth_configurations(16)) == 23
