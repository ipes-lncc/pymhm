"""Independent polynomial verification of the published flux-projection diagnostic."""

from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from examples.helmholtz_stability import projected_solution
from pymhm._legacy.models.waves.helmholtz import solve_helmholtz
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.meshes.cartesian import CartesianMacroMesh


def test_exact_flux_projection_reconstructs_complex_quadratic() -> None:
    """A representable physical normal flux must recover both complex pressure parts."""
    omega = 1.9

    def pressure(points: np.ndarray) -> np.ndarray:
        """Complex quadratic manufactured field."""
        x, y = points.T
        return x**2 + 0.3 * y**2 + x * y + 1j * (x - 2 * y)

    def gradient(points: np.ndarray) -> np.ndarray:
        """Independent analytical gradient."""
        x, y = points.T
        return np.column_stack((2 * x + y + 1j, 0.6 * y + x - 2j))

    def absorbing(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Physical outward Robin datum."""
        return np.sum(gradient(points) * normals, axis=1) - 1j * omega * pressure(points)

    mesh = CartesianMacroMesh(2)
    with threadpool_limits(1):
        solution = solve_helmholtz(
            mesh,
            omega=omega,
            degree=3,
            local_refinement=2,
            skeleton=helmholtz_skeleton(mesh, omega, degree=1),
            source=lambda p: -2.6 - omega**2 * pressure(p),
            absorbing=absorbing,
            quadrature_order=8,
        )
        projected = projected_solution(solution, SimpleNamespace(gradient=gradient))
    assert projected.l2_error(pressure, 8) < 5e-13
    assert projected.gradient_l2_error(gradient, 8) < 3e-12
    np.testing.assert_allclose(projected.trace, solution.trace, atol=3e-12, rtol=0)
