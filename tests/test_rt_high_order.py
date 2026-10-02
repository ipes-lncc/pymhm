"""High-order RT moments, orientation, commuting divergence and reconstructed fields."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.darcy_rt import solve_darcy_rt, solve_darcy_rt_conforming
from pymhm.elements import triangle_quadrature
from pymhm.reconstruction_moments import reconstruct_darcy_moments
from pymhm.rt import rt_evaluate, rt_interpolate


@pytest.mark.parametrize("degree", [3, 4, 5, 6])
def test_high_order_radial_polynomial_and_divergence(degree):
    """Bernstein coordinates and orthonormal cell tests retain the complete RT space."""
    mesh = TriangleMesh.unit_square()
    bary, _ = triangle_quadrature(degree + 3)
    physical = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])

    def exact(x):
        """Use a homogeneous radial enrichment beyond vector Pk."""
        return x * (x[:, 0] ** degree + x[:, 1] ** degree)[:, None]

    coefficients = rt_interpolate(mesh, exact, degree, degree + 3)
    values, divergence = rt_evaluate(mesh, coefficients, degree, bary)
    target = exact(physical.reshape(-1, 2)).reshape(values.shape)
    assert_allclose(values, target, atol=2e-11, rtol=0)
    expected = (degree + 2) * (physical[..., 0] ** degree + physical[..., 1] ** degree)
    assert_allclose(divergence, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize("classical", [False, True])
def test_high_order_mixed_darcy_conservation(degree, classical):
    """The shared high-order space solves an exactly represented pressure and flux."""
    solve = solve_darcy_rt_conforming if classical else solve_darcy_rt

    def exact(x):
        """A representable pressure with nonzero boundary values."""
        return np.sum(x**degree, axis=1)

    result = solve(
        TriangleMesh.unit_square(),
        degree=degree,
        dirichlet=exact,
        quadrature_order=degree + 3,
        source=lambda x: -degree * (degree - 1) * np.sum(x ** (degree - 2), axis=1),
    )
    assert result.l2_error(exact, degree + 3) < 2e-11
    assert result.flux_l2_error(lambda x: -degree * x ** (degree - 1), degree + 3) < 2e-10
    assert max(np.max(abs(v)) for v in result.fine_equilibrium_residuals()) < 2e-10


def test_rt3_reconstruction_uses_all_four_boundary_moments():
    """Shared canonical RT trace mapping preserves the previously unsupported cubic moment."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(3) for _ in mesh.faces))

    def exact(x):
        """Harmonic quartic with a cubic flux and nonzero cubic edge moments."""
        return x[:, 0] ** 4 - 6 * x[:, 0] ** 2 * x[:, 1] ** 2 + x[:, 1] ** 4

    def flux(x):
        """Differentiate the harmonic quartic independently."""
        a, b = x.T
        return np.column_stack((-4 * a**3 + 12 * a * b * b, 12 * a * a * b - 4 * b**3))

    result = solve_darcy(
        mesh, degree=5, skeleton=skeleton, dirichlet=exact, local_refinement=2, quadrature_order=8
    )
    reconstructed = reconstruct_darcy_moments(result, degree=3, quadrature_order=8)
    assert reconstructed.flux_l2_error(flux, 8) < 2e-10
    assert max(np.max(abs(v)) for v in reconstructed.continuous_moment_residuals()) < 2e-10
    assert max(np.max(abs(v)) for v in reconstructed.normal_flux_residuals()) < 2e-12
