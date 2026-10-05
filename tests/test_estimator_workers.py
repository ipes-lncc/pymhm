"""Parallel moment reconstruction and indicators preserve the serial equations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.adaptivity.darcy import solve_adaptive_darcy
from pymhm.estimators.darcy_energy import estimate_darcy_indicator, estimate_weighted_darcy_error
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh
from pymhm.recovery.moments import reconstruct_darcy_moments


def boundary(points):
    """Provide represented quadratic boundary data with a picklable callback."""
    return points[:, 0] ** 2 + points[:, 1] ** 2


@pytest.mark.parametrize("backend", ["thread", "process"])
@pytest.mark.parametrize("convention", ["published", "energy"])
def test_cartesian_indicator_and_moment_parallel_equivalence(backend, convention):
    """Preserve flux moments and estimator terms, and check equilibrium in each execution."""
    mesh = TriangleMesh.unit_square(2)
    material = CartesianCellField(
        np.array([[1.0, 2.0], [5.0, 3.0], [2.0, 4.0]]), spacing=(1 / 3, 1 / 2)
    )
    solution = solve_darcy(
        mesh,
        permeability=material,
        source=1.0,
        dirichlet=boundary,
        degree=2,
        local_refinement=2,
        quadrature_order=5,
    )
    kwargs = dict(degree=2, convention=convention, dirichlet=boundary, quadrature_order=5)
    original = estimate_darcy_indicator(solution, **kwargs)
    parallel = estimate_darcy_indicator(solution, **kwargs, backend=backend, workers=2)
    for name in (
        "flux_defect",
        "nonconformity",
        "divergence_defect",
        "oscillation",
    ):
        assert_allclose(getattr(parallel, name), getattr(original, name), rtol=2e-14, atol=1e-14)
    for estimator in (original, parallel):
        assert_allclose(estimator.equilibrium_defect, 0, rtol=0, atol=1e-12)
    for actual, expected in zip(
        parallel.reconstructed_flux.flux, original.reconstructed_flux.flux, strict=True
    ):
        assert np.linalg.norm(actual - expected) <= 1e-14 + 2e-14 * np.linalg.norm(expected)


def test_public_wrapper_and_adaptive_dispatch():
    """Convenience APIs forward the requested estimator execution contract."""
    mesh = TriangleMesh.unit_square()
    solution = solve_darcy(mesh, source=1, degree=2, local_refinement=2)
    flux = reconstruct_darcy_moments(solution, degree=2, backend="process", workers=1)
    assert max(np.linalg.norm(x) for x in flux.continuous_moment_residuals()) < 1e-12
    weighted = estimate_weighted_darcy_error(solution, degree=2, backend="thread", workers=2)
    adaptive = solve_adaptive_darcy(
        mesh,
        source=1,
        degree=2,
        local_refinement=2,
        iterations=1,
        reconstruction_degree=2,
        estimator_backend="thread",
        estimator_workers=2,
    )
    assert_allclose(adaptive.estimators[0].local_squared, weighted.local_squared, rtol=2e-13)
