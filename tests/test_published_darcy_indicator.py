"""Literal L09 material scalings and explicit adaptive-convention selection."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.adaptive_darcy import solve_adaptive_darcy
from pymhm.adaptive_darcy_balanced import solve_balanced_adaptive_darcy
from pymhm.darcy import solve_darcy
from pymhm.mesh import TriangleMesh
from pymhm.weighted_estimator import (
    PublishedDarcyIndicator,
    estimate_darcy_indicator,
    estimate_weighted_darcy_error,
)


def data(points):
    """Smooth homogeneous-boundary pressure, independently differentiated below."""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def gradient(points):
    """Analytical pressure gradient for the physical energy comparison."""
    x, y = np.pi * points.T
    return np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def problem(coefficient):
    """Keep the exact pressure fixed while scaling permeability and its forcing."""
    return solve_darcy(
        TriangleMesh.unit_square(),
        degree=2,
        local_refinement=2,
        permeability=coefficient,
        source=lambda x: coefficient * 2 * np.pi**2 * data(x),
        quadrature_order=6,
    )


@pytest.mark.parametrize("coefficient", [0.04, 1.0, 25.0])
def test_literal_material_scaling(coefficient):
    """Printed flux/divergence terms scale with K, while eta2 scales with sqrt(K)."""
    solution = problem(coefficient)
    literal = estimate_darcy_indicator(solution, degree=2, quadrature_order=6)
    weighted = estimate_weighted_darcy_error(solution, degree=2, quadrature_order=6)
    assert isinstance(literal, PublishedDarcyIndicator)
    assert literal.convention == "published"
    assert weighted.convention == "energy"
    assert literal.ellipticity_lower_bounds.size == 0
    for name in ("flux_defect", "divergence_defect", "oscillation"):
        assert_allclose(
            getattr(literal, name),
            np.sqrt(coefficient) * getattr(weighted, name),
            rtol=2e-11,
            atol=1e-12,
        )
    assert_allclose(literal.nonconformity, weighted.nonconformity, rtol=1e-13)
    assert_allclose(literal.energy_error(gradient), weighted.energy_error(gradient), rtol=1e-13)
    assert_allclose(
        literal.local_squared,
        (literal.flux_defect + literal.divergence_defect + literal.oscillation) ** 2
        + literal.nonconformity**2,
        rtol=1e-13,
    )


def test_published_variable_material_requires_no_ellipticity_certificate():
    """The literal terms do not introduce an unprinted coefficient lower bound."""
    solution = solve_darcy(
        TriangleMesh.unit_square(),
        degree=2,
        permeability=lambda x: 0.5 + x[:, 0],
        source=1.0,
        local_refinement=2,
    )
    assert estimate_darcy_indicator(solution).total > 0
    with pytest.raises(ValueError, match="certified"):
        estimate_darcy_indicator(solution, convention="energy")
    with pytest.raises(ValueError, match="convention"):
        estimate_darcy_indicator(solution, convention="unknown")


def test_adaptive_published_convention_does_not_change_the_pde():
    """Both adaptive entry points select the same published terms and physical field."""
    mesh = TriangleMesh.unit_square()
    options = dict(
        iterations=1,
        degree=2,
        source=1.0,
        permeability=0.25,
        local_refinement=2,
        reconstruction_degree=2,
        estimator_order=6,
        estimator_convention="published",
    )
    direct = solve_adaptive_darcy(mesh, **options)
    balanced = solve_balanced_adaptive_darcy(mesh, **options)
    assert direct.estimators[0].convention == "published"
    assert balanced.result.estimators[0].convention == "published"
    assert_allclose(direct.totals, balanced.result.totals, rtol=1e-13)
    assert_allclose(
        direct.solutions[0].hybrid.trace, balanced.result.solutions[0].hybrid.trace, rtol=1e-13
    )
