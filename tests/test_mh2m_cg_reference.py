"""Exact field invariants and native checks for the independent cubic reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.mh2m_cg_reference import (
    CubicTriangularField,
    coefficient,
    patch_gradient,
    patch_pressure,
    solve,
)


@pytest.mark.parametrize("n", [1, 3])
def test_cubic_field_reproduces_independent_polynomial(n):
    """Both triangle orientations reproduce a cubic and its analytical gradient."""
    x, y = np.meshgrid(np.linspace(0, 1, 3 * n + 1), np.linspace(0, 1, 3 * n + 1))
    field = CubicTriangularField(
        patch_pressure(np.column_stack((x.ravel(), y.ravel()))).reshape(x.shape)
    )
    points = np.random.default_rng(301).random((71, 2))
    p, g = field.evaluate(points)
    assert_allclose(p, patch_pressure(points), rtol=0, atol=3e-15)
    assert_allclose(g, patch_gradient(points), rtol=0, atol=4e-14)
    constant = CubicTriangularField(np.full(x.shape, 1e9))
    p, g = constant.evaluate(points)
    assert np.array_equal(p, np.full(len(points), 1e9))
    assert np.array_equal(g, np.zeros_like(points))


def test_incident_gradients_and_archive_contract(tmp_path):
    """A continuous pressure retains separate gradients on a horizontal fine interface."""
    y = np.linspace(0, 1, 7)
    field = CubicTriangularField(np.broadcast_to(abs(y[:, None] - 0.5), (7, 7)).copy())
    xy = np.column_stack((np.linspace(0, 1, 11), np.full(11, 0.5)))
    for side in (-1, 1):
        p, g = field.evaluate(xy, y_side=side)
        # Cubic cardinal evaluation subtracts values of magnitude one half.
        assert_allclose(p, 0, rtol=0, atol=8 * np.finfo(float).eps)
        assert_allclose(g, np.tile([0, side], (11, 1)), atol=3e-15)
    path = tmp_path / "reference.npz"
    np.savez(path, pressure=field.pressure, degree=3, diagonal="SW-NE")
    assert np.array_equal(CubicTriangularField.load(path).pressure, field.pressure)
    np.savez(path, pressure=field.pressure, degree=2, diagonal="SW-NE")
    with pytest.raises(ValueError, match="archive"):
        CubicTriangularField.load(path)
    with pytest.raises(ValueError, match="lattice"):
        CubicTriangularField(np.ones((5, 5)))
    with pytest.raises(ValueError, match="unit-square"):
        field.evaluate(np.array([[0.1, 1.1]]))


def test_coefficient_matches_independently_declared_case():
    """Native-reference data match the published-case callback without rescaling."""
    from examples.mh2m_heterogeneous import OscillatoryCoefficient

    points = np.random.default_rng(39).random((107, 2))
    assert_allclose(coefficient(points), OscillatoryCoefficient()(points), rtol=3e-14, atol=2e-13)


def test_cubic_reference_uses_shared_physical_norms():
    """Cubic pressure and quadratic gradient norms agree with analytical integration."""
    from examples.mh2m_heterogeneous_norms import difference

    x, _ = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    reference = CubicTriangularField(1 + x**3)
    zero = CubicTriangularField(np.zeros_like(x))
    norms = difference(reference, zero, 1.0, order=8)
    assert_allclose(norms["reference_pressure_norm"], np.sqrt(23 / 14), rtol=2e-15)
    assert_allclose(norms["reference_gradient_norm"], np.sqrt(9 / 5), rtol=2e-15)
    for name in ("pressure", "flux", "gradient", "energy"):
        assert_allclose(norms[f"{name}_relative_difference"], 1, atol=0, rtol=0)


@pytest.mark.fem
@pytest.mark.parametrize("patch", [False, True])
def test_native_reference_equations_and_field_reader(patch):
    """Independent native values and gradients agree with the persisted physical field."""
    pytest.importorskip("dolfinx")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        _, report = solve(2, 12, patch=patch)
    assert report["relative_equation_residual"] < 2e-13
    assert report["native_replay_pressure_linf"] < 2e-14
    assert report["native_replay_gradient_linf"] < 2e-13
    if patch:
        assert report["patch_pressure_linf"] < 2e-13
        assert report["patch_gradient_linf"] < 3e-12
    else:
        assert report["relative_energy_work_defect"] < 2e-13
        assert report["native_material_difference_linf"] < 2e-12
